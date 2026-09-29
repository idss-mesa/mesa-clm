"""``CudaEncoder``: Qwen3-8B last-token embeddings in process, for the fallback route (plan §6.6).

The recipe reproduces what CLM's head was trained on and what vLLM's pooling runner serves on
the same weights (RESEARCH.md, encoder model; PR #7's ``MpsEncoder`` recipe plus L2):

* ``transformers.AutoModel`` (the bare ``Qwen3Model``, no ``lm_head``) in bfloat16 at the pinned
  revision, from the local Hugging Face cache only (``local_files_only``; offline);
* token ids from the tokenizer with its default special tokens (a no-op for Qwen3, whose
  post-processor adds nothing), truncated from the **left** to the limit (the tail carries the
  target and the question, PR #6), so ``ids[-limit:]`` exactly as vLLM's ``truncation_side
  "left"`` keeps them;
* right padding with an attention mask, ``last_hidden_state`` (post final norm) at the last
  non-pad token, then L2 normalisation in float32.

``embed(texts)`` is the CLM ``Engine`` embedder contract (``(float32 [n, 4096], tokens)`` plus
``healthy()``; DESIGN D16) with truncation to ``max_len`` from the left. ``encode`` is the general
form the fallback's ``/v1/embeddings`` uses (strings or token-id lists, optional truncation,
either side). One forward pass runs at a time (a lock), in batches of ``batch`` sequences sorted
by length.

torch 2.14 compiles some eager Triton kernels against ``Python.h`` at first use;
:func:`prepare_torch` deregisters those overrides (the aten fallbacks are numerically the same;
mesa-anyjev DESIGN D19) unless ``native_triton=True``. The serve venv's uv CPython ships the
header, so either setting works there. torch and transformers are imported lazily: this module
parses and type-checks without them.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any, Final, Literal

import numpy as np
import numpy.typing as npt

MODEL: Final[str] = "Qwen/Qwen3-8B"
REVISION: Final[str] = "b968826d9c46dd6066d109eabc6255188de91218"
MAX_LEN: Final[int] = 4096
HIDDEN: Final[int] = 4096
L2_EPS: Final[float] = 1e-12

TruncationSide = Literal["left", "right"]
F32 = npt.NDArray[np.float32]
Inputs = Sequence[str] | Sequence[Sequence[int]]


def prepare_torch(*, native_triton: bool = False) -> bool:
    """Deregister torch's Triton eager op overrides when ``torch._native`` exists; True if done."""
    if native_triton:
        return False
    try:
        from torch._native import registry
    except ImportError:  # older torch has no native-op overrides
        return False
    registry.deregister_op_overrides(disable_dsl_names="triton")
    return True


def truncate_ids(ids: Sequence[int], limit: int | None, side: TruncationSide) -> list[int]:
    """``ids`` cut to ``limit`` tokens: the tail for ``left``, the head for ``right``."""
    if limit is None or len(ids) <= limit:
        return list(ids)
    return list(ids[-limit:]) if side == "left" else list(ids[:limit])


class CudaEncoder:
    """The in-process encoder (module docstring). ``device`` may be ``cpu`` for K0 step 3b."""

    def __init__(
        self,
        model: str = MODEL,
        revision: str = REVISION,
        *,
        max_len: int = MAX_LEN,
        device: str = "cuda",
        dtype: str = "bfloat16",
        batch: int = 8,
        native_triton: bool = False,
        local_files_only: bool = True,
    ) -> None:
        if max_len < 1 or batch < 1:
            raise ValueError("max_len and batch must be positive")
        self.model_id = model
        self.revision = revision
        self.max_len = max_len
        self.batch = batch
        self.device = device
        self._lock = threading.Lock()
        self.triton_deregistered = prepare_torch(native_triton=native_triton)

        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch: Any = torch
        self.tokenizer: Any = AutoTokenizer.from_pretrained(
            model, revision=revision, local_files_only=local_files_only
        )
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "left"
        pad = self.tokenizer.pad_token_id
        self.pad_id = int(pad if pad is not None else (self.tokenizer.eos_token_id or 0))
        net = AutoModel.from_pretrained(
            model,
            revision=revision,
            dtype=getattr(torch, dtype),
            local_files_only=local_files_only,
            low_cpu_mem_usage=True,
        )
        self.model: Any = net.to(device).eval()
        hidden = int(getattr(self.model.config, "hidden_size", HIDDEN))
        if hidden != HIDDEN:
            raise ValueError(f"{model} has hidden size {hidden}; CLM heads expect {HIDDEN}")

    # -- the Engine embedder contract ---------------------------------------------------------

    def embed(self, texts: list[str]) -> tuple[F32, int]:
        """``(float32 [n, 4096] L2-normalised, tokens encoded)``, left-truncated to ``max_len``."""
        return self.encode(texts, truncate=self.max_len, side="left")

    def healthy(self) -> bool:
        return self.model is not None

    # -- general form (the fallback's /v1/embeddings) ---------------------------------------------

    def tokenize(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        ids = self.tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"]
        return [int(i) for i in ids]

    def encode(
        self, inputs: Inputs, *, truncate: int | None, side: TruncationSide = "left"
    ) -> tuple[F32, int]:
        """Embed strings or token-id lists. ``truncate=None`` refuses inputs over ``max_len``
        (as vLLM does without ``truncate_prompt_tokens``); an empty input is refused."""
        if truncate is not None and not 0 < truncate <= self.max_len:
            raise ValueError(f"truncate_prompt_tokens must be in 1..{self.max_len}")
        seqs: list[list[int]] = []
        for item in inputs:
            ids = self.tokenize(item) if isinstance(item, str) else [int(t) for t in item]
            ids = truncate_ids(ids, truncate, side)
            if not ids:
                raise ValueError("empty input: nothing to embed")
            if len(ids) > self.max_len:
                raise ValueError(
                    f"input has {len(ids)} tokens, more than the maximum context length "
                    f"{self.max_len}; set truncate_prompt_tokens"
                )
            seqs.append(ids)
        out = np.zeros((len(seqs), HIDDEN), dtype=np.float32)
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]), reverse=True)
        for start in range(0, len(order), self.batch):
            chunk = order[start : start + self.batch]
            vecs = self._forward([seqs[i] for i in chunk])
            out[chunk] = vecs
        return out, sum(len(s) for s in seqs)

    def _forward(self, seqs: list[list[int]]) -> F32:
        torch = self._torch
        width = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for row, seq in enumerate(seqs):
            ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            mask[row, : len(seq)] = 1
        with self._lock, torch.inference_mode():
            hidden = self.model(
                input_ids=ids.to(self.device),
                attention_mask=mask.to(self.device),
                use_cache=False,
            ).last_hidden_state
            last = (mask.sum(dim=1) - 1).to(self.device)
            rows = torch.arange(len(seqs), device=self.device)
            pooled = hidden[rows, last].float()
            pooled = torch.nn.functional.normalize(pooled, dim=-1, eps=L2_EPS)
            return np.asarray(pooled.cpu().numpy(), dtype=np.float32)
