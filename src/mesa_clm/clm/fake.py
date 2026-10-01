"""A deterministic encoder and CLM engine for hermetic tests (plan §3 ``clm/fake.py``, §9).

:class:`FakeEncoder` maps a text to a 4096-d L2-normalised vector of hashed character n-grams,
so similar texts get similar vectors and the same text always gets the same vector, on every
platform (``hashlib.blake2b``, never Python's salted ``hash``). Its ``collapse`` knob mixes a
shared component into every vector: at ``collapse=0`` unrelated texts are near-orthogonal, at
``collapse`` close to 1 every vector points the same way, which reproduces upstream issue #15
(different states at cosine 0.958 once a shared question suffix dominates last-token pooling)
so the collapse diagnostic and the state-shuffle controls of X1 can be tested without a GPU.

:class:`FakeClm` is the engine maths of CLM ``engine.py`` over that encoder: ``render.build_pairs``
renders one state text per question and the candidate texts, the fake head projects both sides
(``clm-latest``: a small :func:`mesa_clm.clm.headproj.random_head`; named heads can be added),
``logits = scale * cos / temperature`` and ``render.answer_from_logits`` assembles the answer, so
every probability a test sees went through the vendored contract. ``clm-raw`` scores raw cosines
at ``RAW_SCALE``. Usage mirrors clm-serve: ``billing_units = len(questions)``, ``input_tokens``
counts encoder tokens on cache misses only (the fake keeps the same text cache the server has).
``tests/fakes/clm_transport.py`` serves both over ``httpx.MockTransport``.

Nothing here is a model of language: the fake exists so the pipeline, the provider, the policy
and the sidecar can be exercised end to end with ``method="fake"`` (DESIGN §4.5), never to
produce a number anyone cites.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from mesa_clm import render
from mesa_clm.clm.headproj import RAW_SCALE, HeadProjector, l2, random_head

FAKE_DIM: Final[int] = 4096
DEFAULT_NGRAM: Final[int] = 3
RAW_MODEL: Final[str] = "clm-raw"
LATEST_MODEL: Final[str] = "clm-latest"
_WS = re.compile(r"\s+")

F32 = npt.NDArray[np.float32]


class FakeEncoder:
    """Hashed character n-gram vectors, L2-normalised, with a ``collapse`` knob (module docstring).

    ``embed(texts)`` returns ``(float32 [n, dim], tokens)`` like ``EncoderClient.embed``, where a
    token is a whitespace-separated word (the same rule the fake transport's ``/tokenize`` uses).
    ``truncate(text, max_tokens)`` keeps the tail words, mirroring ``truncation_side: left``.
    """

    def __init__(
        self,
        *,
        dim: int = FAKE_DIM,
        seed: int = 0,
        collapse: float = 0.0,
        ngram: int = DEFAULT_NGRAM,
    ) -> None:
        if dim < 2:
            raise ValueError("dim must be at least 2")
        if not 0.0 <= collapse <= 1.0:
            raise ValueError("collapse must be in [0, 1]")
        if ngram < 1:
            raise ValueError("ngram must be at least 1")
        self.dim = dim
        self.seed = seed
        self.collapse = collapse
        self.ngram = ngram
        self._person = seed.to_bytes(8, "little", signed=True)
        self._shared = self._hashed("\x00shared component\x00")
        self.calls = 0
        self.texts_embedded = 0

    @staticmethod
    def count_tokens(text: str) -> int:
        """The fake tokenizer: whitespace-separated words."""
        return len(text.split())

    @staticmethod
    def truncate(text: str, max_tokens: int) -> str:
        """The tail ``max_tokens`` words (``truncation_side: left``); a no-op when it fits."""
        words = text.split()
        if len(words) <= max_tokens:
            return text
        return " ".join(words[-max_tokens:])

    def _hashed(self, text: str) -> F32:
        """The unit n-gram vector of ``text`` before any collapse mixing."""
        s = " " + _WS.sub(" ", text.strip().lower()) + " "
        n = self.ngram
        grams = [s[i : i + n] for i in range(max(len(s) - n + 1, 0))]
        vec = np.zeros(self.dim, dtype=np.float64)
        if not grams:
            return vec.astype(np.float32)
        idx = np.empty(len(grams), dtype=np.int64)
        sign = np.empty(len(grams), dtype=np.float64)
        for k, g in enumerate(grams):
            h = int.from_bytes(
                hashlib.blake2b(g.encode("utf-8"), digest_size=8, person=self._person).digest(),
                "little",
            )
            idx[k] = h % self.dim
            sign[k] = 1.0 if (h >> 63) & 1 else -1.0
        np.add.at(vec, idx, sign)
        row: F32 = l2(vec[None, :])[0]
        return row

    def vector(self, text: str) -> F32:
        """One text's vector: ``l2(sqrt(1 - c) * hashed + sqrt(c) * shared)`` with ``c = collapse``."""
        v = self._hashed(text).astype(np.float64)
        if self.collapse > 0.0:
            c = self.collapse
            v = math.sqrt(1.0 - c) * v + math.sqrt(c) * self._shared.astype(np.float64)
        row: F32 = l2(v[None, :])[0]
        return row

    def embed(self, texts: Sequence[str]) -> tuple[F32, int]:
        self.calls += 1
        self.texts_embedded += len(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32), 0
        rows = np.stack([self.vector(t) for t in texts])
        return rows, sum(self.count_tokens(t) for t in texts)

    def healthy(self) -> bool:
        return True


class FakeClmError(ValueError):
    """A request the fake engine refuses (unknown model, bad temperature, malformed question);
    the transport maps it to 422 like clm-serve."""


class FakeClm:
    """CLM's engine maths over :class:`FakeEncoder` and small numpy heads (module docstring)."""

    def __init__(
        self,
        encoder: FakeEncoder | None = None,
        *,
        heads: Mapping[str, HeadProjector] | None = None,
        seed: int = 0,
    ) -> None:
        self.encoder = encoder or FakeEncoder(seed=seed)
        self.heads: dict[str, HeadProjector] = dict(heads) if heads is not None else {}
        if LATEST_MODEL not in self.heads:
            self.heads[LATEST_MODEL] = random_head(seed, hidden_size=self.encoder.dim)
        for name, head in self.heads.items():
            if head.cfg.hidden_size != self.encoder.dim:
                raise ValueError(
                    f"head {name!r} expects {head.cfg.hidden_size}-d input, "
                    f"the encoder gives {self.encoder.dim}"
                )
        self._cache: dict[str, F32] = {}
        self.calls = 0

    def reset_cache(self) -> None:
        """Forget every embedded text, so the next request charges ``input_tokens`` again (the
        state a freshly started clm-serve is in)."""
        self._cache.clear()

    def add_head(self, name: str, head: HeadProjector) -> None:
        """Serve another head under ``name`` (what promotion + restart does on the real server)."""
        if name == RAW_MODEL:
            raise ValueError(f"{RAW_MODEL!r} is the raw ablation, not a head")
        if head.cfg.hidden_size != self.encoder.dim:
            raise ValueError("head input width does not match the encoder")
        self.heads[name] = head

    def models(self) -> list[dict[str, Any]]:
        """``clm-latest`` first, promoted heads, then ``clm-raw``, as clm-serve lists them."""
        names = [LATEST_MODEL, *sorted(n for n in self.heads if n != LATEST_MODEL)]
        out = [{"name": n, "kind": "head", "scale": self.heads[n].scale} for n in names]
        out.append({"name": RAW_MODEL, "kind": "raw", "scale": RAW_SCALE})
        return out

    def has(self, model: str) -> bool:
        return model == RAW_MODEL or model in self.heads

    def _vectors(self, texts: Sequence[str]) -> tuple[F32, int]:
        """Vectors for ``texts`` through the text cache; tokens counted on misses only."""
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        tokens = 0
        if missing:
            vecs, tokens = self.encoder.embed(missing)
            for t, v in zip(missing, vecs, strict=True):
                self._cache[t] = v
        if not texts:
            return np.zeros((0, self.encoder.dim), dtype=np.float32), 0
        return np.stack([self._cache[t] for t in texts]), tokens

    def _project(self, model: str, xs: F32, xa: F32) -> tuple[F32, F32, float]:
        if model == RAW_MODEL:
            return l2(xs), l2(xa), RAW_SCALE
        head = self.heads[model]
        return head.project_states(xs), head.project_actions(xa), head.scale

    @staticmethod
    def _check(model_known: bool, model: str, temperature: float) -> None:
        if not model_known:
            raise FakeClmError(f"unknown model {model!r}")
        if not 0.0 < temperature <= 100.0:
            raise FakeClmError("temperature must be in (0, 100]")

    def answer(
        self,
        state: Any,
        questions: Mapping[str, Mapping[str, Any]],
        model: str = LATEST_MODEL,
        temperature: float = 1.0,
    ) -> dict[str, Any]:
        """The ``/v1/systemone`` response body: ``{model, answers, usage}``."""
        self._check(self.has(model), model, temperature)
        if not isinstance(questions, Mapping) or not questions:
            raise FakeClmError("questions must be a non-empty object")
        try:
            pairs = render.build_pairs(state, questions)
        except (ValueError, TypeError, AttributeError) as exc:
            raise FakeClmError(str(exc)) from None
        state_texts = list(dict.fromkeys(s for s, _, _ in pairs.values()))
        action_texts: list[str] = []
        for _, _, texts in pairs.values():
            action_texts.extend(texts)
        xs, tokens_s = self._vectors(state_texts)
        xa, tokens_a = self._vectors(action_texts)
        zs, za, scale = self._project(model, xs, xa)
        state_row = {t: i for i, t in enumerate(state_texts)}
        answers: dict[str, Any] = {}
        k = 0
        for qid, (s_text, keys, texts) in pairs.items():
            cos = za[k : k + len(texts)].astype(np.float64) @ zs[state_row[s_text]].astype(
                np.float64
            )
            k += len(texts)
            logits = (scale * cos / temperature).tolist()
            answers[qid] = render.answer_from_logits(questions[qid], keys, logits)
        self.calls += 1
        return {
            "model": model,
            "answers": answers,
            "usage": {
                "billing_units": len(questions),
                "input_tokens": tokens_s + tokens_a,
                "output_tokens": 0,
            },
        }

    def rank(
        self,
        state: Any,
        candidates: Sequence[str],
        instructions: Any = None,
        model: str = LATEST_MODEL,
        temperature: float = 1.0,
    ) -> list[dict[str, Any]]:
        """``Engine.rank``: the candidates as a Choice keyed ``"0".."n-1"``, best first as
        ``[{rank, candidate, prob}]`` (``rank`` 1-based)."""
        if not candidates or any(not isinstance(c, str) or not c for c in candidates):
            raise FakeClmError("answers must be a non-empty list of non-empty strings")
        q = {
            "type": "choice",
            "instructions": instructions,
            "criteria": {str(i): c for i, c in enumerate(candidates)},
        }
        out = self.answer(state, {"rank": q}, model=model, temperature=temperature)
        probs: dict[str, float] = out["answers"]["rank"]["probabilities"]
        order = sorted(range(len(candidates)), key=lambda i: (-probs[str(i)], i))
        return [
            {"rank": r + 1, "candidate": candidates[i], "prob": probs[str(i)]}
            for r, i in enumerate(order)
        ]
