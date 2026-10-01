# Feature store rebuilt in format 2 (float32 vectors), sparky-1, 2026-10-01

Source: `features_build_m1c.json` (this directory), assembled from the command outputs below
(home directory written as `~`). Diagnostics, not bench cells. The store of `features_build.md`
(format 1) kept float16 vectors only, and scores from it missed X1's pre-registered cross-check
by more than ten times; format 2 keeps the float32 vectors, records the serving lock's vector
recipe and stamps each vector with the `lock_sha` it was embedded under (DESIGN, implementation
note "The feature store"). A format-1 store is refused, not converted, so it was moved aside
(`.local/features-format1/`, git-ignored) and the store rebuilt on the recipe of DESIGN A5
(`encoder_fp` c3b3d5e1a283, `lock_sha` `dd33f9fe…`, vector recipe `ed486dcd…`), right after the
units' one restart.

    uv run mesa-clm features build --snapshot bench/snapshots/2026-09-29.parquet
    uv run mesa-clm features project
    uv run mesa-clm features export-npz --out ~/.mesa/clm/features/c3b3d5e1a283/textcache
    uv run mesa-clm features stats --json

| Item | Value |
|---|---|
| Build | 18:21:05Z to 18:24:49Z; 1,563 texts embedded one per request (batch 1), 0 truncated, 620,336 encoder tokens, **223.7 s** reported (3:44 wall clock), max RSS 197 MiB; the container check before the first write: "pinned image and recipe" |
| Against format 1 | every float32 vector cast to float16 is **bitwise equal** to the format-1 store's float16 vector of the same text (1,563/1,563; float16 digest `e5610655…7c0e`, the digest of both 2026-10-01 format-1 builds); token counts, truncation flags and round-trip cosines equal |
| fp16 round trip | min cosine 0.9999999 (the export gate, ≥ 0.9999) |
| Stamps | `meta`: format `mesa-clm-features/2`, vector recipe `ed486dcd…`, `serving_lock_sha` `dd33f9fe…`; all 1,563 vectors carry `lock_sha` `dd33f9fe…` |
| Projections | `clm-latest` (`clm_model_fp` 78be8c462b2e): 1,563 state and 1,563 action, 3.0 s wall clock |
| TextCache export | `choice_Qwen_Qwen3-8B_4096.npz`, 13,054,678 bytes, sha256 `c6628095…7293`, **byte-identical** to the format-1 export, mode 0600 |
| Store size | 57,421,824 bytes (31,207,424 in format 1: float32 vectors take twice the space) |

Reading:

- The A5 recipe (no network, the API on a unix socket, the KV cache pinned) changed no vector:
  the encoder goldens are bitwise equal too (`serving_m1c.md`), so `encoder_fp` stays
  c3b3d5e1a283, as D5 requires for a change that moves nothing.
- The export is unchanged byte for byte, so the float16 copy CLM's `finetune.py` reads (M7) is
  the same; only X1's offline scoring reads the float32 vectors, and from them the
  pre-registered cross-check passes (`x1_crosscheck.md`).
- The build took 223.7 s where the format-1 builds took 195.2 and 195.9 s
  (`features_build.json`), about 18 ms more per text; the cause is not measured here (the socket
  proxy of DESIGN A5 and the float32 rows written to the store both changed).
