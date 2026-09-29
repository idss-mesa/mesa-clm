---
title: "License"
description: "mesa-clm is MIT licensed by the Regents of the University of New Mexico; Contrastive LM, Qwen3-8B and the vendored AnyJev metrics are Apache-2.0; NEON data are CC BY 4.0."
type: Policy
tags:
  - about
  - license
generated:
  by: "claude/fable-5.1"
  at: "2026-09-29T00:00:00Z"
sources:
  - id: third-party
    resource: "https://github.com/idss-mesa/mesa-clm/blob/main/THIRD_PARTY.md"
    title: "mesa-clm third-party code and data (THIRD_PARTY.md)"
    author: "team:idss-mesa"
status: draft
stale_after: "2027-03-31T00:00:00Z"
---

# License

mesa-clm is released under the MIT License, Copyright (c) 2026 The Regents of the University
of New Mexico. Contrastive LM (CLM) is Apache-2.0 software; its `schema.py` and `LICENSE` are
vendored byte-identical, its `client.py` only as a test oracle, and four of its open pull
requests are carried as serving patches, all with attribution in `THIRD_PARTY.md`. CLM
reimplements the TypeSafe wire format and is not affiliated with TypeSafe AI or Jev. The CLM
head and the Qwen/Qwen3-8B encoder are Apache-2.0 and are downloaded at serving bootstrap, never
committed. AnyJev's `bench/metrics.py` (Apache-2.0) is vendored byte-identical; `anyjev` itself
is not a dependency. NEON metadata in the fixtures and the bench are CC BY 4.0; OLS fixture
terms are under the OBO Foundry ontologies' own licenses. See `THIRD_PARTY.md` for every
third-party component and its hash.
