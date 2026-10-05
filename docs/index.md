---
okf_version: "0.2"
title: mesa-clm
description: "mesa-clm proposes calibrated, ontology-grounded AVUs for the MESA stack with Contrastive LM: rank-first decisions with an abstain anchor over mesa-mcp's OBO/OLS tools, fingerprinted and pre-registered, with provenance next to the mesa-ducklake AVU history."
---

# mesa-clm

**mesa-clm** proposes OBO-grounded AVUs for a dataset the way its sibling
[mesa-anyjev](https://github.com/idss-mesa/mesa-anyjev){target=_blank} does, but asks
[Contrastive LM](https://github.com/Contrastive-LM/CLM){target=_blank} (CLM) instead of AnyJev:
every fit question is one rank over the candidate terms plus a fixed "none of these" anchor,
scored by a frozen Qwen3-8B encoder and CLM's projection heads behind a loopback HTTP server.
A reasoning model only *plans* (which ontologies are in play, what to search); deterministic
code drives [mesa-mcp](https://github.com/idss-mesa/mesa-mcp){target=_blank}'s OBO/OLS tools;
every decision is recorded in a provenance sidecar next to the AVU history kept by
[mesa-ducklake](https://github.com/idss-mesa/mesa-ducklake){target=_blank}. Trust is earned only
through pre-registered, leakage-aware, nested-selection bench cells, and the first release
ships proposed-only.

It is developed by [idss-mesa](https://github.com/idss-mesa){target=_blank} at the University
of New Mexico and released under the MIT license. Status: pre-alpha, milestone M2 done (the
registered framing experiment: `column.ontology_fits` is asked with the F9 framing, and
`term.fits` proposals come from the OLS ranking because no framing qualified for it (K1);
see [Learning and bench](concepts/learning-and-bench.md)), and, by the user's decision after
those results (amendment A6), the closed choices answered by rules with CLM's answers kept for
audit only (see [Decision model](concepts/decision-model.md)), after M1's serving stack,
rank-first annotate pipeline, sidecar and review verbs; proposed-only. Milestone M4 (the
learned tiers) is in its pre-run phase: its analysis plan and code (the probe tier, K2, the
teacher ablation, artifacts and promotion, the citation test, the audits) are written before
any M4 result exists, and no number from it is reported anywhere yet. Pages say which milestone
delivers what they describe, and mark what is still planned.

## Documentation

This site is an [Open Knowledge Format (OKF) v0.2](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md){target=_blank}
knowledge bundle; the corpus is available to agents at [`llms.txt`](llms.txt) and
[`llms-full.txt`](llms-full.txt).

* [Getting started](getting-started/index.md) - Install mesa-clm, annotate and review a first card, and configure it.
* [Concepts](concepts/index.md) - The decision model, tiers and the write policy, provenance and history, learning and the bench, serving.
* [Develop](develop/index.md) - Architecture, milestones and the hermetic test suite.
* [About](about/index.md) - How agents should consume this site, licenses, and the change log.
