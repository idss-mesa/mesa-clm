---
title: "AI agents"
description: "How AI agents should read this documentation bundle: llms.txt, the markdown mirror, trust markers, and where the decisions and facts live."
type: Guide
tags:
  - about
  - agents
  - okf
generated:
  by: "claude/fable-5.1"
  at: "2026-09-29T00:00:00Z"
sources:
  - id: design
    resource: "https://github.com/idss-mesa/mesa-clm/blob/main/DESIGN.md"
    title: "mesa-clm decisions register (DESIGN.md)"
    author: "team:idss-mesa"
  - id: research
    resource: "https://github.com/idss-mesa/mesa-clm/blob/main/RESEARCH.md"
    title: "mesa-clm verified facts (RESEARCH.md)"
    author: "team:idss-mesa"
status: draft
stale_after: "2027-03-31T00:00:00Z"
---

# AI agents

Start from [`llms.txt`](../llms.txt) (an outline with descriptions) or
[`llms-full.txt`](../llms-full.txt) (the whole corpus). Any page URL plus `index.md` returns
that page's Markdown source. Pages without a `verified:` key are unverified (OKF v0.2 §5.3);
`status: draft` pages need review; pages that describe a future milestone say so. For
decisions and facts, prefer `DESIGN.md` (append-only register, pre-registered experiments and
amendments) and `RESEARCH.md` (every fact with its source and `stale_after`) in the
repository; for behaviour, run `mesa-clm` and read its provenance sidecar. Never cite a number
that does not name its results JSON.
