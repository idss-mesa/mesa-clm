#!/usr/bin/env python3
"""Post-build step: make the rendered mesa-clm site consumable by AI agents.

Run AFTER `zensical build --clean --strict`. It:

1. Mirrors every source Markdown file — OKF v0.2 frontmatter intact — into
   the built site at each page's pretty URL:
       docs/a/b.md        -> site/a/b/index.md      (page URL + "index.md")
       docs/a/index.md    -> site/a/index.md
       docs/index.md      -> site/index.md
   so any agent can turn a page URL into its canonical Markdown by appending
   `index.md` (e.g. https://idss-mesa.github.io/mesa-clm/getting-started/install/index.md).
2. Injects agent-discoverable metadata into each page's <head>:
       <meta name="robots" ...>
       <link rel="alternate" type="text/markdown" href="index.md">
       <meta name="okf:type" | okf:status | okf:trust-tier | okf:generated-at
             | okf:generated-by | okf:stale-after>
3. Writes robots.txt advertising sitemap.xml, /llms.txt, /llms-full.txt, the
   Markdown mirror convention and the agent guide.
4. Deletes the zero-byte objects.inv stub Zensical emits.

The base URL is read from zensical.toml (`site_url`). Needs only PyYAML.

Usage: python scripts/postbuild_agent_surface.py [site_dir]   (from the repository root)
"""

from __future__ import annotations

import html
import re
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"


def site_url() -> str:
    text = (ROOT / "zensical.toml").read_text(encoding="utf-8")
    m = re.search(r'^site_url\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return (m.group(1) if m else "/").rstrip("/") + "/"


def frontmatter(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?", text, re.DOTALL)
    if not m:
        return {}
    try:
        data = yaml.safe_load(m.group(1))
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def trust_tier(fm: dict) -> str:
    """OKF §5.3: unverified | machine-confirmed | human-reviewed."""
    v = fm.get("verified")
    if not v:
        return "unverified"
    entries = v if isinstance(v, list) else [v]
    actors = [str(e.get("by", "")) for e in entries if isinstance(e, dict)]
    if any(a.startswith("human:") for a in actors):
        return "human-reviewed"
    return "machine-confirmed" if actors else "unverified"


def head_block(fm: dict) -> str:
    lines = [
        '<meta name="robots" content="index, follow, max-snippet:-1, '
        'max-image-preview:large, max-video-preview:-1">',
        '<link rel="alternate" type="text/markdown" '
        'title="Markdown source (OKF v0.2 frontmatter)" href="index.md">',
    ]

    def meta(name, value):
        if value:
            lines.append(f'<meta name="{name}" content="{html.escape(str(value), quote=True)}">')

    meta("okf:type", fm.get("type"))
    meta("okf:status", fm.get("status", "stable") if fm else None)
    meta("okf:trust-tier", trust_tier(fm) if fm else None)
    gen = fm.get("generated") or {}
    if isinstance(gen, dict):
        meta("okf:generated-at", gen.get("at"))
        meta("okf:generated-by", gen.get("by"))
    meta("okf:stale-after", fm.get("stale_after"))
    return "\n".join(lines) + "\n"


def dest_for(rel: Path, site: Path) -> Path:
    if rel.name == "index.md":
        return site / rel
    return site / rel.parent / rel.stem / "index.md"


def main():
    site = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "site"
    if not site.is_dir():
        print(f"error: {site} not found — run `zensical build` first", file=sys.stderr)
        sys.exit(2)
    base = site_url()

    # 0. Zensical writes an empty objects.inv (a Sphinx inventory stub). No
    # consumer reads it and a zero-byte file only confuses crawlers, so drop it.
    inv = site / "objects.inv"
    if inv.exists() and inv.stat().st_size == 0:
        inv.unlink()

    # 1. Mirror Markdown sources at pretty URLs.
    mirrored = 0
    fms: dict[Path, dict] = {}
    for path in sorted(DOCS.rglob("*.md")):
        rel = path.relative_to(DOCS)
        if rel.parts[0] in ("assets", "stylesheets"):
            continue
        dest = dest_for(rel, site)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        fms[dest.parent.resolve()] = frontmatter(path)
        mirrored += 1

    # 2. Inject <head> metadata into each page whose directory has a mirror.
    injected = 0
    for htmlfile in sorted(site.rglob("index.html")):
        fm = fms.get(htmlfile.parent.resolve())
        if fm is None:
            continue
        text = htmlfile.read_text(encoding="utf-8")
        if 'rel="alternate" type="text/markdown"' in text:
            continue  # idempotent
        text = text.replace("</head>", head_block(fm) + "</head>", 1)
        htmlfile.write_text(text, encoding="utf-8")
        injected += 1

    # 3. robots.txt — explicitly welcome AI fetchers alongside the blanket allow.
    ai_agents = [
        "Googlebot",
        "Google-Extended",
        "GoogleOther",
        "Google-CloudVertexBot",
        "GPTBot",
        "OAI-SearchBot",
        "ChatGPT-User",
        "ClaudeBot",
        "Claude-User",
        "Claude-SearchBot",
        "PerplexityBot",
        "cohere-ai",
        "Applebot-Extended",
        "CCBot",
        "meta-externalagent",
        "Amazonbot",
        "DuckAssistBot",
        "MistralAI-User",
    ]
    ai_block = "".join(f"User-agent: {a}\nAllow: /\n\n" for a in ai_agents)
    (site / "robots.txt").write_text(
        f"# mesa-clm documentation — {base}\n"
        "# This documentation is published for people AND for AI agents.\n"
        "User-agent: *\n"
        "Allow: /\n"
        "\n" + ai_block + f"Sitemap: {base}sitemap.xml\n"
        "\n"
        "# AI agents and harnesses:\n"
        f"#   Machine-readable outline:  {base}llms.txt\n"
        f"#   Full corpus (one file):    {base}llms-full.txt\n"
        "#   Markdown source of any page (OKF v0.2 frontmatter: type, provenance,\n"
        "#   trust, lifecycle): append `index.md` to the page URL.\n"
        f"#   Agent guide:               {base}about/ai-agents/\n"
        "#   For annotations themselves, install mesa-clm and call its CLI or its\n"
        "#   mesa_clm_* tools inside mesa-mcp rather than scraping these pages.\n",
        encoding="utf-8",
    )

    print(
        f"agent surface: mirrored {mirrored} markdown files, "
        f"annotated {injected} pages, wrote robots.txt"
    )


if __name__ == "__main__":
    main()
