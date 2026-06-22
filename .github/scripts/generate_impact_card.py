import os
import subprocess
from pathlib import Path

import requests

DOI = "10.1109/JBHI.2025.3546345"
FALLBACK_CITATIONS = 26
FALLBACK_FWCI = 45.73
FALLBACK_STARS = 28


def compact_number(value):
    if value is None:
        return "--"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M".rstrip("0").rstrip(".")
    if value >= 1_000:
        return f"{value / 1_000:.1f}k".rstrip("0").rstrip(".")
    return str(value)


def get_repo_stars():
    repo = os.environ.get("GITHUB_REPOSITORY", "huangjin520/EMGANet")
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "Codex",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        response = requests.get(
            f"https://api.github.com/repos/{repo}",
            headers=headers,
            timeout=20,
        )
        response.raise_for_status()
        return int(response.json().get("stargazers_count", 0))
    except Exception:
        return FALLBACK_STARS


def get_commit_count():
    try:
        result = subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return int(result.stdout.strip())
    except Exception:
        return None


try:
    data = requests.get(
        f"https://api.openalex.org/works/https://doi.org/{DOI}",
        timeout=20,
    ).json()
    citations = data["cited_by_count"]
    fwci = data["fwci"]
except Exception:
    citations = FALLBACK_CITATIONS
    fwci = FALLBACK_FWCI

repo_stars = get_repo_stars()
commit_count = get_commit_count()

stars_text = compact_number(repo_stars)
commits_text = compact_number(commit_count)

svg = f"""<svg width="700" height="220" viewBox="0 0 700 220" xmlns="http://www.w3.org/2000/svg">
  <rect x="0.5" y="0.5" width="699" height="219" rx="8" fill="#ffffff" stroke="#d0d7de" />
  <path d="M8 0.5 H692 A7.5 7.5 0 0 1 699.5 8 V46 H0.5 V8 A7.5 7.5 0 0 1 8 0.5 Z" fill="#f6f8fa" />
  <path d="M8 0.5 H692 A7.5 7.5 0 0 1 699.5 8 V8.5 H0.5 V8 A7.5 7.5 0 0 1 8 0.5 Z" fill="#0969da" />
  <line x1="0.5" y1="46" x2="699.5" y2="46" stroke="#d8dee4" />

  <g font-family="Segoe UI, Arial, sans-serif" font-size="15">
    <circle cx="30" cy="88" r="6" fill="#0969da" fill-opacity="0.12" stroke="#0969da" />
    <text x="48" y="93" fill="#24292f" font-weight="600">Total Citations:</text>
    <text x="488" y="93" fill="#24292f" font-weight="700" text-anchor="end">{citations}</text>

    <line x1="24" y1="107" x2="520" y2="107" stroke="#d8dee4" />
    <circle cx="30" cy="124" r="6" fill="#1f883d" fill-opacity="0.12" stroke="#1f883d" />
    <text x="48" y="129" fill="#24292f" font-weight="600">FWCI:</text>
    <text x="488" y="129" fill="#1f883d" font-weight="700" text-anchor="end">{fwci:.2f}</text>

    <line x1="24" y1="143" x2="520" y2="143" stroke="#d8dee4" />
    <path d="M30 151.5 L32.1 156.1 L37 156.7 L33.4 160 L34.4 164.8 L30 162.3 L25.6 164.8 L26.6 160 L23 156.7 L27.9 156.1 Z" fill="#0969da" fill-opacity="0.14" stroke="#0969da" stroke-linejoin="round" />
    <text x="48" y="165" fill="#24292f" font-weight="600">Total Stars Earned:</text>
    <text x="488" y="165" fill="#0969da" font-weight="700" text-anchor="end">{stars_text}</text>

    <line x1="24" y1="179" x2="520" y2="179" stroke="#d8dee4" />
    <circle cx="30" cy="195" r="5.5" fill="#8250df" fill-opacity="0.12" stroke="#8250df" />
    <path d="M41 195 H30" stroke="#8250df" stroke-width="2" stroke-linecap="round" />
    <text x="48" y="200" fill="#24292f" font-weight="600">Total Commits:</text>
    <text x="488" y="200" fill="#8250df" font-weight="700" text-anchor="end">{commits_text}</text>
  </g>

  <circle cx="604" cy="126" r="44" fill="#f6f8fa" stroke="#d0d7de" />
  <circle cx="604" cy="126" r="32" fill="#ffffff" stroke="#0969da" stroke-width="3" />
  <text x="604" y="122" fill="#0969da" font-family="Segoe UI, Arial, sans-serif" font-size="16" font-weight="800" text-anchor="middle">
    EMGANet
  </text>
  <text x="604" y="139" fill="#57606a" font-family="Segoe UI, Arial, sans-serif" font-size="9" font-weight="700" text-anchor="middle">
    IMPACT
  </text>
  <text x="604" y="187" fill="#57606a" font-family="Segoe UI, Arial, sans-serif" font-size="10" font-weight="600" text-anchor="middle">
    OpenAlex + GitHub
  </text>

  <rect x="548" y="27" width="112" height="24" rx="12" fill="#ddf4ff" stroke="#54aeff" />
  <text x="604" y="43" fill="#0969da" font-family="Segoe UI, Arial, sans-serif" font-size="11" font-weight="700" text-anchor="middle">
    LIVE UPDATE
  </text>
</svg>
"""

OUT_DIR = Path("img")
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(OUT_DIR / "impact_card.svg", "w", encoding="utf-8") as f:
    f.write(svg)

print(f"Citations: {citations}")
print(f"FWCI: {fwci:.2f}")
print(f"Stars: {stars_text}")
print(f"Commits: {commits_text}")
