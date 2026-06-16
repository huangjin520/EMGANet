import requests
from pathlib import Path

DOI = "10.1109/JBHI.2025.3546345"

data = requests.get(
    f"https://api.openalex.org/works/https://doi.org/{DOI}",
    timeout=20,
).json()

citations = data["cited_by_count"]
fwci = data["fwci"]

top1 = (
    data["citation_normalized_percentile"]
    .get("is_in_top_1_percent", False)
)

svg = f"""
<svg width="700" height="220"
     xmlns="http://www.w3.org/2000/svg">

<rect width="700"
      height="220"
      rx="15"
      fill="#f6f8fa"/>

<text x="30" y="45"
      font-size="28"
      font-family="Arial"
      font-weight="bold">
EMGANet Impact
</text>

<text x="30" y="90" font-size="20">
Total Citations
</text>
<text x="520" y="90" font-size="20">
{citations}
</text>

<text x="30" y="130" font-size="20">
FWCI
</text>
<text x="520" y="130" font-size="20">
{fwci:.2f}
</text>

<text x="30" y="170" font-size="20">
Top 1% Paper
</text>
<text x="520" y="170" font-size="20">
{"Yes" if top1 else "No"}
</text>

</svg>
"""

OUT_DIR = Path("img")
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(OUT_DIR / "impact_card.svg", "w", encoding="utf-8") as f:
    f.write(svg)

print(f"Citations: {citations}")
print(f"FWCI: {fwci:.2f}")