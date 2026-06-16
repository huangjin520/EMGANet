from pathlib import Path

OUT_DIR = Path("img")
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(OUT_DIR / "impact_card.svg", "w", encoding="utf-8") as f:
    f.write(svg)