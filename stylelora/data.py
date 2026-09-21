"""Pulling one style out of WikiArt, and putting it in a shape the trainer wants.

Both styles come from the same collection, preprocessed the same way, in equal
numbers. Two LoRAs trained on two different sources would differ by source as
well as by style, and there would be no way afterwards to say which.
"""

from __future__ import annotations

import io
import json
import time
import urllib.request
from pathlib import Path

from PIL import Image

DATASET = "huggan/wikiart"
STYLES = ("Baroque", "Art_Nouveau")
PER_STYLE = 20

# What sd-turbo was trained at. Feeding it another size means asking the model
# to work at a scale it has never seen.
SIZE = 512

# WikiArt's style column is a class label; these are its names in order.
STYLE_NAMES = (
    "Abstract_Expressionism",
    "Action_painting",
    "Analytical_Cubism",
    "Art_Nouveau",
    "Baroque",
    "Color_Field_Painting",
    "Contemporary_Realism",
    "Cubism",
    "Early_Renaissance",
    "Expressionism",
    "Fauvism",
    "High_Renaissance",
    "Impressionism",
    "Mannerism_Late_Renaissance",
    "Minimalism",
    "Naive_Art_Primitivism",
    "New_Realism",
    "Northern_Renaissance",
    "Pointillism",
    "Pop_Art",
    "Post_Impressionism",
    "Realism",
    "Rococo",
    "Romanticism",
    "Symbolism",
    "Synthetic_Cubism",
    "Ukiyo_e",
)

ROWS_URL = "https://datasets-server.huggingface.co/rows"
TOTAL_ROWS = 11_320
PAGE = 100


def style_index(name: str) -> int:
    if name not in STYLE_NAMES:
        raise ValueError(f"{name} is not a WikiArt style")
    return STYLE_NAMES.index(name)


def prepare(image: Image.Image, size: int = SIZE) -> Image.Image:
    """Centre-crop to a square, then resize.

    Never squash: a stretched painting teaches a stretched style. Never crop
    from a corner either -- that would cut the same side off every painting,
    which changes the work rather than its frame.
    """
    rgb = image.convert("RGB")
    side = min(rgb.size)
    left = (rgb.width - side) // 2
    top = (rgb.height - side) // 2
    square = rgb.crop((left, top, left + side, top + side))
    return square.resize((size, size), Image.Resampling.LANCZOS)


def fetch(style: str, count: int = PER_STYLE, out: Path | None = None) -> list[Path]:
    """Download `count` images of one style and write them as PNGs.

    The rows endpoint rate-limits under load and clears on its own, so a failed
    page waits and retries rather than aborting a run that is most of the way
    through.
    """
    wanted = style_index(style)
    folder = out or Path("data") / style
    folder.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    offset = 0
    while len(saved) < count and offset < TOTAL_ROWS:
        url = (
            f"{ROWS_URL}?dataset=huggan%2Fwikiart&config=default&split=train"
            f"&offset={offset}&length={PAGE}"
        )
        try:
            rows = json.loads(urllib.request.urlopen(url, timeout=60).read())["rows"]
        except Exception:  # noqa: BLE001 -- a rate limit should pause, not abort
            time.sleep(5)
            continue
        for row in rows:
            if row["row"]["style"] != wanted or len(saved) >= count:
                continue
            try:
                raw = urllib.request.urlopen(row["row"]["image"]["src"], timeout=60).read()
            except Exception:  # noqa: BLE001
                continue
            path = folder / f"{len(saved):02d}.png"
            prepare(Image.open(io.BytesIO(raw))).save(path)
            saved.append(path)
        offset += PAGE
    return saved
