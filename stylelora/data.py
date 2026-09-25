"""Pulling one style out of WikiArt, and putting it in a shape the trainer wants.

Both styles come from the same collection, preprocessed the same way, in equal
numbers. Two LoRAs trained on two different sources would differ by source as
well as by style, and there would be no way afterwards to say which.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

DATASET = "huggan/wikiart"
STYLES = ("Baroque", "Art_Nouveau")

# Fetched per style, then split.
#
# The last HOLDOUT images are never trained on. The style centre is built from
# them instead, because a centre made of the training images rewards an adapter
# for memorising them: reproduce one painting and you sit exactly on the target
# without having learned a style at all. Held-out images make the target
# something the adapter has never seen.
#
# The remaining TRAIN_POOL is what the data axis sweeps inside, so a 20-image
# run trains on a subset of what a 100-image run sees.
#
# The ceiling is Baroque's 466 rows (Art Nouveau has 760, Impressionism 3345 --
# this is an 11,320-row subset of WikiArt, not the whole of it). Raising
# PER_STYLE past 466 would leave the two styles with different counts, which is
# the one thing the comparison is built to avoid.
PER_STYLE = 130
HOLDOUT = 30

TRAIN_POOL = PER_STYLE - HOLDOUT

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

# WikiArt's genre column, turned into captions that say what is in the
# picture and nothing about how it is painted.
#
# One caption for every image was the first design, and it was wrong: with the
# same text on all twenty, the adapter had to carry the content as well as the
# style, and what both styles ended up sharing -- "a painting, not a
# photograph" -- was the only signal strong enough to survive. The two
# adapters became indistinguishable from each other.
#
# A caption that names the subject leaves the adapter only the style to learn.
GENRE_CAPTIONS = {
    "abstract_painting": "an abstract composition",
    "cityscape": "a view of a city",
    "genre_painting": "a scene of everyday life",
    "illustration": "an illustration",
    "landscape": "a landscape",
    "nude_painting": "a nude figure",
    "portrait": "a portrait",
    "religious_painting": "a religious scene",
    "sketch_and_study": "a study of a figure",
    "still_life": "a still life",
    "Unknown Genre": "a painting",
}

GENRE_NAMES = (
    "abstract_painting", "cityscape", "genre_painting", "illustration",
    "landscape", "nude_painting", "portrait", "religious_painting",
    "sketch_and_study", "still_life", "Unknown Genre",
)

# One WikiArt row as the rows endpoint returns it: style, genre and artist are
# class-label integers, image carries the URL.
Row = dict[str, Any]

CAPTIONS_FILE = "captions.json"

# The scan is a hundred sequential requests and takes about ten minutes, while
# what it returns never changes. Kept next to the images rather than in a temp
# directory so a Colab runtime that mounts Drive keeps it too.
SCAN_CACHE = Path("data") / "rows.json"

ROWS_URL = "https://datasets-server.huggingface.co/rows"
TOTAL_ROWS = 11_320
PAGE = 100


def split(folder: Path) -> tuple[list[Path], list[Path]]:
    """The training pool and the held-out images, in fetch order.

    Positional, not random: the folder order is the dataset order, so every
    run in the sweep sees the same paintings in the same order and two runs
    differ by how many rather than by which.
    """
    paths = sorted(folder.glob("*.png"))
    return paths[:TRAIN_POOL], paths[TRAIN_POOL:]


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


def caption_for(genre: int) -> str:
    """What is in the picture, said without naming how it is painted."""
    if not 0 <= genre < len(GENRE_NAMES):
        return GENRE_CAPTIONS["Unknown Genre"]
    return GENRE_CAPTIONS[GENRE_NAMES[genre]]


def choose(rows: list[Row], count: int, seed: int = 0) -> list[Row]:
    """Which of a style's rows to train on.

    **Shuffled, not the first N.** WikiArt is ordered by artist, so taking rows
    in order took one artist's body of work: twenty "Baroque" images that were
    twenty Rembrandts. Shuffling with a fixed seed keeps the nesting -- the
    first 20 of a shuffle are a subset of the first 100 -- while drawing from
    the whole style.

    **Round-robin across artists, not a cap.** A cap was the first design and it
    refused more than it fixed: at twenty images a one-eighth cap needs eight
    painters, and only two of this dataset's sixteen styles have that many.
    Taking one from each artist in turn spreads the set as evenly as the style
    allows and never fails -- a style by a single painter still returns a full
    set, and `concentration` is what says so.

    **No sketches.** Half the Rembrandts were pen studies and etchings on cream
    paper, which teach line and paper rather than a painted style. WikiArt
    labels them in its genre column, so they go by name instead of by eye.
    """
    sketch = GENRE_NAMES.index("sketch_and_study")
    shuffled = [row for row in rows if row["genre"] != sketch]
    random.Random(seed).shuffle(shuffled)

    queues: dict[int, list[Row]] = {}
    for row in shuffled:
        queues.setdefault(row["artist"], []).append(row)

    taken: list[Row] = []
    while len(taken) < count:
        served = False
        for queue in queues.values():
            if len(taken) >= count:
                break
            if queue:
                taken.append(queue.pop(0))
                served = True
        if not served:  # every artist exhausted
            break
    return taken


def concentration(rows: list[Row]) -> float:
    """The largest share any one artist holds.

    A style is only a style if more than one hand made it. At 1.0 the adapter
    would learn a painter, and the label on the result would be wrong in a way
    no separation gate can see -- the first run of this project trained on 466
    Rembrandts labelled "Baroque".
    """
    if not rows:
        raise ValueError("no rows to measure")
    counts = Counter(row["artist"] for row in rows)
    return counts.most_common(1)[0][1] / len(rows)


# Above this, the set is one painter wearing a style's name.
CONCENTRATION_LIMIT = 0.5


def contact_sheet(folder: Path, thumb: int = 200, cols: int = 5) -> Image.Image:
    """Every image in one picture, so the set can be looked at before training.

    The separation gate asks whether two styles differ from each other. It
    cannot ask whether either of them is the style on the label, and the first
    run of this project trained for three hours on two sets that were not --
    every number correct, every number about the wrong thing. Nothing catches
    that except looking.
    """
    paths = sorted(folder.glob("*.png"))
    if not paths:
        raise ValueError(f"no images in {folder}")
    rows = (len(paths) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, rows * thumb), (255, 255, 255))
    for i, path in enumerate(paths):
        with Image.open(path) as image:
            sheet.paste(image.resize((thumb, thumb)), (i % cols * thumb, i // cols * thumb))
    return sheet


def scan() -> list[Row]:
    """Every row's metadata, no images.

    Choosing well means choosing over the whole style rather than over
    whatever came first, so the list has to exist before anything is picked.
    It is metadata only -- a hundred small requests and not one painting.

    Returned rather than fetched per style, because comparing five styles
    would otherwise walk the dataset five times.
    """
    if SCAN_CACHE.exists():
        cached: list[Row] = json.loads(SCAN_CACHE.read_text())
        return cached

    rows: list[Row] = []
    offset = 0
    while offset < TOTAL_ROWS:
        url = (
            f"{ROWS_URL}?dataset=huggan%2Fwikiart&config=default&split=train"
            f"&offset={offset}&length={PAGE}"
        )
        try:
            page = json.loads(urllib.request.urlopen(url, timeout=60).read())["rows"]
        except Exception:  # noqa: BLE001 -- a rate limit should pause, not abort
            time.sleep(5)
            continue
        rows.extend(row["row"] for row in page)
        offset += PAGE

    SCAN_CACHE.parent.mkdir(parents=True, exist_ok=True)
    SCAN_CACHE.write_text(json.dumps(rows))
    return rows


def fetch(
    style: str,
    count: int = PER_STYLE,
    out: Path | None = None,
    rows: list[Row] | None = None,
) -> list[Path]:
    """Download `count` images of one style, write them, and note their subjects.

    Every matching row is listed first and the choice made over the whole set --
    see `choose`. Downloading while scanning is faster and is what produced a
    one-artist training set.

    The rows endpoint rate-limits under load and clears on its own, so a failed
    page waits and retries rather than aborting a scan that is most of the way
    through.
    """
    wanted = style_index(style)
    folder = out or Path("data") / style
    folder.mkdir(parents=True, exist_ok=True)

    matching = [row for row in (rows if rows is not None else scan()) if row["style"] == wanted]
    saved: list[Path] = []
    captions: dict[str, str] = {}
    chosen = choose(matching, count)
    for row in chosen:
        try:
            raw = urllib.request.urlopen(row["image"]["src"], timeout=60).read()
        except Exception:  # noqa: BLE001 -- one dead link is not a failed run
            continue
        path = folder / f"{len(saved):02d}.png"
        prepare(Image.open(io.BytesIO(raw))).save(path)
        captions[path.name] = caption_for(row["genre"])
        saved.append(path)

    (folder / CAPTIONS_FILE).write_text(json.dumps(captions, indent=2))
    if len(saved) < count:
        raise ValueError(f"{style}: asked for {count}, got {len(saved)} -- the style has no more")
    share = concentration(chosen)
    if share > CONCENTRATION_LIMIT:
        top = Counter(row["artist"] for row in chosen).most_common(1)[0][0]
        raise ValueError(
            f"{style}: artist {top} paints {share:.0%} of this set, over the "
            f"{CONCENTRATION_LIMIT:.0%} limit. The adapter would learn a painter, not a "
            f"style, and no measurement downstream could tell the difference."
        )
    return saved


def main() -> None:
    """Fetch some styles and write a contact sheet of each, to be looked at."""
    parser = argparse.ArgumentParser(description="fetch styles and show what arrived")
    parser.add_argument("styles", nargs="+", help=f"any of: {', '.join(STYLE_NAMES)}")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("data"))
    args = parser.parse_args()

    rows = None
    for style in args.styles:
        folder = args.out / style
        have = sorted(folder.glob("*.png"))
        if len(have) < args.count:
            if rows is None:
                print(f"scanning {TOTAL_ROWS} rows once...", flush=True)
                rows = scan()
            have = fetch(style, count=args.count, out=folder, rows=rows)
        sheet = folder.parent / f"{style}.jpg"
        contact_sheet(folder).save(sheet, quality=88)
        print(f"{style:24} {len(have):>3} images -> {sheet}")


if __name__ == "__main__":
    main()
