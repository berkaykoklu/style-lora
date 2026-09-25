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

# No artist may supply more than 1/ARTIST_SHARE of a style's training set.
# Eight is loose enough that a style dominated by two or three painters still
# fills up, and tight enough that no single one of them defines it.
ARTIST_SHARE = 8
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

    Three rules, each answering something the first version of this got wrong.

    **Shuffled, not the first N.** WikiArt is ordered by artist, so taking rows
    in order took one artist's body of work: twenty "Baroque" images that were
    twenty Rembrandts. Shuffling with a fixed seed keeps the nesting -- the
    first 20 of a shuffle are a subset of the first 100 -- while sampling across
    the whole style.

    **Capped per artist.** Shuffling alone only fixes where in the list we look.
    If one painter is sixty percent of a style's rows he is still sixty percent
    of the sample, and the adapter learns him rather than the style. No artist
    may supply more than an eighth of the set.

    **No sketches.** Half the Rembrandts were pen studies and etchings on cream
    paper, which teach line and paper, not a painted style. WikiArt labels them
    in its genre column, so they can be dropped by name instead of by eye.
    """
    sketch = GENRE_NAMES.index("sketch_and_study")
    paintings = [row for row in rows if row["genre"] != sketch]

    shuffled = list(paintings)
    random.Random(seed).shuffle(shuffled)

    cap = max(count // ARTIST_SHARE, 1)
    taken: list[Row] = []
    per_artist: Counter[int] = Counter()
    for row in shuffled:
        if len(taken) >= count:
            break
        if per_artist[row["artist"]] >= cap:
            continue
        per_artist[row["artist"]] += 1
        taken.append(row)
    return taken


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


def fetch(style: str, count: int = PER_STYLE, out: Path | None = None) -> list[Path]:
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

    # Scan first, choose second, download third. The scan is metadata only, so
    # listing every row of a style costs a hundred small requests and no images.
    matching: list[Row] = []
    offset = 0
    while offset < TOTAL_ROWS:
        url = (
            f"{ROWS_URL}?dataset=huggan%2Fwikiart&config=default&split=train"
            f"&offset={offset}&length={PAGE}"
        )
        try:
            rows = json.loads(urllib.request.urlopen(url, timeout=60).read())["rows"]
        except Exception:  # noqa: BLE001 -- a rate limit should pause, not abort
            time.sleep(5)
            continue
        matching.extend(row["row"] for row in rows if row["row"]["style"] == wanted)
        offset += PAGE

    saved: list[Path] = []
    captions: dict[str, str] = {}
    for row in choose(matching, count):
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
        artists = len({row["artist"] for row in matching})
        raise ValueError(
            f"{style}: asked for {count}, got {len(saved)}. The style has "
            f"{len(matching)} rows across {artists} artists, and no artist may "
            f"supply more than a {ARTIST_SHARE}th of the set -- so it is too "
            f"concentrated to fill this many. Lower the count or pick another style."
        )
    return saved


def main() -> None:
    """Fetch some styles and write a contact sheet of each, to be looked at."""
    parser = argparse.ArgumentParser(description="fetch styles and show what arrived")
    parser.add_argument("styles", nargs="+", help=f"any of: {', '.join(STYLE_NAMES)}")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("data"))
    args = parser.parse_args()

    for style in args.styles:
        folder = args.out / style
        have = sorted(folder.glob("*.png"))
        if len(have) < args.count:
            have = fetch(style, count=args.count, out=folder)
        sheet = folder.parent / f"{style}.jpg"
        contact_sheet(folder).save(sheet, quality=88)
        print(f"{style:24} {len(have):>3} images -> {sheet}")


if __name__ == "__main__":
    main()
