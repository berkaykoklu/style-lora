"""Two styles out of one collection, in the shape the trainer wants.

Both styles come from the same dataset, preprocessed the same way, in equal
numbers. Two LoRAs trained on two different sources would differ by source as
well as by style, and there would be no way afterwards to say which.

Ukiyo-e and Baroque, because the pair has to clear two bars at once. Anyone can
tell a Japanese woodblock print from a dark Dutch oil painting, which is what
makes a blind human judgement possible at all. And both of them depict people,
places and scenes -- so the styles differ in *how* they depict, not in whether
they depict anything. A pair like Baroque against Abstract Expressionism would
be more distinct and useless: an adapter that removed every recognisable object
would score well, and this project measures precisely what style costs in
recognisable objects.

The previous source (`huggan/wikiart`) was abandoned after measuring it: its
Baroque is 466 works by one painter, it holds no Ukiyo-e at all, and its best
style pair separates no better than the pair that had already failed.
"""

from __future__ import annotations

import io
import json
import random
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image

DATASET = "keremberke/painting-style-classification"
PARQUET = (
    "https://huggingface.co/api/datasets/keremberke/"
    "painting-style-classification/parquet/full/train/0.parquet"
)

STYLES = ("Ukiyo_e", "Baroque")

# Ukiyo-e is the smaller of the two at 66 works, and the two sets have to match.
# Sixty leaves the holdout intact and still fills a training pool larger than
# the reference implementation trains on.
PER_STYLE = 60
HOLDOUT = 20
TRAIN_POOL = PER_STYLE - HOLDOUT

# What Stable Diffusion 1.5 was trained at. Another size asks the model to work
# at a scale it has never seen.
SIZE = 512

# The dataset ships its labels in a loading script, which `datasets` no longer
# runs, so the order is recorded here instead. It is the order the label
# integers index into.
LABEL_NAMES = (
    "Realism", "Art_Nouveau_Modern", "Analytical_Cubism", "Cubism", "Expressionism",
    "Action_painting", "Synthetic_Cubism", "Symbolism", "Ukiyo_e", "Naive_Art_Primitivism",
    "Post_Impressionism", "Impressionism", "Fauvism", "Rococo", "Minimalism",
    "Mannerism_Late_Renaissance", "Color_Field_Painting", "High_Renaissance", "Romanticism",
    "Pop_Art", "Contemporary_Realism", "Baroque", "New_Realism", "Pointillism",
    "Northern_Renaissance", "Early_Renaissance", "Abstract_Expressionism",
)

CAPTIONS_FILE = "captions.json"
FALLBACK_CAPTION = "a painting"

# Words a caption may not carry.
#
# The captioner is a model, not a fixed table, and it will happily write "an
# ukiyo-e print of a wave" or "a baroque portrait". That hands the style to the
# text encoder, and the adapter is then measured on a job the prompt was already
# doing -- the same mistake as putting the style word in a test prompt, arriving
# through a different door.
#
# Stripped rather than rejected: the rest of the caption is still the subject,
# which is what it is for.
STYLE_WORDS = (
    "ukiyo-e", "ukiyo", "baroque", "woodblock", "woodcut", "japanese", "dutch",
    "rembrandt", "hokusai", "hiroshige", "renaissance", "impressionist",
)

Row = dict[str, Any]
Captioner = Callable[[list[Image.Image]], list[str]]


def sanitise(text: str) -> str:
    """A caption with the style taken out of it.

    Returns the fallback when nothing recognisable is left, because a caption
    reduced to "a of" teaches the adapter less than a caption that admits it
    knows nothing.
    """
    kept = [word for word in text.split() if word.strip(".,'\"").lower() not in STYLE_WORDS]
    cleaned = " ".join(kept).strip(" .,")
    return cleaned if len(cleaned.split()) >= 3 else FALLBACK_CAPTION


def style_index(name: str) -> int:
    if name not in LABEL_NAMES:
        raise ValueError(f"{name} is not a style in {DATASET}")
    return LABEL_NAMES.index(name)


def choose(rows: list[Row], count: int, seed: int = 0) -> list[Row]:
    """Which of a style's rows to train on.

    Shuffled with a fixed seed rather than taken in order. Order in these
    collections tracks the artist, and taking the first N once produced twenty
    "Baroque" images that were twenty Rembrandts. A seeded shuffle also keeps
    the nesting the data axis depends on: the first 20 of a shuffle are a subset
    of the first 60, so a small run trains on a subset of what a large one sees
    rather than on different paintings.

    This dataset carries no artist column, so a set cannot be checked for being
    one painter the way the previous source could. The styles here are distant
    enough that it matters less, but it is a limit and not a solved problem.
    """
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:count]


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


def split(folder: Path) -> tuple[list[Path], list[Path]]:
    """The training pool and the held-out images, in fetch order.

    The last HOLDOUT are never trained on, and the style centre is built from
    them. A centre made of the training images rewards an adapter for memorising
    one: reproduce a painting and you sit exactly on the target without having
    learned a style at all.
    """
    paths = sorted(folder.glob("*.png"))
    return paths[:TRAIN_POOL], paths[TRAIN_POOL:]


def contact_sheet(folder: Path, thumb: int = 200, cols: int = 5) -> Image.Image:
    """Every image in one picture, so a set can be looked at before training.

    The separation gate asks whether two styles differ from each other. It
    cannot ask whether either of them is the style on the label, and an earlier
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


def download(destination: Path) -> Path:
    """The dataset as one parquet file.

    Fetched with urllib rather than through `load_dataset`, which rewrites a
    huggingface.co URL as a repository path and then cannot find it. The
    original loading script no longer runs at all -- `datasets` dropped script
    support -- so the auto-converted parquet is what is left.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        urllib.request.urlretrieve(PARQUET, destination)
    return destination


def write_images(rows: list[Row], folder: Path) -> list[Path]:
    """Square 512px PNGs, numbered in the order they were chosen."""
    folder.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for row in rows:
        raw = row["image"]
        image = raw if isinstance(raw, Image.Image) else Image.open(io.BytesIO(raw["bytes"]))
        path = folder / f"{len(saved):02d}.png"
        prepare(image).save(path)
        saved.append(path)
    return saved


def write_captions(paths: list[Path], folder: Path, caption: Captioner) -> dict[str, str]:
    """One caption per image, naming its subject and not its style.

    The caption carries the content so the adapter is left with the style, which
    is the only thing the two runs are meant to differ by. One shared caption
    was the first design and it was wrong: with the same text on all of them,
    the adapter had to carry the content too, and what both styles shared --
    "a painting, not a photograph" -- was the only signal strong enough to
    survive. The two adapters came out indistinguishable.

    The captioner is passed in. It is a second model, it only runs on the
    machine that has a GPU, and the arithmetic here should be testable without
    either.
    """
    images = [Image.open(p).convert("RGB") for p in paths]
    texts = caption(images)
    if len(texts) != len(paths):
        raise ValueError(f"{len(paths)} images, {len(texts)} captions")
    written = {
        path.name: sanitise(text)
        for path, text in zip(paths, texts, strict=True)
    }
    (folder / CAPTIONS_FILE).write_text(json.dumps(written, indent=2))
    return written
