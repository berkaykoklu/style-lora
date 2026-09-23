# Running the training on Colab

Training needs a GPU; measuring does not. Colab trains and generates, this
repository scores and reports. The code is the same in both places — Colab
clones it rather than keeping its own copy, so there is nothing to keep in
sync.

**Before you start:** Runtime → Change runtime type → **T4 GPU**.

---

## 1. Check you actually got a GPU

```python
!nvidia-smi --query-gpu=name,memory.total --format=csv
```

Expect something like `Tesla T4, 15360 MiB`. If this errors, the runtime is on
CPU and the rest will take hours instead of minutes.

## 2. Clone and install

```python
!git clone -q https://github.com/berkaykoklu/style-lora
%cd style-lora
!pip install -q -e .
```

## 3. Fetch the training images

```python
from stylelora.data import STYLES, fetch

for style in STYLES:
    print(style, len(fetch(style)))
```

Expect `Baroque 20` and `Art_Nouveau 20`. The rows endpoint rate-limits under
load; `fetch` waits and retries rather than aborting, so a slow run is normal
and a stalled one is not.

## 4. Run the separation gate

Do this before training, not after. If the two styles do not pull apart in CLIP
space then nothing measured later means anything, and the hours spent training
would have been spent producing a curve nobody can trust.

```python
from pathlib import Path
from PIL import Image
from stylelora.score import embed_images
from stylelora.separation import measure

vectors = {
    s: embed_images([Image.open(p) for p in sorted(Path("data", s).glob("*.png"))])
    for s in STYLES
}
result = measure(vectors[STYLES[0]], vectors[STYLES[1]])
print(f"within {STYLES[0]}: {result.within_a:.3f}")
print(f"within {STYLES[1]}: {result.within_b:.3f}")
print(f"between:           {result.between:.3f}")
print(f"margin:            {result.margin:.3f}   separated: {result.separated}")
```

Measured locally: margin **0.098** against a floor of 0.05. If Colab disagrees
by much, say so before going further.

## 5. Time a short run before committing to a long one

```python
!python -m stylelora.train --style Baroque --steps 50
```

Read the `s/step` figure and multiply by 500. If that is more than about twenty
minutes, drop `--steps` rather than waiting — the number is worth knowing
either way, and guessing at it is what filled a laptop's memory twice.

## 6. Train both styles

Identical settings for both. If step five made you lower `--steps`, pass the
same lowered value here for **both** — the comparison only holds while the two
were trained alike.

```python
!python -m stylelora.train --style Baroque
!python -m stylelora.train --style Art_Nouveau
```

## 7. Check the adapter did something

A LoRA that failed to load, or trained on nothing, produces images identical to
the base model — and every number afterwards would be measuring the base model
twice.

```python
from pathlib import Path
from stylelora.generate import generate
from stylelora.score import embed_images

base = embed_images(generate(None, 0.0, seed=0))
lora = embed_images(generate(Path("runs/Baroque/lora.pt"), 1.0, seed=0))
print(f"base vs LoRA similarity: {float((base * lora).sum(axis=1).mean()):.3f}")
```

Well below 1.0 is what you want. Near 1.0 means the adapter is not being
applied; stop and fix that first.

## 8. Bring the results home

The weights are small — a few megabytes each. Everything downstream runs on a
laptop.

```python
!zip -qr runs.zip runs
from google.colab import files
files.download("runs.zip")
```

Unzip it into the repository as `runs/`, then carry on locally with the sweep,
the scoring and the site.
