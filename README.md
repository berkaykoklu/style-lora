# style-lora

Two art styles taught to Stable Diffusion with LoRA, and an attempt to measure
what the style costs in prompt adherence.

The measurement is the project. Turning an adapter up locks the style in and, at
some point, stops the model drawing what was asked for — and the interesting part
turned out to be whether the usual way of measuring that can see it at all.

## What came out of it

**CLIP says the adapters cost nothing.** Its prompt score — image-text similarity
against the prompt — sits at or above the base model's at every adapter strength
from 0.2 to 1.0. No trade-off anywhere on the curve.

**A blind human rater says content falls about a point on five.** For one of the
two styles it falls on every prompt where it moved at all (p = 0.008, sign test,
paired per prompt).

So the metric could not see the cost it was built to measure. That is the result.

The styles themselves were learned, and the same rating shows it: each adapter was
rated as its own style far more than the base model was (+2.38 on 15 of 16 prompts
for one, +1.31 on 13 of 16 for the other). One of them also *lowers* the other
style's rating, which is the difference between learning a style and merely
looking more painted.

Every number is in [`findings.json`](findings.json), with where it came from.

### What is not published

The rater's absolute 1-5 levels. One person, who paints, judging against real
Ukiyo-e and real Baroque — an absolute level there is a verdict on forty training
images, not a result about the pipeline. The paired differences are that same
person's scale compared against itself, so a harsh hand moves both ends and leaves
the gap intact. Those are published; the levels are not, and a test enforces it.

## How it is measured

Four things stand between a number and a claim here.

**A word gate, before training.** Can CLIP sort the two styles' held-out images by
a sentence naming each one? If not, nothing measured later means anything. This
pair scores 40 out of 40. The pair this project started with scored 78%.

**A held-out style centre.** The images the score is measured against are twenty
per style that no run trained on. A centre built from the training images rewards
an adapter for memorising one.

**Controls.** Sepia, darkening, blurring and desaturation are applied to the base
model's own output and scored the same way. They know nothing about either style,
so whatever they earn is what the measure gives away for free. They earn about a
tenth of what the adapter does, and blurring moves the score the wrong way.

**Pairing.** Every human comparison is the same prompt rendered by the base model
and by an adapter, differenced per prompt. A subject the model draws badly drags
both down together; two averages would carry that into the answer.

There is also a style axis that owes nothing to subject matter: the direction
between two *sentences* describing the styles, rather than the mean of real
paintings. A painting's CLIP embedding carries what is in it, so a centre built
from portraits scores portraits. A sentence names no subject.

## What went wrong first, and why it is in here

The first version of this trained for hours on two sets of images that were not
the styles on their labels.

WikiArt's Baroque, in the subset being used, is 466 works by a single painter —
half of them pen etchings on cream paper. The Art Nouveau was mostly Russian
genre and landscape painting, with a stock-photo watermark in one of them. Every
number computed on them was correct and about the wrong thing.

Nothing caught it. The separation gate asks whether two sets differ *from each
other*, which Rembrandt etchings and Russian landscapes certainly do; it cannot
ask whether either is its label. The fix was measuring the dataset instead of
trusting it, and then looking at the images — the pipeline now renders a contact
sheet of each training set before anything is trained.

Two more things were wrong at the same time and hid each other: a batch size of
one, which showed the model 500 images where the reference implementation shows
8000, and sd-turbo, whose two-step distillation collapsed everything above half
strength to a single point regardless of rank, data or training length.

## Limits

- **One rater.** This is the largest gap in the result.
- Sixteen prompts per source, forty training images per style.
- The two styles were shown at different strengths, 1.2 and 1.5, because that is
  where each looked right. So this judges two chosen operating points, not two
  styles at one strength — and part of the content cost follows from both being
  above 1.0.
- The CLIP sweep used six prompts; the gate and the rating used larger, different
  sets.

## Using the adapters

Two files, one per style, each a LoRA for `sd-legacy/stable-diffusion-v1-5`. They
are not in this repository — see below.

Through this repository's own helper, which fixes the seeds and settings the
measurements were taken at:

```python
from pathlib import Path
from stylelora.generate import generate

images = generate(Path("runs/Ukiyo_e/pytorch_lora_weights.safetensors"), 1.2,
                  ("a woman holding a lantern",), seed=0)
```

Or with plain diffusers:

```python
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "sd-legacy/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")

pipe.load_lora_weights("runs/Ukiyo_e", weight_name="pytorch_lora_weights.safetensors")
pipe.fuse_lora(lora_scale=1.2)

image = pipe("a woman holding a lantern", num_inference_steps=30, guidance_scale=7.5).images[0]

pipe.unfuse_lora()
pipe.unload_lora_weights()
```

Four things worth knowing, each of which cost time here:

**Strength 1.2 for Ukiyo-e, 1.5 for Baroque.** Those are where each style looked
right to the person who rated them, not a default. Baroque needing more is itself
a finding: CLIP had the two separating about equally at 1.0.

**No style words in the prompt.** Write the subject and nothing else. "A Japanese
woodblock print of a lantern" lets the base model do the work and the adapter's
contribution disappears into the wording.

**Unfuse before loading another.** Fusing writes the adapter into the weights. A
fused adapter left behind stacks onto the next one, and the output looks plausible
either way.

**Check it actually loaded.** `load_lora_weights` warns and carries on with the
base model when the key names do not match, so a silent failure looks like a
working run with a weak adapter:

```python
assert pipe.get_list_adapters().get("unet"), "nothing was loaded"
```

### Where the weights are

Not in this repository: `runs/` is ignored, and the two files are about 50 MB
each. Training them takes under half an hour on one rented GPU —
[`experiments.ipynb`](experiments.ipynb) does it end to end, and writes them to
Drive so a dropped session costs one run rather than both.

## Running it

Training needs a GPU; everything else does not.

```bash
pip install -e .
pytest                                   # 150 tests
python score_form.py results.json        # score a blind rating
```

[`experiments.ipynb`](experiments.ipynb) runs the pipeline end to end on Colab:
data, a look at it, the gates, training, and a look at the output. It mounts
Drive, so a dropped session costs one run rather than all of them.

`web/` is the blind rating form — one HTML file with the images inside it, posting
to a Vercel function. Whoever fills it in needs no account and sends no files.
`key.json`, which says which card came from which adapter, is deliberately not in
this repository: everything under `web/public/` is served, and a served key is not
a blind test.

## Layout

| | |
|---|---|
| `stylelora/data.py` | fetching, sampling, the contact sheet |
| `stylelora/train.py` | LoRA training, following the diffusers reference |
| `stylelora/generate.py` | fixed prompts and seeds at a given strength |
| `stylelora/score.py` | CLIP, the style centre, the text axis |
| `stylelora/separation.py` | the gate that runs before training |
| `stylelora/control.py` | the transforms that must not beat the adapter |
| `form.py`, `score_form.py` | the blind rating form, and scoring it |
| `findings.json` | every published number, with its provenance |
