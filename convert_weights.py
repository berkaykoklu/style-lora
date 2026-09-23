"""Rewrite weights saved in peft's key naming into the one diffusers reads.

Training itself was correct; only the file it wrote was unreadable by the
loader. Converting costs seconds against fifteen minutes of retraining, and
the run being converted is the one whose numbers are already recorded.

Throwaway: delete once no peft-named weights are left anywhere. It sits at the
repository root because runs/ is ignored, and a migration script nobody can
clone is no use.

    python convert_weights.py [runs-directory]
"""

import sys
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from diffusers.utils.state_dict_utils import convert_state_dict_to_diffusers

from stylelora.data import STYLES
from stylelora.train import WEIGHTS_NAME

root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")

for style in STYLES:
    src = root / style / "lora.pt"
    if not src.exists():
        print(f"{style}: {src} yok, atlandı")
        continue
    peft_sd = torch.load(src, map_location="cpu")
    StableDiffusionPipeline.save_lora_weights(
        save_directory=str(root / style),
        unet_lora_layers=convert_state_dict_to_diffusers(peft_sd),
        safe_serialization=True,
    )
    print(f"{style}: {len(peft_sd)} anahtar → {root / style / WEIGHTS_NAME}")
