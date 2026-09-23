import os
from pathlib import Path

import pytest
from PIL import Image

from stylelora.train import BASE_MODEL, CAPTION, LR, RANK, STEPS, train


def test_the_base_model_is_the_cached_one() -> None:
    """A different id here means a multi-gigabyte download nobody asked for."""
    assert BASE_MODEL == "stabilityai/sd-turbo"


def test_the_rank_is_small_enough_to_be_a_style_adapter() -> None:
    """Rank is capacity. A large one starts memorising paintings rather than
    learning the style they share."""
    assert RANK <= 16


def test_the_caption_names_no_style() -> None:
    """The images carry the style. Saying it in the caption as well would teach
    the model to wait for the word before applying it."""
    for word in ("baroque", "nouveau", "style"):
        assert word not in CAPTION.lower()


def test_both_styles_would_get_the_same_settings() -> None:
    """One comparison, one difference. Tuning per style would add a second."""
    assert isinstance(STEPS, int) and isinstance(LR, float)


def test_an_empty_folder_is_refused_rather_than_trained_on_nothing(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(ValueError):
        train("Test", images=empty, out=tmp_path / "lora", steps=1)


# Loading the real pipeline takes several gigabytes and filled a 16 GB machine
# the first time this suite ran it. It stays out of the default run; set
# STYLELORA_HEAVY=1 to include it, with nothing else heavy open.
heavy = pytest.mark.skipif(
    os.environ.get("STYLELORA_HEAVY") != "1",
    reason="loads the full diffusion pipeline; set STYLELORA_HEAVY=1",
)


@heavy
def test_a_tiny_run_writes_weights(tmp_path: Path) -> None:
    images = tmp_path / "imgs"
    images.mkdir()
    for i in range(2):
        Image.new("RGB", (512, 512), (40 * i, 80, 160)).save(images / f"{i}.png")

    out = train("Test", images=images, out=tmp_path / "lora", steps=2)

    assert out.exists()


def test_training_runs_in_full_precision_on_every_device() -> None:
    """Half precision gave a loss of nan from the first step on a T4: the VAE
    overflows float16's range and every weight after that is ruined."""
    from stylelora.train import _dtype

    for device in ("cuda", "mps", "cpu"):
        assert _dtype(device).itemsize == 4


def test_a_finite_loss_passes_through() -> None:
    import torch

    from stylelora.train import check_finite

    check_finite(torch.tensor(0.42), step=3)


def test_a_nan_loss_stops_the_run() -> None:
    """Half precision produced exactly this, and the run still reported success."""
    import torch

    from stylelora.train import check_finite

    with pytest.raises(RuntimeError, match="not usable"):
        check_finite(torch.tensor(float("nan")), step=3)


def test_an_infinite_loss_stops_the_run() -> None:
    """inf is what nan comes from; catching only nan would let the step that
    created it through."""
    import torch

    from stylelora.train import check_finite

    with pytest.raises(RuntimeError):
        check_finite(torch.tensor(float("inf")), step=0)


def test_the_error_names_the_step_it_failed_on() -> None:
    import torch

    from stylelora.train import check_finite

    with pytest.raises(RuntimeError, match="step 17"):
        check_finite(torch.tensor(float("nan")), step=17)


def test_images_are_encoded_in_chunks_not_all_at_once() -> None:
    """Twenty images through the VAE in one pass filled a 15 GB card before
    the first training step, for a loop that uses one image at a time."""
    from stylelora.data import PER_STYLE
    from stylelora.train import ENCODE_BATCH

    assert ENCODE_BATCH < PER_STYLE


def test_training_noise_comes_from_a_ddpm_schedule() -> None:
    """sd-turbo ships a Euler scheduler built for sampling, which adds noise as
    x + sigma * noise and reads sigma from an inference schedule. Training
    against it put latents at a scale the UNet had never seen, and both styles
    came out as the same smeared texture."""
    from diffusers import DDPMScheduler

    from stylelora.train import training_scheduler

    scheduler = training_scheduler()

    assert isinstance(scheduler, DDPMScheduler)
    # alphas_cumprod is the DDPM forward process; a sampling scheduler has
    # sigmas instead, which is exactly the mix-up this guards against.
    assert hasattr(scheduler, "alphas_cumprod")
    assert not hasattr(scheduler, "sigmas")


def test_training_stays_in_the_noise_range_the_model_samples() -> None:
    """sd-turbo visits [999, 499] at two steps. Training uniformly across the
    whole schedule spent half its gradient where inference never goes, and
    both styles came out as the same warm blur."""
    from stylelora.train import TIMESTEP_FLOOR

    assert TIMESTEP_FLOOR >= 499
    assert TIMESTEP_FLOOR < 999
