import os
from pathlib import Path

import pytest
from PIL import Image

from stylelora.data import FALLBACK_CAPTION
from stylelora.train import BASE_MODEL, LR, RANK, STEPS, train


def test_the_base_model_is_the_one_the_recipe_was_written_for() -> None:
    """The deprecated `runwayml/stable-diffusion-v1-5` was removed from the Hub;
    this is the mirror its own model card points at."""
    assert BASE_MODEL == "sd-legacy/stable-diffusion-v1-5"


def test_the_rank_is_in_the_range_the_reference_trains_styles_at() -> None:
    """Rank is capacity. The reference trains a style at 62; a rank in single
    figures was one of the settings that made the first adapters barely move."""
    assert 16 <= RANK <= 64


def test_the_fallback_caption_names_no_style() -> None:
    """Captions say what is in the picture, never how it is painted. The rest
    of this rule lives in test_data, where the sanitiser is."""
    for word in ("baroque", "ukiyo", "style", "woodblock"):
        assert word not in FALLBACK_CAPTION.lower()


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


def test_half_precision_only_where_there_is_a_scaler_for_it() -> None:
    """Half precision alone gave a loss of nan from the first step: the VAE
    overflows float16's range. Abandoning it entirely was an over-correction --
    it halves what fits in a batch -- so it is used the way the reference does,
    with a GradScaler, which exists on CUDA and not on MPS."""
    from stylelora.train import _dtype

    assert _dtype("cuda").itemsize == 2
    assert _dtype("mps").itemsize == 4
    assert _dtype("cpu").itemsize == 4


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


def test_the_batch_is_not_one() -> None:
    """A batch of one was the largest single divergence from the reference:
    500 steps showed the model 500 images where the reference shows 8000, and
    the adapters that came out of it barely moved."""
    from stylelora.train import ACCUMULATION, BATCH

    assert BATCH * ACCUMULATION >= 8


def test_the_learning_rate_scales_with_the_batch() -> None:
    """A bigger batch is a steadier gradient and can take a longer stride.
    Leaving the rate fixed would make changing the batch quietly change how
    hard each step pulls, which is a second difference in a one-difference
    comparison."""
    from stylelora.train import ACCUMULATION, BATCH, LR

    assert pytest.approx(8e-5) == LR * BATCH * ACCUMULATION


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


def test_training_samples_the_whole_noise_schedule() -> None:
    """The floor this replaces was a patch for sd-turbo, which visits only two
    timesteps at inference. SD 1.5 visits the whole range, so a floor would now
    leave half of it untrained -- the opposite of the bug it was added for."""
    import stylelora.train as trainer

    assert not hasattr(trainer, "TIMESTEP_FLOOR")


def test_the_near_clean_end_of_the_schedule_is_weighted_down() -> None:
    """Every timestep is a different task and they do not contribute equally.
    Weighting by min(SNR, gamma)/SNR holds back the steps where the latent is
    mostly signal, and leaves the noisy ones at full weight."""
    import torch

    from stylelora.train import SNR_GAMMA, snr_weights, training_scheduler

    scheduler = training_scheduler()
    near_clean = snr_weights(scheduler, torch.tensor([10]), SNR_GAMMA)
    noisy = snr_weights(scheduler, torch.tensor([900]), SNR_GAMMA)

    # The near-clean end is where the epsilon objective produces outsized
    # gradients, so that is the end held back.
    assert float(near_clean) < 0.1
    assert float(noisy) == pytest.approx(1.0)


def test_the_base_model_is_not_a_distilled_one() -> None:
    """sd-turbo is distilled to two steps, and everything it produced above
    half strength collapsed to one point regardless of rank or data."""
    from stylelora.train import BASE_MODEL

    assert "turbo" not in BASE_MODEL
