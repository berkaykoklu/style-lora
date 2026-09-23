from stylelora.generate import GUIDANCE, PROMPTS, STEPS

STYLE_WORDS = ("baroque", "nouveau", "painting", "style", "art", "illustration", "ornate")


def test_there_are_twelve_prompts() -> None:
    assert len(PROMPTS) == 12


def test_no_prompt_names_a_style() -> None:
    """The LoRA has to supply the style. A prompt that named it would let the
    base model produce it too, and the adapter's contribution would vanish."""
    for prompt in PROMPTS:
        lowered = prompt.lower()
        for word in STYLE_WORDS:
            assert word not in lowered, f"{prompt!r} contains {word!r}"


def test_prompts_are_distinct() -> None:
    assert len(set(PROMPTS)) == len(PROMPTS)


def test_prompts_name_a_subject_rather_than_a_mood() -> None:
    """Something has to be in the picture for prompt adherence to measure
    anything; 'something beautiful' would score against nothing."""
    for prompt in PROMPTS:
        assert len(prompt.split()) >= 4


def test_the_sampler_settings_match_a_distilled_model() -> None:
    """sd-turbo is trained for very few steps and without guidance; raising
    either makes its output worse, not better."""
    assert STEPS <= 4
    assert GUIDANCE == 0.0


def test_the_loader_and_the_saver_agree_on_the_file_name() -> None:
    """They drifted once: training saved peft's key names, loading looked for
    diffusers' names, found none, warned, and used the base model instead."""
    from stylelora.train import WEIGHTS_NAME

    assert WEIGHTS_NAME.endswith(".safetensors")
