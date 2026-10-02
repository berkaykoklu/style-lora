from score_form import latest_per_rater, ratings_by_source

KEY = [
    {"n": 1, "source": "base", "prompt": "a"},
    {"n": 2, "source": "Ukiyo_e", "prompt": "b"},
    {"n": 3, "source": "Baroque", "prompt": "c"},
]


def test_one_rating_per_person_the_fullest_one() -> None:
    """The form posts on every press, so a person part-way through leaves
    several records that are prefixes of each other. Counting them all would
    weight whoever pressed the button most."""
    responses = [
        {"rater": "a", "answers": {"1": {}}},
        {"rater": "a", "answers": {"1": {}, "2": {}}},
        {"rater": "b", "answers": {"1": {}}},
    ]

    kept = latest_per_rater(responses)

    assert len(kept) == 2
    assert {len(r["answers"]) for r in kept} == {2, 1}


def test_records_without_a_rater_are_dropped() -> None:
    """Those predate the id and are the test submissions made while wiring the
    endpoint up."""
    assert latest_per_rater([{"answers": {"1": {}}}, {"rater": "a", "answers": {"1": {}}}]) == [
        {"rater": "a", "answers": {"1": {}}}
    ]


def test_a_rating_is_filed_under_the_adapter_that_drew_the_card() -> None:
    responses = [
        {
            "rater": "a",
            "answers": {
                "1": {"ukiyo": 1, "baroque": 1, "content": 5},
                "2": {"ukiyo": 5, "baroque": 1, "content": 4},
                "3": {"ukiyo": 1, "baroque": 5, "content": 4},
            },
        }
    ]

    collected = ratings_by_source(responses, KEY)

    assert collected["Ukiyo_e"]["ukiyo"] == [5]
    assert collected["Baroque"]["baroque"] == [5]
    assert collected["base"]["ukiyo"] == [1]


def test_a_card_the_key_does_not_know_is_skipped() -> None:
    """A key from a different build would otherwise file answers under the
    wrong adapter and still produce a table."""
    responses = [{"rater": "a", "answers": {"99": {"ukiyo": 5, "baroque": 5, "content": 5}}}]

    assert ratings_by_source(responses, KEY) == {}


def test_a_skipped_question_does_not_become_a_zero() -> None:
    responses = [{"rater": "a", "answers": {"2": {"ukiyo": 4, "content": 3}}}]

    collected = ratings_by_source(responses, KEY)

    assert collected["Ukiyo_e"]["ukiyo"] == [4]
    assert collected["Ukiyo_e"]["baroque"] == []


def test_ratings_given_against_an_older_form_are_dropped() -> None:
    """Rebuilding the form reshuffles it, so an old rating is about a different
    picture than its card number now names -- and nothing in the data says so."""
    from score_form import after

    responses = [
        {"stored": "2026-09-28T22:16:57.000Z", "rater": "old"},
        {"stored": "2026-10-01T11:42:50.000Z", "rater": "new"},
    ]

    kept = after(responses, "2026-10-01")

    assert [r["rater"] for r in kept] == ["new"]


def test_a_response_with_no_timestamp_is_dropped_by_a_cutoff() -> None:
    """The earliest records predate both the rater id and the field; they are
    the oldest of all and must not slip through as unknown."""
    from score_form import after

    assert after([{"rater": "a"}], "2026-10-01") == []


# --- the sign test ----------------------------------------------------------


def test_every_change_in_one_direction_is_unlikely() -> None:
    from score_form import sign_test

    assert sign_test([1] * 8) < 0.01


def test_an_even_split_is_not_evidence() -> None:
    from score_form import sign_test

    assert sign_test([1, 1, 1, 1, -1, -1, -1, -1]) == 1.0


def test_ties_are_dropped_not_counted_as_agreement() -> None:
    """Eight drops out of eight non-ties is stronger evidence than eight drops
    out of sixteen cards, and counting the ties would hide that."""
    from score_form import sign_test

    with_ties = sign_test([-1] * 8 + [0] * 8)
    without = sign_test([-1] * 8)

    assert with_ties == without


def test_no_direction_at_all_is_no_evidence() -> None:
    from score_form import sign_test

    assert sign_test([0, 0, 0]) == 1.0


def test_pairing_is_on_the_prompt_not_the_card() -> None:
    """A prompt the model draws badly drags both the base and the adapter down.
    Comparing two averages would carry that noise into the answer."""
    from score_form import paired

    key = [
        {"n": 1, "source": "base", "prompt": "hard one"},
        {"n": 2, "source": "Ukiyo_e", "prompt": "hard one"},
        {"n": 3, "source": "base", "prompt": "easy one"},
        {"n": 4, "source": "Ukiyo_e", "prompt": "easy one"},
    ]
    responses = [{"rater": "a", "answers": {
        "1": {"ukiyo": 1}, "2": {"ukiyo": 3},
        "3": {"ukiyo": 2}, "4": {"ukiyo": 4},
    }}]

    assert sorted(paired(responses, key, "Ukiyo_e", "ukiyo")) == [2, 2]


def test_a_prompt_with_no_base_to_compare_against_is_skipped() -> None:
    from score_form import paired

    key = [{"n": 1, "source": "Ukiyo_e", "prompt": "lonely"}]
    responses = [{"rater": "a", "answers": {"1": {"ukiyo": 5}}}]

    assert paired(responses, key, "Ukiyo_e", "ukiyo") == []
