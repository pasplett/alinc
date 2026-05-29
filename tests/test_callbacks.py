import pytest

from alinc.callbacks import BestModelCB, EarlyStoppingCB


@pytest.mark.parametrize(
    ("mode", "values", "expected"),
    [
        ("min", [5.0, 4.0, 4.5, 4.6], [False, False, False, True]),
        ("max", [1.0, 2.0, 1.5, 1.4], [False, False, False, True]),
    ],
)
def test_early_stopping_tracks_improvements_and_patience(mode, values, expected):
    callback = EarlyStoppingCB(mode=mode, patience=2)

    assert [callback(value) for value in values] == expected
    assert callback.best == (4.0 if mode == "min" else 2.0)
    assert "Current best value" in callback.info()


@pytest.mark.parametrize(
    ("mode", "values", "expected"),
    [
        ("min", [3.0, 2.0, 2.5], [True, True, False]),
        ("max", [3.0, 4.0, 3.5], [True, True, False]),
    ],
)
def test_best_model_callback_reports_only_strict_improvements(mode, values, expected):
    callback = BestModelCB(mode=mode)

    assert [callback(value) for value in values] == expected
    assert callback.best == (2.0 if mode == "min" else 4.0)


@pytest.mark.parametrize("callback_cls", [EarlyStoppingCB, BestModelCB])
def test_callbacks_reject_unknown_modes(callback_cls):
    with pytest.raises(KeyError):
        callback_cls(mode="median")

