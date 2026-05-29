import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
Chem = pytest.importorskip("rdkit.Chem")

from alinc.custom_datasets.zaretzki import (
    ALLOWABLE_FEATURES,
    atom_to_feature_vector,
    bond_to_feature_vector,
    safe_index,
)
import alinc.datasets as datasets_module
from alinc.datasets import load_dataset, log_progress


def test_safe_index_falls_back_to_misc_bucket():
    values = ["known", "misc"]

    assert safe_index(values, "known") == 0
    assert safe_index(values, "surprise") == 1


def test_atom_and_bond_feature_extractors_return_ogb_style_vectors():
    mol = Chem.MolFromSmiles("CCO")
    atom = mol.GetAtomWithIdx(0)
    bond = mol.GetBondWithIdx(0)

    atom_features = atom_to_feature_vector(atom)
    bond_features = bond_to_feature_vector(bond)

    assert len(atom_features) == 9
    assert len(bond_features) == 3
    assert atom_features[0] == safe_index(
        ALLOWABLE_FEATURES["possible_atomic_num_list"], atom.GetAtomicNum() + 1
    )
    assert bond_features[0] == safe_index(
        ALLOWABLE_FEATURES["possible_bond_type_list"], str(bond.GetBondType())
    )


def test_load_dataset_rejects_unknown_dataset_names():
    with pytest.raises(KeyError, match="Dataset 'missing' not found"):
        load_dataset("missing")


@pytest.mark.parametrize(
    ("key", "expected_cls", "expected_name"),
    [
        ("pattern", "gnn", "PATTERN"),
        ("cluster", "gnn", "CLUSTER"),
        ("pascalvoc-sp", "lrgb", "PascalVOC-SP"),
        ("coco-sp", "lrgb", "Coco-SP"),
    ],
)
def test_load_dataset_dispatches_known_benchmark_datasets(
    monkeypatch, key, expected_cls, expected_name
):
    calls = []

    class FakeGNNDataset:
        def __init__(self, **kwargs):
            calls.append(("gnn", kwargs))

    class FakeLRGBDataset:
        def __init__(self, **kwargs):
            calls.append(("lrgb", kwargs))

    monkeypatch.setattr(datasets_module, "DATA_PATH", "data-root")
    monkeypatch.setattr(datasets_module, "GNNBenchmarkDataset_Progress", FakeGNNDataset)
    monkeypatch.setattr(datasets_module, "LRGBDataset_Progress", FakeLRGBDataset)

    dataset = load_dataset(key, split="train")

    assert dataset.__class__.__name__ in {"FakeGNNDataset", "FakeLRGBDataset"}
    assert calls == [
        (
            expected_cls,
            {"root": "data-root", "name": expected_name, "split": "train"},
        )
    ]


def test_load_dataset_dispatches_zaretzki_and_ignores_user_root(monkeypatch):
    calls = []

    class FakeZaretzkiDataset:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(datasets_module, "DATA_PATH", "data-root")
    monkeypatch.setattr(datasets_module, "ZaretzkiDataset", FakeZaretzkiDataset)

    dataset = load_dataset("zaretzki", root="ignored", split="test", seed=7)

    assert isinstance(dataset, FakeZaretzkiDataset)
    assert calls == [{"root": "data-root", "split": "test", "seed": 7}]


def test_log_progress_yields_all_items_and_reports_completion(capsys):
    items = list(log_progress([1, 2, 3], every=2, desc="Tiny job"))

    assert items == [1, 2, 3]
    out = capsys.readouterr().out
    assert "Tiny job: 2/3 done" in out
    assert "Tiny job: 3/3 done" in out
