import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from torch_geometric.data import Data

from dal_toolbox_graph.graph_data import (
    GraphActiveLearningDataModule,
    QueryDataset,
    RelabeledDataset,
    list_diff,
    list_union,
)


def make_graph_dataset(n=6):
    return [
        Data(
            x=torch.full((2, 1), float(i)),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            y=torch.tensor(i % 2),
        )
        for i in range(n)
    ]


def test_random_init_moves_samples_between_labeled_and_unlabeled_pools():
    dataset = make_graph_dataset()
    dm = GraphActiveLearningDataModule(dataset, train_batch_size=2, predict_batch_size=2, seed=0)

    dm.random_init(2)

    assert len(dm.labeled_indices) == 2
    assert len(dm.unlabeled_indices) == 4
    assert set(dm.labeled_indices).isdisjoint(dm.unlabeled_indices)

    state = dm.state_dict()
    restored = GraphActiveLearningDataModule(dataset)
    restored.load_state_dict(state)
    assert set(restored.labeled_indices) == set(dm.labeled_indices)
    assert set(restored.unlabeled_indices) == set(dm.unlabeled_indices)


def test_update_annotations_rejects_duplicate_and_already_labeled_indices():
    dm = GraphActiveLearningDataModule(make_graph_dataset())

    with pytest.raises(ValueError, match="duplicate"):
        dm.update_annotations([1, 1])

    dm.update_annotations([1])
    with pytest.raises(ValueError, match="already"):
        dm.update_annotations([1])


def test_dataloaders_return_expected_indices_and_batches_after_initialization():
    dataset = make_graph_dataset()
    dm = GraphActiveLearningDataModule(dataset, train_batch_size=2, predict_batch_size=3, seed=1)
    dm.update_annotations([0, 2])

    unlabeled_loader, unlabeled_indices = dm.unlabeled_dataloader(subset_size=2)
    labeled_loader, labeled_indices = dm.labeled_dataloader()

    assert len(unlabeled_indices) == 2
    assert set(unlabeled_indices).issubset({1, 3, 4, 5})
    assert set(labeled_indices) == {0, 2}
    assert next(iter(unlabeled_loader))[1].numel() <= 3
    assert next(iter(labeled_loader))[1].tolist() == [0, 2]


def test_train_dataloader_requires_labeled_pool():
    dm = GraphActiveLearningDataModule(make_graph_dataset())

    with pytest.raises(ValueError, match="No instances labeled"):
        dm.train_dataloader()


def test_query_and_relabeled_dataset_wrappers():
    dataset = make_graph_dataset(3)
    query_dataset = QueryDataset(dataset)
    data, idx = query_dataset[2]

    assert idx == 2
    assert torch.equal(data.x, dataset[2].x)

    relabeled = RelabeledDataset([(dataset[0], 0), (dataset[1], 1)], [3, 4])
    assert relabeled[0][1] == 3
    assert relabeled[1][1] == 4

    with pytest.raises(ValueError, match="same length"):
        RelabeledDataset([(dataset[0], 0)], [1, 2])


def test_list_set_helpers_return_expected_members():
    assert set(list_union([1, 2], [2, 3])) == {1, 2, 3}
    assert set(list_diff([1, 2, 3], [2])) == {1, 3}
