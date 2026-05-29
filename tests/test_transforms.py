import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from torch_geometric.data import Data

from alinc.transforms import AddLapEigendec, AddNodeDegree, AddPageRank
from alinc.transforms.eigendec import eigvec_normalizer, get_lap_decomp_stats
from alinc.transforms.voc_norm import COCONodeNorm, VOCEdgeNorm, VOCNodeNorm


def path_graph():
    return Data(
        x=torch.ones(3, 2),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
    )


def test_add_node_degree_uses_source_node_degrees():
    data = AddNodeDegree()(path_graph())

    assert torch.equal(data.degree, torch.tensor([1.0, 2.0, 1.0]))


def test_add_pagerank_adds_one_score_per_node():
    pytest.importorskip("networkx")

    data = AddPageRank(alpha=0.85, eps=1e-8)(path_graph())

    assert data.ppr.shape == (3,)
    assert torch.isclose(data.ppr.sum(), torch.tensor(1.0), atol=1e-5)


def test_laplacian_eigendecomposition_pads_to_requested_frequency_count():
    data = AddLapEigendec(is_undirected=True, max_freqs=5)(path_graph())

    assert data.EigVals.shape == (3, 5, 1)
    assert data.EigVecs.shape == (3, 5)
    assert torch.isnan(data.EigVals[:, 3:, :]).all()
    assert torch.isnan(data.EigVecs[:, 3:]).all()
    assert torch.allclose(data.EigVals[0, :3, 0], torch.tensor([0.0, 1.0, 3.0]), atol=1e-5)


@pytest.mark.parametrize("normalization", ["L1", "L2", "abs-max", "wavelength", "wavelength-asin", "wavelength-soft"])
def test_eigvec_normalizer_returns_finite_values_for_supported_modes(normalization):
    eigvecs = torch.tensor([[1.0, -1.0], [1.0, 0.0], [1.0, 1.0]])
    eigvals = torch.tensor([0.0, 2.0])

    normalized = eigvec_normalizer(eigvecs, eigvals, normalization=normalization)

    assert normalized.shape == eigvecs.shape
    assert torch.isfinite(normalized).all()


def test_lap_decomp_stats_sorts_and_repeats_eigenvalues():
    evals = np.array([2.0, 0.0, 1.0])
    evects = np.eye(3)

    eigvals, eigvecs = get_lap_decomp_stats(evals, evects, max_freqs=2)

    assert eigvals.shape == (3, 2, 1)
    assert eigvecs.shape == (3, 2)
    assert torch.equal(eigvals[0, :, 0], torch.tensor([0.0, 1.0], dtype=eigvals.dtype))


def test_voc_and_coco_normalizers_apply_stored_statistics():
    voc_node_norm = VOCNodeNorm()
    voc_edge_norm = VOCEdgeNorm()
    coco_node_norm = COCONodeNorm()

    voc_data = Data(
        x=voc_node_norm.node_x_mean.view(1, -1).clone(),
        edge_attr=voc_edge_norm.edge_x_mean.view(1, -1).clone(),
    )
    coco_data = Data(x=coco_node_norm.node_x_mean.view(1, -1).clone())

    assert torch.allclose(voc_node_norm(voc_data).x, torch.zeros(1, 14))
    assert torch.allclose(voc_edge_norm(voc_data).edge_attr, torch.zeros(1, 2))
    assert torch.allclose(coco_node_norm(coco_data).x, torch.zeros(1, 14))

