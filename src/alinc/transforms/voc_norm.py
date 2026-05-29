import torch

from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

"""
VOC Superpixels Input Feature Normalization as proposed in:
https://arxiv.org/pdf/2309.00367

Code based on:
https://github.com/toenshoff/LRGB/blob/main/graphgps/encoder/voc_superpixels_encoder.py
"""

class VOCNodeNorm(BaseTransform):
    def __init__(self):
        super().__init__()

        self.node_x_mean = torch.tensor([
            4.5824501e-01, 4.3857411e-01, 4.0561178e-01, 6.7938097e-02,
            6.5604292e-02, 6.5742709e-02, 6.5212941e-01, 6.2894762e-01,
            6.0173863e-01, 2.7769071e-01, 2.6425251e-01, 2.3729359e-01,
            1.9344997e+02, 2.3472206e+02
        ])
        self.node_x_std = torch.tensor([
            2.5952947e-01, 2.5716761e-01, 2.7130592e-01, 5.4822665e-02,
            5.4429270e-02, 5.4474957e-02, 2.6238337e-01, 2.6600540e-01,
            2.7750680e-01, 2.5197381e-01, 2.4986187e-01, 2.6069802e-01,
            1.1768297e+02, 1.4007195e+02
        ])

    def forward(self, data: Data) -> Data:
        data.x = (data.x - self.node_x_mean.view(1, -1)) / self.node_x_std.view(1, -1)
        return data


class VOCEdgeNorm(BaseTransform):
    def __init__(self):
        super().__init__()
        self.edge_x_mean = torch.tensor([0.07640745, 33.73478])
        self.edge_x_std = torch.tensor([0.0868775, 20.945076])

    def forward(self, data: Data) -> Data:
        data.edge_attr = (data.edge_attr - self.edge_x_mean.view(1, -1)) / self.edge_x_std.view(1, -1)
        return data


class COCONodeNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.node_x_mean = torch.tensor([
            4.6977347e-01, 4.4679317e-01, 4.0790915e-01, 7.0808627e-02,
            6.8686441e-02, 6.8498217e-02, 6.7777938e-01, 6.5244222e-01,
            6.2096798e-01, 2.7554795e-01, 2.5910738e-01, 2.2901227e-01,
            2.4261935e+02, 2.8985367e+02
        ])
        self.node_x_std = torch.tensor([
            2.6218116e-01, 2.5831082e-01, 2.7416739e-01, 5.7440419e-02,
            5.6832556e-02, 5.7100497e-02, 2.5929087e-01, 2.6201612e-01,
            2.7675411e-01, 2.5456995e-01, 2.5140920e-01, 2.6182330e-01,
            1.5152475e+02, 1.7630779e+02
        ])

    def forward(self, data: Data) -> Data:
        data.x = (data.x - self.node_x_mean.view(1, -1)) / self.node_x_std.view(1, -1)
        return data


class COCOEdgeNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.edge_x_mean = torch.tensor([0.07848548, 43.68736])
        self.edge_x_std = torch.tensor([0.08902349, 28.473562])

    def forward(self, data: Data) -> Data:
        data.edge_attr = (data.edge_attr - self.edge_x_mean.view(1, -1)) / self.edge_x_std.view(1, -1)
        return data