from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree

class AddNodeDegree(BaseTransform):
    """Adds node degree."""
    def __init__(
            self, 
            attr_name: str = "degree"
        ):
        super().__init__()
        self.attr_name = attr_name

    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        num_nodes = data.num_nodes
        assert num_nodes is not None

        node_degrees = degree(data.edge_index[0], num_nodes)
        setattr(data, self.attr_name, node_degrees)

        return data
    
