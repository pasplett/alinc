from alinc.path import DATA_PATH
from torch_geometric.datasets import GNNBenchmarkDataset

def load_dataset(key, *args, **kwargs):
    model_dict = {
        "pattern": GNNBenchmarkDataset(
            root=DATA_PATH,
            name="PATTERN",
            *args, **kwargs
        ),
        "cluster": GNNBenchmarkDataset(
            root=DATA_PATH,
            name="CLUSTER",
            *args, **kwargs
        )
    }
    return model_dict[key.lower()]