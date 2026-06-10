import os.path as osp
import pickle
import time
import logging
import torch

from alinc.custom_datasets import ZaretzkiDataset
from alinc.path import DATA_PATH
from torch_geometric.data import Data
from torch_geometric.datasets import GNNBenchmarkDataset, LRGBDataset
from torch_geometric.io import fs
from torch_geometric.transforms import Compose


logger = logging.getLogger(__name__)


def load_dataset(key, *args, **kwargs):
    if key.lower() == "pattern":
        dataset = GNNBenchmarkDataset_Progress(
            root=DATA_PATH,
            name="PATTERN",
            *args, **kwargs
        )
    elif key.lower() == "cluster":
        dataset = GNNBenchmarkDataset_Progress(
            root=DATA_PATH,
            name="CLUSTER",
            *args, **kwargs
        )
    elif key.lower() == "pascalvoc-sp":
        dataset = LRGBDataset_Progress(
            root=DATA_PATH,
            name="PascalVOC-SP",
            *args, **kwargs
        )
    elif key.lower() == "coco-sp":
        dataset = LRGBDataset_Progress(
            root=DATA_PATH,
            name="Coco-SP",
            *args, **kwargs
        )
    elif key.lower() == "zaretzki":
        kwargs.pop("root", None)
        dataset = ZaretzkiDataset(
            root=DATA_PATH,
            **kwargs
        )
    else:
        raise KeyError(f"Dataset '{key}' not found.")
    return dataset


def log_progress(iterable, every=10, desc="Processing"):
    """
    A lightweight, cluster-safe progress logger.
    Logs every `every` steps or at the end.
    """
    total = len(iterable)
    start_time = time.time()
    for i, item in enumerate(iterable, 1):
        yield item
        if i % every == 0 or i == total:
            elapsed = time.time() - start_time
            msg = f"{desc}: {i}" + (f"/{total}" if total else "") + f" done ({elapsed:.1f}s elapsed)"
            print(msg)
            logger.info(msg)


class GNNBenchmarkDataset_Progress(GNNBenchmarkDataset):
    def process(self) -> None:
        if self.name == 'CSL':
            data_list = self.process_CSL()
            self.save(data_list, self.processed_paths[0])
        else:
            inputs = fs.torch_load(self.raw_paths[0])
            for i in range(len(inputs)):
                data_list = [Data(**data_dict) for data_dict in inputs[i]]

                if self.pre_filter is not None:
                    data_list = [d for d in data_list if self.pre_filter(d)]

                if self.pre_transform is not None:
                    data_list = [self.pre_transform(d) for d in log_progress(
                        data_list, every=100, desc="Pre-transform"
                    )]

                self.save(data_list, self.processed_paths[i])


class LRGBDataset_Progress(LRGBDataset):
    def process(self) -> None:
        if self.name == 'pcqm-contact':
            # PCQM-Contact
            self.process_pcqm_contact()
        else:
            if self.name == 'coco-sp':
                # Label remapping for coco-sp.
                # See self.label_remap_coco() func
                label_map = self.label_remap_coco()

            for split in ['train', 'val', 'test']:
                if self.name.split('-')[1] == 'sp':
                    # PascalVOC-SP and COCO-SP
                    with open(osp.join(self.raw_dir, f'{split}.pickle'),
                              'rb') as f:
                        graphs = pickle.load(f)
                elif self.name.split('-')[0] == 'peptides':
                    # Peptides-func and Peptides-struct
                    graphs = fs.torch_load(
                        osp.join(self.raw_dir, f'{split}.pt'))

                data_list = []
                for graph in log_progress(graphs, every=100, desc=f'Processing {split} dataset'):
                    if self.name.split('-')[1] == 'sp':
                        """
                        PascalVOC-SP and COCO-SP
                        Each `graph` is a tuple (x, edge_attr, edge_index, y)
                            Shape of x : [num_nodes, 14]
                            Shape of edge_attr : [num_edges, 2]
                            Shape of edge_index : [2, num_edges]
                            Shape of y : [num_nodes]
                        """
                        x = graph[0].to(torch.float)
                        edge_attr = graph[1].to(torch.float)
                        edge_index = graph[2]
                        y = torch.LongTensor(graph[3])
                    elif self.name.split('-')[0] == 'peptides':
                        """
                        Peptides-func and Peptides-struct
                        Each `graph` is a tuple (x, edge_attr, edge_index, y)
                            Shape of x : [num_nodes, 9]
                            Shape of edge_attr : [num_edges, 3]
                            Shape of edge_index : [2, num_edges]
                            Shape of y : [1, 10] for Peptides-func,  or
                                         [1, 11] for Peptides-struct
                        """
                        x = graph[0]
                        edge_attr = graph[1]
                        edge_index = graph[2]
                        y = graph[3]

                    if self.name == 'coco-sp':
                        for i, label in enumerate(y):
                            y[i] = label_map[label.item()]

                    data = Data(x=x, edge_index=edge_index,
                                edge_attr=edge_attr, y=y)

                    if self.pre_filter is not None and not self.pre_filter(
                            data):
                        continue

                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    data_list.append(data)

                path = osp.join(self.processed_dir, f'{split}.pt')
                self.save(data_list, path)
