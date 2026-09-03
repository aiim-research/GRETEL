import torch
from os.path import join, exists
from os import makedirs

from src.dataset.generators.base import Generator
from src.dataset.instances.graph import GraphInstance

from torch_geometric.datasets import TUDataset as downloader

class TUDataset(Generator):

    def prepare_data(self):
        base_path = join(self.context.working_store_path, self.dataset_name)
        self.context.logger.info("Dataset Data Path:\t" + base_path)

        if not exists(base_path):
            self.context.logger.info("Creating dataset dir...")
            makedirs(base_path, exist_ok=True)

        # ↓ this downloads, process and caches directly in base_path
        self._pyg_dataset = downloader(
            base_path,
            name=self.dataset_name,
            use_node_attr=True,
            use_edge_attr=True
        )
        self.context.logger.info(f"Loaded {self.dataset_name} with {len(self._pyg_dataset)} graphs.")
        return base_path

    def init(self):
        self.dataset_name = self.local_config['parameters']['alias']
        self.prepare_data()
        self.generate_dataset()

    def generate_dataset(self):
        if not len(self.dataset.instances):
            self.populate()

    def populate(self):
        data = self._pyg_dataset

        num_node_feats = int(data[0].x.size(1)) if hasattr(data[0], "x") and data[0].x is not None else 0
        features_map = {f'attribute_{i}': i for i in range(num_node_feats)}
        self.dataset.node_features_map = features_map

        for gid, g in enumerate(data):
            x = g.x if getattr(g, "x", None) is not None else torch.empty((g.num_nodes, 0))
            ei = g.edge_index  # [2, E]

            adj_matrix = torch.zeros((g.num_nodes, g.num_nodes), dtype=torch.float, device=ei.device)
            adj_matrix[ei[0], ei[1]] = 1.0

            edge_features = None
            if getattr(g, "edge_attr", None) is not None:
                edge_features = g.edge_attr
            elif getattr(g, "edge_weight", None) is not None:
                edge_features = g.edge_weight

            adj_np = adj_matrix.cpu().numpy()
            x_np = x.cpu().numpy() if x.numel() > 0 else None
            edge_np = edge_features.cpu().numpy() if edge_features is not None else None

            label = int(g.y.item()) if getattr(g, "y", None) is not None else None

            self.dataset.instances.append(
                GraphInstance(
                    id=gid,
                    label=label,
                    data=adj_np,
                    graph_features=None,
                    node_features=x_np,
                    edge_features=edge_np
                )
            )
