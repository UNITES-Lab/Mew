import os
import pickle
import numpy as np
import multiprocessing
import matplotlib
import matplotlib.pyplot as plt
import warnings

import torch
from torch.utils.data import RandomSampler

import torch_geometric as tg
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph

from src.graph_build import plot_graph
from src.features import get_feature_names, nx_to_tg_graph, nx_to_tg_hetero_graph
from src.utils import (
    EDGE_TYPES,
    get_cell_type_metadata,
    get_biomarker_metadata,
)


class CellularGraphDataset(Dataset):
    """ Main dataset structure for cellular graphs
    Inherited from https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Dataset.html
    """
    def __init__(self,
                 root,
                 transform=[],
                 pre_transform=None,
                 raw_folder_name='graph',
                 processed_folder_name='tg_graph',
                 node_features=["cell_type", "expression", "neighborhood_composition", "center_coord"],
                 edge_features=["edge_type", "distance"],
                 cell_type_mapping=None,
                 cell_type_freq=None,
                 biomarkers=None,
                 subgraph_size=0,
                 subgraph_source='on-the-fly',
                 subgraph_allow_distant_edge=True,
                 subgraph_radius_limit=-1,
                 sampling_avoid_unassigned=True,
                 unassigned_cell_type='Unassigned',
                 **feature_kwargs):
        """ Initialize the dataset

        Args:
            root (str): path to the dataset directory
            transform (list): list of transformations (see `transform.py`),
                applied to each output graph on-the-fly
            pre_transform (list): list of transformations, applied to each graph before saving
            raw_folder_name (str): name of the sub-folder containing raw graphs (gpickle)
            processed_folder_name (str): name of the sub-folder containing processed graphs (pyg data object)
            node_features (list): list of all available node feature items,
                see `features.process_features` for details
            edge_features (list): list of all available edge feature items
            cell_type_mapping (dict): mapping of unique cell types to integer indices,
                see `utils.get_cell_type_metadata`
            cell_type_freq (dict): mapping of unique cell types to their frequencies,
                see `utils.get_cell_type_metadata`
            biomarkers (list): list of biomarkers,
                see `utils.get_expression_biomarker_metadata`
            subgraph_size (int): number of hops for subgraph, 0 means using the full cellular graph
            subgraph_source (str): source of subgraphs, one of 'on-the-fly', 'chunk_save'
            subgraph_allow_distant_edge (bool): whether to consider distant edges
            subgraph_radius_limit (float): radius (distance to center cell in pixel) limit for subgraphs,
                -1 means no limit
            sampling_avoid_unassigned (bool): whether to avoid sampling cells with unassigned cell type
            unassigned_cell_type (str): name of the unassigned cell type
            feature_kwargs (dict): optional arguments for processing features
                see `features.process_features` for details
        """
        self.root = root
        self.raw_folder_name = raw_folder_name
        self.processed_folder_name = processed_folder_name
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        # Find all unique cell types in the dataset
        if cell_type_mapping is None or cell_type_freq is None:
            try:
                with open(self.raw_dir+'/cell_type_mapping_freq.pkl', 'rb') as f:
                    self.cell_type_mapping, self.cell_type_freq = pickle.load(f)
            
            except:
                nx_graph_files = [os.path.join(self.raw_dir, f) for f in self.raw_file_names]
                self.cell_type_mapping, self.cell_type_freq = get_cell_type_metadata(nx_graph_files)
                
                with open(self.raw_dir+'/cell_type_mapping_freq.pkl', 'wb') as f:
                    pickle.dump([self.cell_type_mapping, self.cell_type_freq], f)

        else:
            self.cell_type_mapping = cell_type_mapping
            self.cell_type_freq = cell_type_freq

        # Find all available biomarkers for cells in the dataset
        if biomarkers is None:
            try:
                with open(self.raw_dir+'/biomarkers.pkl', 'rb') as f:
                    self.biomarkers = pickle.load(f)
            except:
                nx_graph_files = [os.path.join(self.raw_dir, f) for f in self.raw_file_names]
                self.biomarkers, _ = get_biomarker_metadata(nx_graph_files)
                
                with open(self.raw_dir+'/biomarkers.pkl', 'wb') as f:
                    pickle.dump(self.biomarkers, f)

        else:
            self.biomarkers = biomarkers

        # Node features & edge features
        self.node_features = node_features
        self.edge_features = edge_features
        if "cell_type" in self.node_features:
            assert self.node_features.index("cell_type") == 0, "cell_type must be the first node feature"
        if "edge_type" in self.edge_features: # distant, neighboring based on 20 micrometer
            assert self.edge_features.index("edge_type") == 0, "edge_type must be the first edge feature"

        self.node_feature_names = get_feature_names(node_features,
                                                    cell_type_mapping=self.cell_type_mapping,
                                                    biomarkers=self.biomarkers)
        self.edge_feature_names = get_feature_names(edge_features,
                                                    cell_type_mapping=self.cell_type_mapping,
                                                    biomarkers=self.biomarkers)

        # Prepare kwargs for node and edge featurization
        self.feature_kwargs = feature_kwargs
        self.feature_kwargs['cell_type_mapping'] = self.cell_type_mapping
        self.feature_kwargs['biomarkers'] = self.biomarkers

        # Note this command below calls the `process` function
        super(CellularGraphDataset, self).__init__(root, None, pre_transform)

        # Transformations, e.g. masking features, adding graph-level labels
        self.transform = transform

        # SPACE-GM uses n-hop ego graphs (subgraphs) to perform prediction
        self.subgraph_size = subgraph_size  # number of hops, 0 = use full graph
        self.subgraph_source = subgraph_source
        self.subgraph_allow_distant_edge = subgraph_allow_distant_edge
        self.subgraph_radius_limit = subgraph_radius_limit

        self.N = len(self.processed_paths)
        # Sampling frequency for each cell type
        self.sampling_freq = {self.cell_type_mapping[ct]: 1. / (self.cell_type_freq[ct] + 1e-5)
                              for ct in self.cell_type_mapping}
        self.sampling_freq = torch.from_numpy(np.array([self.sampling_freq[i] for i in range(len(self.sampling_freq))]))
        # Avoid sampling unassigned cell
        self.unassigned_cell_type = unassigned_cell_type
        if sampling_avoid_unassigned and unassigned_cell_type in self.cell_type_mapping:
            self.sampling_freq[self.cell_type_mapping[unassigned_cell_type]] = 0.

        # Cache for subgraphs
        self.cached_data = {}

    def set_indices(self, inds=None):
        """Limit subgraph sampling to a subset of region indices,
        helpful when splitting dataset into training/validation/test regions
        """
        self._indices = inds
        return

    def set_subgraph_source(self, subgraph_source):
        """Set subgraph source"""
        assert subgraph_source in ['chunk_save', 'on-the-fly']
        self.subgraph_source = subgraph_source

    def set_transforms(self, transform=[]):
        """Set transformation functions"""
        self.transform = transform

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.raw_folder_name)

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.processed_folder_name)

    @property
    def raw_file_names(self):
        return sorted([f for f in os.listdir(self.raw_dir) if f.endswith('.gpkl')])

    @property
    def processed_file_names(self):
        # Only files for full graphs
        return sorted([f for f in os.listdir(self.processed_dir) if f.endswith('.gpt') and 'hop' not in f])

    def len(self):
        return len(self.processed_paths)

    def process(self):
        """Featurize all cellular graphs"""
        for raw_path in self.raw_paths:
            G = pickle.load(open(raw_path, 'rb'))
            region_id = G['layer_1'].region_id
            if os.path.exists(os.path.join(self.processed_dir, '%s.0.gpt' % region_id)):
                continue

            # Transform networkx graphs to pyg data objects, and add features for nodes and edges
            data_list = nx_to_tg_hetero_graph(G,
                                       node_features=self.node_features,
                                       edge_features=self.edge_features,
                                       **self.feature_kwargs)

            for i, d in enumerate(data_list):
                try:
                    assert d.region_id == region_id  # graph identifier
                    assert d['cell'].x.shape[1] == len(self.node_feature_names)  # make sure feature dimension matches
                    assert d['cell', 'geom', 'cell'].edge_attr.shape[1] == len(self.edge_feature_names)
                    assert d['cell', 'type', 'cell'].edge_attr.shape[1] == len(self.edge_feature_names)
                except:
                    continue
                d.component_id = i
                if self.pre_transform is not None:
                    for transform_fn in self.pre_transform:
                        d = transform_fn(d)
                torch.save(d, os.path.join(self.processed_dir, '%s.%d.gpt' % (d.region_id, d.component_id)))
        return

    def __getitem__(self, idx):
        """Sample a graph/subgraph from the dataset and apply transformations"""
        # data = self.get(self.indices()[idx])
        data = self.cached_data[idx]
        # Apply transformations
        for transform_fn in self.transform:
            data = transform_fn(data)
        return data

    def get(self, idx):
        data = self.get_full(idx)    
        return data
        
    def get_subgraph(self, idx, center_ind):
        """Get a subgraph from the dataset"""
        # Check cache
        if (idx, center_ind) in self.cached_data:
            return self.cached_data[(idx, center_ind)]
        if self.subgraph_source == 'on-the-fly':
            # Construct subgraph on-the-fly
            return self.calculate_subgraph(idx, center_ind)
        elif self.subgraph_source == 'chunk_save':
            # Load subgraph from pre-saved chunk file
            return self.get_saved_subgraph_from_chunk(idx, center_ind)

    def get_full(self, idx):
        """Read the full cellular graph of region `idx`"""
        if idx in self.cached_data:
            return self.cached_data[idx]
        else:
            data = torch.load(self.processed_paths[idx])
            self.cached_data[idx] = data
            return data

    def get_full_nx(self, idx):
        """Read the full cellular graph (nx.Graph) of region `idx`"""
        return pickle.load(open(self.raw_paths[idx], 'rb'))

    def calculate_subgraph(self, idx, center_ind):
        """Generate the n-hop subgraph around cell `center_ind` from region `idx`"""
        data = self.get_full(idx)
        if not self.subgraph_allow_distant_edge:
            edge_type_mask = (data.edge_attr[:, 0] == EDGE_TYPES["neighbor"])
        else:
            edge_type_mask = None
        sub_node_inds = k_hop_subgraph(int(center_ind),
                                       self.subgraph_size,
                                       data.edge_index,
                                       edge_type_mask=edge_type_mask,
                                       relabel_nodes=False,
                                       num_nodes=data.x.shape[0])[0]

        if self.subgraph_radius_limit > 0:
            # Restrict to neighboring cells that are within the radius (distance to center cell) limit
            assert "center_coord" in self.node_features
            coord_feature_inds = [i for i, n in enumerate(self.node_feature_names) if n.startswith('center_coord')]
            assert len(coord_feature_inds) == 2
            center_cell_coord = data.x[[center_ind]][:, coord_feature_inds]
            neighbor_cells_coord = data.x[sub_node_inds][:, coord_feature_inds]
            dists = ((neighbor_cells_coord - center_cell_coord)**2).sum(1).sqrt()
            sub_node_inds = sub_node_inds[(dists < self.subgraph_radius_limit)]

        # Construct subgraphs as separate pyg data objects
        sub_x = data.x[sub_node_inds]
        sub_edge_index, sub_edge_attr = subgraph(sub_node_inds,
                                                 data.edge_index,
                                                 edge_attr=data.edge_attr,
                                                 relabel_nodes=True)

        relabeled_node_ind = list(sub_node_inds.numpy()).index(center_ind)

        sub_data = {'center_node_index': relabeled_node_ind,  # center node index in the subgraph
                    'original_center_node': center_ind,  # center node index in the original full cellular graph
                    'x': sub_x,
                    'edge_index': sub_edge_index,
                    'edge_attr': sub_edge_attr,
                    'num_nodes': len(sub_node_inds)}

        # Assign graph-level attributes
        for k in data:
            if not k[0] in sub_data:
                sub_data[k[0]] = k[1]

        sub_data = tg.data.Data.from_dict(sub_data)
        self.cached_data[(idx, center_ind)] = sub_data
        return sub_data

    def get_saved_subgraph_from_chunk(self, idx, center_ind):
        """Read the n-hop subgraph around cell `center_ind` from region `idx`
        Subgraph will be extracted from a pre-saved chunk file, which is generated by calling
        `save_all_subgraphs_to_chunk`
        """
        full_graph_path = self.processed_paths[idx]
        subgraphs_path = full_graph_path.replace('.gpt', '.%d-hop.gpt' % self.subgraph_size)
        if not os.path.exists(subgraphs_path):
            warnings.warn("Subgraph save %s not found" % subgraphs_path)
            return self.calculate_subgraph(idx, center_ind)

        subgraphs = torch.load(subgraphs_path)
        # Store to cache first
        for j, g in enumerate(subgraphs):
            self.cached_data[(idx, j)] = g
        return self.cached_data[(idx, center_ind)]

    def pick_center(self, data):
        """Randomly pick a center cell from a full cellular graph, cell type balanced"""
        cell_types = data["x"][:, 0].long()
        freq = self.sampling_freq.gather(0, cell_types)
        freq = freq / freq.sum()
        center_node_ind = np.random.choice(np.arange(len(freq)), p=freq.cpu().data.numpy())
        return center_node_ind

    def load_to_cache(self, idx, subgraphs=True):
        """Pre-load full cellular graph of region `idx` and all its n-hop subgraphs to cache"""
        data = torch.load(self.processed_paths[idx])
        self.cached_data[idx] = data
        if subgraphs or self.subgraph_source == 'chunk_save':
            subgraphs_path = self.processed_paths[idx].replace('.gpt', '.%d-hop.gpt' % self.subgraph_size)
            if not os.path.exists(subgraphs_path):
                raise FileNotFoundError("Subgraph save %s not found, please run `save_all_subgraphs_to_chunk`."
                                        % subgraphs_path)
            neighbor_graphs = torch.load(subgraphs_path)
            for j, ng in enumerate(neighbor_graphs):
                self.cached_data[(idx, j)] = ng

    def save_all_subgraphs_to_chunk(self):
        """Save all n-hop subgraphs for all regions to chunk files (one file per region)"""
        for idx, p in enumerate(self.processed_paths):
            data = self.get_full(idx)
            n_nodes = data.x.shape[0]
            neighbor_graph_path = p.replace('.gpt', '.%d-hop.gpt' % self.subgraph_size)
            if os.path.exists(neighbor_graph_path):
                continue
            subgraphs = []
            for node_i in range(n_nodes):
                subgraphs.append(self.calculate_subgraph(idx, node_i))
            torch.save(subgraphs, neighbor_graph_path)
        return

    def clear_cache(self):
        del self.cached_data
        self.cached_data = {}
        return

    def plot_subgraph(self, idx, center_ind):
        """Plot the n-hop subgraph around cell `center_ind` from region `idx`"""
        xcoord_ind = self.node_feature_names.index('center_coord-x')
        ycoord_ind = self.node_feature_names.index('center_coord-y')

        _subg = self.calculate_subgraph(idx, center_ind)
        coords = _subg.x.data.numpy()[:, [xcoord_ind, ycoord_ind]].astype(float)
        x_c, y_c = coords[_subg.center_node_index]

        G = self.get_full_nx(idx)
        sub_node_inds = []
        for n in G.nodes:
            c = np.array(G.nodes[n]['center_coord']).astype(float).reshape((1, -1))
            if np.linalg.norm(coords - c, ord=2, axis=1).min() < 1e-2:
                sub_node_inds.append(n)
        assert len(sub_node_inds) == len(coords)
        _G = G.subgraph(sub_node_inds)

        node_colors = [self.cell_type_mapping[_G.nodes[n]['cell_type']] for n in _G.nodes]
        node_colors = [matplotlib.cm.tab20(ct) for ct in node_colors]
        plot_graph(_G, node_colors=node_colors)
        xmin, xmax = plt.gca().xaxis.get_data_interval()
        ymin, ymax = plt.gca().yaxis.get_data_interval()

        scale = max(x_c - xmin, xmax - x_c, y_c - ymin, ymax - y_c) * 1.05
        plt.xlim(x_c - scale, x_c + scale)
        plt.ylim(y_c - scale, y_c + scale)
        plt.plot([x_c], [y_c], 'x', markersize=5, color='k')

    def plot_graph_legend(self):
        """Legend for cell type colors"""
        plt.clf()
        plt.figure(figsize=(2, 2))
        for ct, i in self.cell_type_mapping.items():
            plt.plot([0], [0], '.', label=ct, color=matplotlib.cm.tab20(int(i) % 20))
        plt.legend()
        plt.plot([0], [0], color='w', markersize=10)
        plt.axis('off')


def k_hop_subgraph(node_ind,
                   subgraph_size,
                   edge_index,
                   edge_type_mask=None,
                   relabel_nodes=False,
                   num_nodes=None):
    """A customized k-hop subgraph fn that filter for edge_type

    Args:
        node_ind (int): center node index
        subgraph_size (int): number of hops for the neighborhood subgraph
        edge_index (torch.Tensor): edge index tensor for the full graph
        edge_type_mask (torch.Tensor): edge type mask
        relabel_nodes (bool): if to relabel node indices to consecutive integers
        num_nodes (int): number of nodes in the full graph

    Returns:
        subset (torch.LongTensor): indices of nodes in the subgraph
        edge_index (torch.LongTensor): edges in the subgraph
        inv (toch.LongTensor): location of the center node in the subgraph
        edge_mask (torch.BoolTensor): edge mask indicating which edges were preserved
    """

    num_nodes = edge_index.max().item() + 1 if num_nodes is None else num_nodes
    col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)
    edge_type_mask = torch.ones_like(edge_mask) if edge_type_mask is None else edge_type_mask

    if isinstance(node_ind, (int, list, tuple)):
        node_ind = torch.tensor([node_ind], device=row.device).flatten()
    else:
        node_ind = node_ind.to(row.device)

    subsets = [node_ind]
    next_root = node_ind

    for _ in range(subgraph_size):
        node_mask.fill_(False)
        node_mask[next_root] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        subsets.append(col[edge_mask])
        next_root = col[edge_mask * edge_type_mask]  # use nodes connected with mask=True to span

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:node_ind.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_ind = row.new_full((num_nodes, ), -1)
        node_ind[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_ind[edge_index]

    return subset, edge_index, inv, edge_mask