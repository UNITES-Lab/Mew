import time
import torch
from torch_geometric.transforms import SIGN
from tqdm import trange
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
import torch.distributions as dist
from torch_geometric.utils import add_remaining_self_loops

class PrecomputingBase(torch.nn.Module):
    def __init__(self, num_layer, device):
        super(PrecomputingBase, self).__init__()
        self.device = device
        self.num_layers = num_layer

    def precompute(self, data):
        print("precomputing features, may take a while.")
        t1 = time.time()
        print('Start Precomputing...!')
        
        self.xs = []
        for i in trange(len(data)):
            _data = data[i]
            # Geom data
            geom_data = Data()
            geom_data.x = _data['cell'].x
            geom_data.edge_index = _data['cell', 'geom', 'cell'].edge_index
            geom_data.edge_attr = _data['cell', 'geom', 'cell'].edge_attr
            geom_data.edge_index , geom_data.edge_attr = add_remaining_self_loops(geom_data.edge_index, geom_data.edge_attr)
            geom_data = geom_data.to(self.device)
            _geom_x = SIGN(self.num_layers)(geom_data, stochastic=False)
            geom_x = [_geom_x.x.cpu()] + [_geom_x[f"x{i}"] for i in range(1, self.num_layers + 1)]
            
            # Cell-type data
            cell_type_data = Data()
            cell_type_data.x = _data['cell'].x
            cell_type_data.edge_index = _data['cell', 'type', 'cell'].edge_index
            cell_type_data.edge_attr = _data['cell', 'type', 'cell'].edge_attr
            cell_type_data.edge_index , cell_type_data.edge_attr = add_remaining_self_loops(cell_type_data.edge_index, cell_type_data.edge_attr)
            cell_type_data = cell_type_data.to(self.device)
            _cell_type_x = SIGN(self.num_layers)(cell_type_data, stochastic=True)
            cell_type_x = [_cell_type_x.x.cpu()] + [_cell_type_x[f"x{i}"] for i in range(1, self.num_layers + 1)]

            self.xs.append([geom_x, cell_type_x])

        t2 = time.time()
        print("Precomputing finished by %.4f s." % (t2 - t1))
    
        return self.xs

    def forward(self, xs):
        raise NotImplementedError


class SIGN(BaseTransform):
    def __init__(self, K):
        self.K = K

    def __call__(self, data, stochastic=False):
        assert data.edge_index is not None
        assert data.x is not None        
        if not stochastic:
            weight = torch.ones_like(data.edge_attr[:,1])
            _edge_weight = data.edge_attr.sum(dim=1) 

        else:
            weight = data.edge_attr[:,1]
            weight = 1 - weight
            _edge_weight = torch.ones_like(data.edge_attr[:,1])
        
        xs = [data.x]
        for i in range(1, self.K + 1):
            adj_t = self.stochastic_adj(data, weight, _edge_weight=_edge_weight)
            xs += [adj_t @ xs[-1]]
            data[f'x{i}'] = xs[-1].cpu()

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(K={self.K})'

    def stochastic_adj(self, data, weight, _edge_weight=None):
        bernoulli_dist = dist.Bernoulli(weight)
        mask = bernoulli_dist.sample().bool()

        N = data.num_nodes
        N_edges = data.edge_index.shape[1]

        if _edge_weight == None:
            edge_attr = torch.ones((N_edges,), device=data.device)
        else:
            edge_attr = _edge_weight[mask]

        edge_index = data.edge_index[:,mask]
        row, col = edge_index 
        
        deg = torch.zeros(N, device=row.device, dtype=edge_attr.dtype)
        deg.scatter_add_(dim=0, index=col, src=edge_attr)

        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_attr  * deg_inv_sqrt[col]
        adj = torch.sparse_coo_tensor(edge_index, edge_weight, size=torch.Size([N, N]))
        adj_t = adj.t()

        return adj_t

