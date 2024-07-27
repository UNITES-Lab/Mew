import numpy as np
import torch
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros

class FeedForwardNet(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, n_layers, dropout):
        super(FeedForwardNet, self).__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        if n_layers == 1:
            self.layers.append(nn.Linear(in_feats, out_feats))
        else:
            self.layers.append(nn.Linear(in_feats, hidden))
            for i in range(n_layers - 2):
                self.layers.append(nn.Linear(hidden, hidden))
            self.layers.append(nn.Linear(hidden, out_feats))
        if self.n_layers > 1:
            self.prelu = nn.PReLU()
            self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer_id, layer in enumerate(self.layers):
            x = layer(x)
            if layer_id < self.n_layers - 1:
                x = self.dropout(self.prelu(x))
        return x

class SIGN_v2(nn.Module):
    def __init__(self, num_layer, num_feat, emb_dim, ffn_layers=2, dropout=0.25):
        super(SIGN_v2, self).__init__()

        in_feats = num_feat
        emb_dim = emb_dim
        out_feats = emb_dim
        num_hops = num_layer + 1

        self.dropout = nn.Dropout(dropout)
        self.prelu = nn.PReLU()
        self.inception_ffs = nn.ModuleList()

        self.batch_norms = torch.nn.ModuleList()
        for hop in range(num_hops):
            self.inception_ffs.append(
                FeedForwardNet(in_feats, emb_dim, emb_dim, ffn_layers, dropout))
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
        self.project = FeedForwardNet(num_hops * emb_dim, emb_dim, out_feats,
                                      ffn_layers, dropout)

    def forward(self, feats):
        feats = [feat.float() for feat in feats]
        hidden = []
        for i, (feat, ff) in enumerate(zip(feats, self.inception_ffs)):
            emb = ff(feat)
            hidden.append(self.batch_norms[i](emb))
        out = self.project(self.dropout(self.prelu(torch.cat(hidden, dim=-1))))
        return out

    def reset_parameters(self):
        for ff in self.inception_ffs:
            ff.reset_parameters()
        self.project.reset_parameters()


class SIGN_pred(torch.nn.Module):
    def __init__(self,
                 num_layer=2,
                 num_feat=38,
                 emb_dim=256,
                 num_additional_feat=0,
                 num_node_tasks=15,
                 num_graph_tasks=2,
                 node_embedding_output="last",
                 drop_ratio=0,
                 graph_pooling="mean",
                 attn_weight=False,
                 shared=False,
                 ):
        super(SIGN_pred, self).__init__()
        self.drop_ratio = drop_ratio
        self.emb_dim = emb_dim
        self.num_node_tasks = num_node_tasks
        self.num_graph_tasks = num_graph_tasks
        self.attn_weight = attn_weight
        
        self.sign = SIGN_v2(num_layer, num_feat, emb_dim, dropout=drop_ratio)
        self.sign2 = self.sign if shared else SIGN_v2(num_layer, num_feat, emb_dim, dropout=drop_ratio)
        self.leakyrelu = nn.LeakyReLU(0.3)
        self.attention = nn.Parameter(torch.empty(size=(emb_dim, 1)))
        glorot(self.attention)
        if self.attn_weight:
            self.w1 = torch.nn.Linear(emb_dim, emb_dim)
            self.w2 = torch.nn.Linear(emb_dim, emb_dim)
            torch.nn.init.xavier_uniform_(self.w1.weight.data)
            torch.nn.init.xavier_uniform_(self.w1.weight.data)
                
        # Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if node_embedding_output == "concat":
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if node_embedding_output == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        # For node and graph predictions
        self.mult = 1
        if graph_pooling[:-1] == "set2set":
            self.mult *= 2
        if node_embedding_output == "concat":
            self.mult *= (self.num_layer + 1)

        node_embedding_dim = self.mult * self.emb_dim
        if self.num_graph_tasks > 0:
            self.graph_pred_module = torch.nn.Sequential(
                torch.nn.Linear(node_embedding_dim + num_additional_feat, node_embedding_dim),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(node_embedding_dim, node_embedding_dim),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(node_embedding_dim, self.num_graph_tasks))


    def from_pretrained(self, model_file):
        original_dict = torch.load(model_file)
        new_dict = {key.replace('gnn.', '') if 'gnn.' in key else key: value for key, value in original_dict.items()}
        if self.num_graph_tasks > 0:
            graph_tasks_idx = []
            for key in new_dict.keys():
                if key.startswith('graph_pred_module'):
                    idx_part = key.split('.')[1]
                    graph_tasks_idx.append(int(idx_part))

            for i in range(len(self.graph_pred_module)):
                if i in graph_tasks_idx: 
                    self.graph_pred_module[i].weight = torch.nn.Parameter(new_dict[f'graph_pred_module.{i}.weight'])
                    self.graph_pred_module[i].bias = torch.nn.Parameter(new_dict[f'graph_pred_module.{i}.bias'])
                    del new_dict[f'graph_pred_module.{i}.weight']
                    del new_dict[f'graph_pred_module.{i}.bias']

        self.gnn.load_state_dict(new_dict)

    def forward(self, data, batch=None, embed=False):
        batch = batch if batch != None else torch.zeros(len(data[0][0]),).long()

        node_representation_1 = self.sign(data[0]) # geom
        node_representation_2 = self.sign2(data[1]) # cell_type

        if self.attn_weight:
            geom_ = self.leakyrelu(torch.mm(self.w1(node_representation_1), self.attention))
            cell_type_ = self.leakyrelu(torch.mm(self.w2(node_representation_2), self.attention))
        else:
            geom_ = self.leakyrelu(torch.mm(node_representation_1, self.attention))
            cell_type_ = self.leakyrelu(torch.mm(node_representation_2, self.attention))

        values = torch.softmax(torch.cat((geom_, cell_type_), dim=1), dim=1)
        node_representation = (values[:,0].unsqueeze(1) * node_representation_1) + (values[:,1].unsqueeze(1) * node_representation_2)
        
        return_vals = []
        if self.num_graph_tasks > 0:
            input = self.pool(node_representation, batch.to(node_representation.device))
            graph_pred = self.graph_pred_module(input)
            return_vals.append(graph_pred)

        if embed:
            return return_vals, values, node_representation_1, node_representation_2
        else:
            return return_vals, values

# Loss functions
class BinaryCrossEntropy(torch.nn.Module):
    """Weighted binary cross entropy loss function"""
    def __init__(self, **kwargs):
        super(BinaryCrossEntropy, self).__init__(**kwargs)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, y_pred, y, w):
        return (self.loss_fn(y_pred, y) * w).mean()

class CrossEntropy(torch.nn.Module):
    """Cross entropy loss function for multi-class classification"""
    def __init__(self, **kwargs):
        super(CrossEntropy, self).__init__(**kwargs)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, y_pred, y, w):
        return self.loss_fn(y_pred, y)


class CoxSGDLossFn(torch.nn.Module):
    """Cox SGD loss function"""
    def __init__(self, top_n=2, regularizer_weight=0.05, **kwargs):
        self.top_n = top_n
        self.regularizer_weight = regularizer_weight
        super(CoxSGDLossFn, self).__init__(**kwargs)

    def forward(self, y_pred, length, event):
        assert y_pred.shape[0] == length.shape[0] == event.shape[0]
        n_samples = y_pred.shape[0]
        num_tasks = y_pred.shape[1]
        loss = 0

        for task in range(num_tasks):
            _length = length[:, task]
            _event = event[:, task]
            _pred = y_pred[:, task]

            pair_mat = (_length.reshape((1, -1)) - _length.reshape((-1, 1)) > 0) * _event.reshape((-1, 1))
            if self.top_n > 0:
                p_with_rand = pair_mat * (1 + torch.rand_like(pair_mat))
                rand_thr_ind = torch.argsort(p_with_rand, axis=1)[:, -(self.top_n + 1)]
                rand_thr = p_with_rand[(torch.arange(n_samples), rand_thr_ind)].reshape((-1, 1))
                pair_mat = pair_mat * (p_with_rand > rand_thr)

            valid_sample_is = torch.nonzero(pair_mat.sum(1)).flatten()
            pair_mat[(valid_sample_is, valid_sample_is)] = 1

            score_diff = (_pred.reshape((1, -1)) - _pred.reshape((-1, 1)))
            score_diff_row_max = torch.max(score_diff, axis=1, keepdims=True).values
            loss_tmp = (torch.exp(score_diff - score_diff_row_max) * pair_mat).sum(1)
            loss += (score_diff_row_max[:, 0][valid_sample_is] + torch.log(loss_tmp[valid_sample_is])).sum()
            
            regularizer = torch.abs(pair_mat.sum(0) * _pred.flatten()).sum()
            loss += self.regularizer_weight * regularizer

        return loss