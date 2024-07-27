import numpy as np
import pandas as pd
from copy import deepcopy
import torch
from src.utils import generate_charville_label

class FeatureMask(object):
    """ Transformer object for masking features """
    def __init__(self,
                 dataset,
                 use_neighbor_node_features=None,
                 use_center_node_features=None,
                 use_edge_features=None,
                 **kwargs):
        """ Construct the transformer

        Args:
            dataset (CellularGraphDataset): dataset object
            use_neighbor_node_features (list): list of node feature items to use
                for non-center nodes, all other features will be masked out
            use_center_node_features (list): list of node feature items to use
                for the center node, all other features will be masked out
            use_edge_features (list): list of edge feature items to use,
                all other features will be masked out
        """

        self.node_feature_names = dataset.node_feature_names
        self.edge_feature_names = dataset.edge_feature_names

        self.use_neighbor_node_features = use_neighbor_node_features if \
            use_neighbor_node_features is not None else dataset.node_features
        self.use_center_node_features = use_center_node_features if \
            use_center_node_features is not None else dataset.node_features
        self.use_edge_features = use_edge_features if \
            use_edge_features is not None else dataset.edge_features

        self.center_node_feature_masks = [
            1 if any(name.startswith(feat) for feat in self.use_center_node_features)
            else 0 for name in self.node_feature_names]
        self.neighbor_node_feature_masks = [
            1 if any(name.startswith(feat) for feat in self.use_neighbor_node_features)
            else 0 for name in self.node_feature_names]

        self.center_node_feature_masks = \
            torch.from_numpy(np.array(self.center_node_feature_masks).reshape((-1,))).float()
        self.neighbor_node_feature_masks = \
            torch.from_numpy(np.array(self.neighbor_node_feature_masks).reshape((1, -1))).float()

    def __call__(self, data):
        data = deepcopy(data)
        if "center_node_index" in data:
            center_node_feat = data.x[data.center_node_index].detach().data.clone()
        else:
            center_node_feat = None
        data = self.transform_neighbor_node(data)
        data = self.transform_center_node(data, center_node_feat)
        return data

    def transform_neighbor_node(self, data):
        """Apply neighbor node feature masking"""
        data['cell'].x = data['cell'].x * self.neighbor_node_feature_masks
        return data

    def transform_center_node(self, data, center_node_feat=None):
        """Apply center node feature masking"""
        if center_node_feat is None:
            return data
        assert "center_node_index" in data
        center_node_feat = center_node_feat * self.center_node_feature_masks
        data['cell'].x[data['cell'].center_node_index] = center_node_feat
        return data


class AddCenterCellType(object):
    """Transformer for center cell type prediction"""
    def __init__(self, dataset, **kwargs):
        self.node_feature_names = dataset.node_feature_names
        self.cell_type_feat = self.node_feature_names.index('cell_type')
        # Assign a placeholder cell type for the center node
        self.placeholder_cell_type = max(dataset.cell_type_mapping.values()) + 1

    def __call__(self, data):
        data = deepcopy(data)
        assert "center_node_index" in data, \
            "Only subgraphs with center nodes are supported, cannot find `center_node_index`"
        center_node_feat = data.x[data.center_node_index].detach().clone()
        center_cell_type = center_node_feat[self.cell_type_feat]
        data.node_y = center_cell_type.long().view((1,))
        data.x[data.center_node_index, self.cell_type_feat] = self.placeholder_cell_type
        return data


class AddCenterCellBiomarkerExpression(object):
    """Transformer for center cell biomarker expression prediction"""
    def __init__(self, dataset, **kwargs):
        self.node_feature_names = dataset.node_feature_names
        self.bm_exp_feat = np.array([
            i for i, feat in enumerate(self.node_feature_names)
            if feat.startswith('biomarker_expression')])

    def __call__(self, data):
        assert "center_node_index" in data, \
            "Only subgraphs with center nodes are supported, cannot find `center_node_index`"
        center_node_feat = data.x[data.center_node_index].detach().clone()
        center_cell_exp = center_node_feat[self.bm_exp_feat].float()
        data.node_y = center_cell_exp.view(1, -1)
        return data


class AddCenterCellIdentifier(object):
    """Transformer for adding another feature column for identifying center cell
    Helpful when predicting node-level tasks.
    """
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, data):
        assert "center_node_index" in data, \
            "Only subgraphs with center nodes are supported, cannot find `center_node_index`"
        center_cell_identifier_column = torch.zeros((data.x.shape[0], 1), dtype=data.x.dtype)
        center_cell_identifier_column[data.center_node_index, 0] = 1.
        data.x = torch.cat([data.x, center_cell_identifier_column], dim=1)
        return data


class AddGraphLabel(object):
    """Transformer for adding graph-level task labels"""
    def __init__(self, graph_label_file, tasks=[], **kwargs):
        """ Construct the transformer

        Args:
            graph_label_file (str): path to the csv file containing graph-level
                task labels. This file should always have the first column as region id.
            tasks (list): list of tasks to use, corresponding to column names
                of the csv file. If empty, use all tasks in the file
        """
        self.label_df = pd.read_csv(graph_label_file)
        graph_tasks = list(self.label_df.columns) if len(tasks) == 0 else tasks
        if ('charville' in graph_label_file) & ('recurrence_event' not in self.label_df.columns):
            self.label_df = generate_charville_label(self.label_df)
            self.label_df.to_csv(graph_label_file, index=False) # update csv file
        self.label_df.index = self.label_df.index.map(str) # Convert index to str
        self.graph_tasks = graph_tasks
        self.tasks, self.class_label_weights = self.build_class_weights(graph_tasks)
        self.c_index = True if ('survival_status' in np.stack(graph_tasks)) or ('recurrence_event' in np.stack(graph_tasks)) else False

    def build_class_weights(self, graph_tasks):
        valid_tasks = []
        class_label_weights = {}
        for task in graph_tasks:
            ar = list(self.label_df[task])
            valid_vals = [_y for _y in ar if _y == _y]
            unique_vals = set(valid_vals)
            if not all(v.__class__ in [int, float] for v in unique_vals):
                # Skip tasks with non-numeric labels
                continue
            valid_tasks.append(task)
            if len(unique_vals) > 5:
                # More than 5 unique values in labels, likely a regression task
                class_label_weights[task] = {_y: 1 for _y in unique_vals}
            else:
                # Classification task, compute class weights
                val_counts = {_y: valid_vals.count(_y) for _y in unique_vals}
                max_count = max(val_counts.values())
                class_label_weights[task] = {_y: max_count / val_counts[_y] for _y in unique_vals}
        return valid_tasks, class_label_weights

    def fetch_label(self, region_id, task_name):
        # Updated from SPACE-GM
        new_int = int(region_id.split('_')[1][1:])
        if 'UPMC' in region_id:
            new_int += 4
        if len(str(new_int)) == 1:
            new_int = f'00{new_int}'
        elif len(str(new_int)) == 2:
            new_int = f'0{new_int}'
        new_region_id = f'SpaceGMP-65_c{new_int}_' + region_id.split('_')[2] + '_' + region_id.split('_')[3] + '_'  + region_id.split('_')[4]
        if 'acquisition_id_visualizer' in self.label_df:
            y = self.label_df[self.label_df["acquisition_id_visualizer"] == new_region_id][task_name].item()
        else:
            y = self.label_df[self.label_df["region_id"] == new_region_id][task_name].item()

        if y != y: # np.nan
            y = 0
            w = 0
        else:
            w = self.class_label_weights[task_name][y]
        return y, w

    def fetch_length_event(self, region_id, task_name):
        # Updated from SPACE-GM
        new_int = int(region_id.split('_')[1][1:])
        if 'UPMC' in region_id:
            new_int += 4
        if len(str(new_int)) == 1:
            new_int = f'00{new_int}'
        elif len(str(new_int)) == 2:
            new_int = f'0{new_int}'
        new_region_id = f'SpaceGMP-65_c{new_int}_' + region_id.split('_')[2] + '_' + region_id.split('_')[3] + '_'  + region_id.split('_')[4]
        
        if 'acquisition_id_visualizer' in self.label_df:
            length = self.label_df[self.label_df["acquisition_id_visualizer"] == new_region_id][task_name[0]].item()
            event = self.label_df[self.label_df["acquisition_id_visualizer"] == new_region_id][task_name[1]].item()
        else:
            length = self.label_df[self.label_df["region_id"] == new_region_id][task_name[0]].item()
            event = self.label_df[self.label_df["region_id"] == new_region_id][task_name[1]].item()

        return length, event

    def __call__(self, data):
        graph_y = []
        graph_w = []

        for task in self.graph_tasks:
            if self.c_index:
                y, w = self.fetch_length_event(data.region_id, task)
            else:
                y, w = self.fetch_label(data.region_id, task)
            
            graph_y.append(y)
            graph_w.append(w)
            data.graph_y = torch.from_numpy(np.array(graph_y).reshape((1, -1)))
            data.graph_w = torch.from_numpy(np.array(graph_w).reshape((1, -1)))
                
        return data

class AddTwoGraphLabel(object):
    """Transformer for adding graph-level task labels"""
    def __init__(self, graph_label_files=[], tasks=[], **kwargs):
        """ Construct the transformer

        Args:
            graph_label_file (str): path to the csv file containing graph-level
                task labels. This file should always have the first column as region id.
            tasks (list): list of tasks to use, corresponding to column names
                of the csv file. If empty, use all tasks in the file
        """
        self.label_df_upmc = pd.read_csv(graph_label_files[0])
        self.label_df_dfci = pd.read_csv(graph_label_files[1])

        self.label_df_upmc.index = self.label_df_upmc.index.map(str)  # Convert index to str
        self.label_df_dfci.index = self.label_df_dfci.index.map(str)  # Convert index to str

        graph_tasks_upmc = tasks[0]
        graph_tasks_dfci = tasks[1]
        self.graph_tasks_upmc = graph_tasks_upmc
        self.graph_tasks_dfci = graph_tasks_dfci

        self.tasks_upmc, self.class_label_weights_upmc = self.build_class_weights(graph_tasks_upmc, label_df=self.label_df_upmc)
        self.tasks_dfci, self.class_label_weights_dfci = self.build_class_weights(graph_tasks_dfci, label_df=self.label_df_dfci)
        self.c_index = False

    def build_class_weights(self, graph_tasks, label_df=None):
        valid_tasks = []
        class_label_weights = {}
        for task in graph_tasks:
            if type(task) == list:
                task = task[0] # length
            ar = list(label_df[task])
            valid_vals = [_y for _y in ar if _y == _y]
            unique_vals = set(valid_vals)
            if not all(v.__class__ in [int, float] for v in unique_vals):
                # Skip tasks with non-numeric labels
                continue
            valid_tasks.append(task)

            if len(unique_vals) > 5:
                # More than 5 unique values in labels, likely a regression task
                class_label_weights[task] = {_y: 1 for _y in unique_vals}
            else:
                # Classification task, compute class weights
                val_counts = {_y: valid_vals.count(_y) for _y in unique_vals}
                max_count = max(val_counts.values())
                class_label_weights[task] = {_y: max_count / val_counts[_y] for _y in unique_vals}
        
        return valid_tasks, class_label_weights

    def fetch_label(self, region_id, task_name):
        # Updated from SPACE-GM
        self.label_df = self.label_df_upmc if 'UPMC' in region_id else self.label_df_dfci
        new_int = int(region_id.split('_')[1][1:])
        if 'UPMC' in region_id:
            new_int += 4
        if len(str(new_int)) == 1:
            new_int = f'00{new_int}'
        elif len(str(new_int)) == 2:
            new_int = f'0{new_int}'
        
        if 'UPMC' in region_id:
            new_region_id = f'SpaceGMP-65_c{new_int}_' + region_id.split('_')[2] + '_' + region_id.split('_')[3] + '_'  + region_id.split('_')[4]
        else:
            new_region_id = region_id
        if "acquisition_id_visualizer" in self.label_df:
            y = self.label_df[self.label_df["acquisition_id_visualizer"] == new_region_id][task_name].item()
            
        else:
            y = self.label_df[self.label_df["region_id"] == new_region_id][task_name].item()
            
        if y != y: # np.nan
            y = 0
            w = 0
        else:
            self.class_label_weights = self.class_label_weights_upmc if 'UPMC' in region_id else self.class_label_weights_dfci
            w = self.class_label_weights[task_name][y]
        return y, w

    def __call__(self, data):
        graph_y = []
        graph_w = []

        self.tasks = self.tasks_upmc if 'UPMC' in data.region_id else self.tasks_dfci
        for task in self.tasks:
            y, w = self.fetch_label(data.region_id, task)
            graph_y.append(y)
            graph_w.append(w)
        data.graph_y = torch.from_numpy(np.array(graph_y).reshape((1, -1)))
        data.graph_w = torch.from_numpy(np.array(graph_w).reshape((1, -1)))

        return data