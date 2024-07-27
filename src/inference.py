import time
import numpy as np
from tqdm import trange 
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score

def collect_predict_for_all_nodes(model,
                                  xs,
                                  device,
                                  inds=None,
                                  graph=False,
                                  **kwargs):
    
    model = model.to(device)
    model.eval()
    
    node_results = {}
    graph_results = {}
    voronoi_attn_results = {}
    cell_type_attn_results = {}

    start_time = time.time()
    for i in inds:
        input_1 = [_x.to(device) for _x in xs[i][0]]
        input_2 = [_x.to(device) for _x in xs[i][1]]
        res, attn_values = model([input_1, input_2])

        voronoi_attn = attn_values[:,0].mean().item()
        cell_type_attn = attn_values[:,1].mean().item()

        graph_results[i] = res[0].cpu().data.numpy()
        voronoi_attn_results[i] = voronoi_attn
        cell_type_attn_results[i] = cell_type_attn
    
    # print(f'Prediction for {len(inds)} graphs:', time.time() - start_time)
    
    return node_results, graph_results, voronoi_attn_results, cell_type_attn_results


def graph_classification_evaluate_fn(graph_preds,
                                     graph_ys,
                                     graph_ws=None,
                                     task='classification',
                                     print_res=True):
    """ Evaluate graph classification accuracy

    Args:
        graph_preds (array-like): binary classification logits for graph-level tasks, (num_subgraphs, num_tasks)
        graph_ys (array-like): binary labels for graph-level tasks, (num_subgraphs, num_tasks)
        graph_ws (array-like): weights for graph-level tasks, (num_subgraphs, num_tasks)
        print_res (bool): if to print the accuracy results

    Returns:
        list: list of metrics on all graph-level tasks
    """
    if graph_ws is None:
        graph_ws = np.ones_like(graph_ys)
    scores = []
    if task != 'classification':
        for task_i in trange(graph_preds.shape[1]):
            _pred = graph_preds[:, task_i]
            _times = graph_ys[:, task_i]
            _observed = graph_ws[:, task_i]
            idx = _times == _times
            s = concordance_index(_observed[idx], _pred[idx], _times[idx])
            scores.append(s)
    else:    
        for task_i in trange(graph_ys.shape[1]):
            _label = graph_ys[:, task_i]
            _pred = graph_preds[:, task_i]
            _w = graph_ws[:, task_i]
            s = roc_auc_score(_label[np.where(_w > 0)], _pred[np.where(_w > 0)])
            scores.append(s)
    if print_res:
        print("GRAPH %s" % str(scores))
    return scores


def full_graph_graph_classification_evaluate_fn(
        dataset_yw,
        graph_results,
        task,
        aggr='mean',
        print_res=True):

    n_tasks = list(graph_results.values())[0].shape[1]
    graph_preds = []
    graph_ys = []
    graph_ws = []
    for i in graph_results:
        graph_pred = [p for p in graph_results[i] if ((p is not None) and np.all(p == p))]
        graph_pred = np.stack(graph_pred, 0)

        if aggr == 'mean':
            graph_pred = np.nanmean(graph_pred, 0)
        else:
            raise NotImplementedError("Only mean-aggregation is supported now")

        graph_y = dataset_yw[i][0].numpy().flatten()
        graph_w = dataset_yw[i][1].numpy().flatten()
        graph_preds.append(graph_pred)
        graph_ys.append(graph_y)
        graph_ws.append(graph_w)

    graph_preds = np.concatenate(graph_preds, 0).reshape((-1, n_tasks))
    graph_ys = np.concatenate(graph_ys, 0).reshape((-1, n_tasks))
    graph_ws = np.concatenate(graph_ws, 0).reshape((-1, n_tasks))

    return graph_classification_evaluate_fn(graph_preds, graph_ys, graph_ws, task, print_res=print_res)
