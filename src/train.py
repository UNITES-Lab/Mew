import os
import numpy as np
import torch
import torch.optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from src.inference import collect_predict_for_all_nodes
from tqdm import trange
from copy import deepcopy

class CombinedDataset(Dataset):
    def __init__(self, graph_dataset, x_dataset):
        self.graph_dataset = graph_dataset
        self.x_dataset = x_dataset

    def __getitem__(self, idx):
        graph_y, graph_w = self.graph_dataset[idx]
        geom_x, cell_type_x = self.x_dataset[idx]
        geom_x_tensor = torch.stack(geom_x)
        cell_type_x_tensor = torch.stack(cell_type_x)
        batch_length = geom_x[0].shape[0]
        
        return graph_y, graph_w, geom_x_tensor, cell_type_x_tensor, batch_length

    def __len__(self):
        return len(self.graph_dataset)

def custom_collate(batch):
    graph_y_list, graph_w_list, geom_x_list, cell_type_x_list, batch_length_list = zip(*batch)
    batch_idx_list = torch.tensor(sum([[i]*batch_length_list[i] for i in range(len(batch))], []))
    graph_y = torch.stack(graph_y_list).squeeze(2)
    graph_w = torch.stack(graph_w_list).squeeze(2)
    geom_x_layers = torch.cat(geom_x_list, dim=1)
    cell_type_x_layers = torch.cat(cell_type_x_list, dim=1)
    
    return batch_idx_list, graph_y, graph_w, geom_x_layers, cell_type_x_layers

def train_full_graph(model,
                   dataset_yw,
                   xs,
                   device,
                   filename,
                   fold,
                   logger,
                   graph_task_loss_fn=None,
                   train_inds=None,
                   valid_inds=None,
                   test_inds=None,
                   early_stop=100,
                   num_iterations=1e5,
                   evaluate_freq=1e4,
                   evaluate_fn=[],
                   evaluate_on_train=True,
                   batch_size=64,
                   lr=0.001,
                   graph_loss_weight=1.,
                   task='classification',
                   name=None,
                   save_model=False,
                   **kwargs):

    if train_inds is None:
        train_inds = np.arange(len(xs))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    node_losses = []
    graph_losses = []

    train_dataset = [dataset_yw[i] for i in train_inds]
    train_xs = [xs[i] for i in train_inds]

    train_dataset_combined = CombinedDataset(train_dataset, train_xs)
    train_loader = DataLoader(train_dataset_combined, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate)

    best_score = 0
    cnt = 0
    
    for i_iter in trange(int(num_iterations)):
        model.train()
        model.zero_grad()

        batch, graph_y, graph_w, geom_x, cell_type_x = next(iter(train_loader))
        loss = 0.
        geom_x = [_x.to(device) for _x in geom_x]
        cell_type_x = [_x.to(device) for _x in cell_type_x]

        res, _, z1, z2 = model([geom_x, cell_type_x], batch, embed=True)
        if res[0].sum().isnan():
            return [-1] * model.num_graph_tasks

        if model.num_graph_tasks > 0:
            assert graph_task_loss_fn is not None, \
                "Please specify `graph_task_loss_fn` in the training kwargs"

            graph_pred = res[-1]
            graph_y, graph_w = graph_y.float().to(device), graph_w.float().to(device)
            
            if model.num_graph_tasks > 1:
                graph_y, graph_w = graph_y.squeeze(1), graph_w.squeeze(1)
            idx = graph_y[:,0] == graph_y[:,0]
            graph_loss = graph_task_loss_fn(graph_pred[idx], graph_y[idx], graph_w[idx])
            loss += graph_loss * graph_loss_weight
        
            graph_losses.append(graph_loss.to('cpu').data.item())
            
        loss.backward()
        optimizer.step()

        if i_iter > 0 and (i_iter+1) % evaluate_freq == 0:
            model.eval()
            summary_str = "Finished iterations %d" % (i_iter+1)
            if len(node_losses) > evaluate_freq:
                summary_str += ", node loss %.2f" % np.mean(node_losses[-evaluate_freq:])
            if len(graph_losses) > evaluate_freq:
                summary_str += ", graph loss %.2f" % np.mean(graph_losses[-evaluate_freq:])
            print(summary_str)

            fn = evaluate_fn[0]
            result = fn(model,
                dataset_yw,
                xs,
                device,
                filename,
                fold,
                logger,
                train_inds=train_inds if evaluate_on_train else None,
                valid_inds=valid_inds,
                batch_size=batch_size,
                task=task,
                **kwargs)
            
            _txt = ",".join([s if isinstance(s, str) else ("%.3f" % s) for s in result])
            logger.info(f'[{name}-{task}][Fold {fold}][Epoch {i_iter+1}/{num_iterations}]-{_txt} \t Filename: {filename}')
            if cnt > 0:
                print(best_txt_test)
            valid_score = np.mean(result[-model.num_graph_tasks:])
            
            if best_score < valid_score:
                cnt +=1
                print(f'Iter: {i_iter+1}, Reached best valid score! Start testing ...')
                best_score = valid_score
                
                # Test on test_inds
                if test_inds != None:
                    result_test = fn(model,
                        dataset_yw,
                        xs,
                        device,
                        filename,
                        fold,
                        logger,
                        train_inds=None,
                        valid_inds=test_inds,
                        batch_size=batch_size,
                        task=task,
                        save=True,
                        **kwargs)

                    _txt_test = ",".join([s if isinstance(s, str) else ("%.3f" % s) for s in result_test])
                    best_txt_test = f'[{name}-{task}][Fold {fold}][(Best Epoch) {i_iter+1}/{num_iterations}][(Test)]-{_txt_test} \t Filename: {filename}'
                    logger.info(best_txt_test)
                
                    best_epoch = i_iter

                    if save_model:
                        fn_save = evaluate_fn[1]
                        fn_save(model,
                            name,
                            task,
                            fold)
            
                else:
                    _txt_test = _txt
            
            elif i_iter - best_epoch >= early_stop:
                print('Early Stop!')
                break

            else:
                pass

    return model, _txt_test


def evaluate_by_full_graph(model,
                           dataset_yw,
                           xs,
                           device,
                           filename,
                           fold,
                           logger,
                           train_inds=None,
                           valid_inds=None,
                           batch_size=64,
                           shuffle=True,
                           full_graph_graph_task_evaluate_fn=None,
                           score_file=None,
                           save=False,
                           task='classification',
                           **kwargs):
    
    score_row = ["Eval-Full-Graph"]
    if train_inds is not None:
        score_row.append("Train")
        node_preds, graph_preds, voronoi_attn, cell_type_attn = collect_predict_for_all_nodes(
            model,
            xs,
            device,
            inds=train_inds,
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs)

        if model.num_graph_tasks > 0:
            # Evaluate graph-level predictions
            assert full_graph_graph_task_evaluate_fn is not None, \
                "Please specify `full_graph_graph_task_evaluate_fn` in the training kwargs"
            score_row.append("attn-score")
            score_row.extend([np.nanmean(np.array(list(voronoi_attn.values()))), np.nanmean(np.array(list(cell_type_attn.values())))])
            score_row.append("graph-score")
            score_row.extend(full_graph_graph_task_evaluate_fn(dataset_yw, graph_preds, task, print_res=False))

    if valid_inds is not None:
        score_row.append("Valid")
        node_preds, graph_preds, voronoi_attn, cell_type_attn = collect_predict_for_all_nodes(
            model,
            xs,
            device,
            inds=valid_inds,
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs)

        region_ids = []
        graph_pred_list = []
        graph_ys = []
        graph_ws = []
        
        for _i, (i, pred) in enumerate(graph_preds.items()):
            region_id = valid_inds[_i]
            graph_y = dataset_yw[i][0]
            graph_w = dataset_yw[i][1]
            graph_pred_ = np.nanmean(pred, 0)

            region_ids.append(region_id)
            graph_pred_list.append(graph_pred_)
            graph_ys.append(graph_y)
            graph_ws.append(graph_w)
        
        graph_ys = np.stack(graph_ys).squeeze(1)
        graph_ws = np.stack(graph_ws).squeeze(1)

        if model.num_graph_tasks > 0:
            assert full_graph_graph_task_evaluate_fn is not None, \
                "Please specify `full_graph_graph_task_evaluate_fn` in the training kwargs"
            score_row.append("attn-score")
            score_row.extend([np.nanmean(np.array(list(voronoi_attn.values()))), np.nanmean(np.array(list(cell_type_attn.values())))])
            score_row.append("graph-score")
            score_row.extend(full_graph_graph_task_evaluate_fn(dataset_yw, graph_preds, task, print_res=False))

    if score_file is not None:
        with open(score_file, 'a') as f:
            f.write(",".join([s if isinstance(s, str) else ("%.3f" % s) for s in score_row]) + '\n')
    
    return score_row


def save_model_weight(model,
                      name,
                      task,
                      fold):
    os.makedirs('./ckpts/', exist_ok=True)
    model_tmp = deepcopy(model).cpu()
    torch.save(model_tmp.state_dict(), f'./ckpts/Mew_{name}_{task}_{fold}.pt')