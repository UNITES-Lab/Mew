import os
import torch
import src
from src.utils import setup_logger, seed_everything, preprocess_generalization
import numpy as np
import datetime
import argparse
import time
import re
from src.precomputing import PrecomputingBase
import gc

torch.set_num_threads(4)

def str2bool(s):
    if s not in {'False', 'True', 'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return (s == 'True') or (s == 'true')

def parse_args():
    parser = argparse.ArgumentParser(description='Mew')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--data', type=str, default='charville') # upmc, dfci, charville
    parser.add_argument('--task', type=str, default='classification') # classification, cox
    parser.add_argument('--use_node_features', metavar='S', type=str, nargs='+', default=['biomarker_expression', 'SIZE'])
    parser.add_argument('--shared', type=str2bool, default=False) # True, False
    parser.add_argument('--attn_weight', type=str2bool, default=False) # True, False
    parser.add_argument('--lr', type=float, default=0.0001) # 0.1, 0.01, 0.001, 0.0001
    parser.add_argument('--num_layers', type=int, default=4) # 1, 2, 3, 4
    parser.add_argument('--emb_dim', type=int, default=512) # 64, 128, 256, 512
    parser.add_argument('--batch_size', type=int, default=16) # 16, 32
    parser.add_argument('--drop_ratio', type=float, default=0.5) # 0.0, 0.25, 0.5
    parser.add_argument('--pool', type=str, default='mean')
    parser.add_argument('--num_epochs', type=int, default=1000) 
    parser.add_argument('--eval_epoch', type=int, default=100)
    parser.add_argument('--early_stop', type=int, default=300)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_model', type=str2bool, default=True)
    return parser.parse_known_args()

def main():
    args, _ = parse_args()
    filename = f'Mew.txt'
    if args.data == 'dfci':
        args.task = 'classification'

    os.makedirs(f'./log/{args.data}/{args.task}', exist_ok=True)
    logger = setup_logger('./', '-', f'./log/{args.data}/{args.task}/{filename}')
    logger.info(datetime.datetime.now())
    seed_everything(args.seed)

    print(f'Data: {args.data}, Task: {args.task}')

    # Settings
    data = args.data # 'upmc', 'charville'
    print(args.use_node_features)
    use_node_features = args.use_node_features
    device = f'cuda:{args.device}'

    data_root = './dataset/'+data + "_data" if data in ['upmc', 'charville'] else './dataset/general_data'
    print(data_root)
    raw_data_root = f"{data_root}/raw_data" 
    dataset_root = f"{data_root}/dataset_mew"
    graph_label_file = f"{data_root}/{data}_labels.csv"

    if data == "upmc":
        _common_cell_type_dict, _common_biomarker_list = None, None
        if args.task == 'classification':
            graph_tasks = ['primary_outcome', 'recurred', 'hpvstatus']
        
        elif args.task == 'cox':
            graph_tasks = [['survival_day', 'survival_status']]

    elif data == "charville":
        _common_cell_type_dict, _common_biomarker_list = None, None
        if args.task == 'classification':
            graph_tasks = ['primary_outcome', 'recurrence']    

        elif args.task == 'cox':
            graph_tasks = [['survival_legnth', 'survival_status'], ['recurrence_interval', 'recurrence_event']]

    else: # 'dfci'
        _common_cell_type_dict, _common_biomarker_list = preprocess_generalization('./dataset')
        graph_label_file1 = f"./dataset/upmc_data/upmc_labels.csv"
        graph_label_file2 = f"./dataset/dfci_data/dfci_labels.csv"
        graph_tasks1 = ['primary_outcome'] # from UPMC
        graph_tasks2 = ['pTR_label'] # from DFCI
        graph_tasks = graph_tasks2

    # Generate cellular graphs from raw inputs
    nx_graph_root = os.path.join(dataset_root, "graph")
    fig_save_root = os.path.join(dataset_root, "fig")
    model_save_root = os.path.join(dataset_root, "model")

    region_ids = set([f.split('.')[0] for f in os.listdir(raw_data_root)])
    os.makedirs(nx_graph_root, exist_ok=True)
    os.makedirs(fig_save_root, exist_ok=True)
    os.makedirs(model_save_root, exist_ok=True)

    # Save graph generated from each region
    for region_id in region_ids: 
        graph_output = os.path.join(nx_graph_root, "%s.gpkl" % region_id)
        if not os.path.exists(graph_output):
            print("Processing %s" % region_id)
            _voronoi_file=os.path.join(raw_data_root, "%s.json" % region_id) if data == 'upmc' else None
            G = src.construct_graph_for_region(
                region_id,
                cell_coords_file=os.path.join(raw_data_root, "%s.cell_data.csv" % region_id),
                cell_types_file=os.path.join(raw_data_root, "%s.cell_types.csv" % region_id),
                cell_biomarker_expression_file=os.path.join(raw_data_root, "%s.expression.csv" % region_id),
                cell_features_file=os.path.join(raw_data_root, "%s.cell_features.csv" % region_id),
                voronoi_file=_voronoi_file,
                graph_output=graph_output,
                voronoi_polygon_img_output=None,
                graph_img_output=None,
                common_cell_type_dict=_common_cell_type_dict,
                common_biomarker_list=_common_biomarker_list,
                figsize=10)

    # Define Cellular Graph Dataset
    dataset_kwargs = {
        'raw_folder_name': 'graph',
        'processed_folder_name': 'tg_graph',
        'node_features': ["cell_type", "SIZE", "biomarker_expression", "neighborhood_composition", "center_coord"],
        'edge_features': ["edge_type", "distance"],
        'cell_type_mapping': None,
        'cell_type_freq': None, 
        'biomarkers': None
    }
    feature_kwargs = {
        "biomarker_expression_process_method": "linear",
        "biomarker_expression_lower_bound": -2,
        "biomarker_expression_upper_bound": 3,
        "neighborhood_size": 10,
    }
    dataset_kwargs.update(feature_kwargs)
    dataset = src.CellularGraphDataset(dataset_root, **dataset_kwargs)
    
    # Define Transformers
    transformers = [
        src.FeatureMask(dataset,
                            use_center_node_features=use_node_features,
                            use_neighbor_node_features=use_node_features)
    ]
    if data in ['upmc', 'charville']:
        transformers.append(src.AddGraphLabel(graph_label_file, tasks=graph_tasks))
    else:
        transformers.append(src.AddTwoGraphLabel([graph_label_file1, graph_label_file2], tasks=[graph_tasks1, graph_tasks2]))

    dataset.set_transforms(transformers)

    # Precomputing & Label Extracting
    region_ids = [dataset.get_full(i).region_id for i in range(dataset.N)]
    coverslip_ids = [r_id.split('_')[1] for r_id in region_ids]
    
    num_feat = dataset[0]['cell'].x.shape[1]
    precomputer = PrecomputingBase(args.num_layers, device)
    sign_xs = precomputer.precompute(dataset)
    dataset_yw = [[dataset[i].graph_y, dataset[i].graph_w] for i in range(len(dataset))]
    del dataset
    gc.collect()

    # Define train/valid split
    if data == "upmc":
        fold0_coverslips = {'train': ['c001', 'c002', 'c004', 'c006'], 'val': ['c007'], 'test': ['c003', 'c005']}
        fold1_coverslips = {'train': ['c002', 'c003', 'c005', 'c007'], 'val': ['c004'], 'test': ['c001', 'c006']}
        fold2_coverslips = {'train': ['c001', 'c003', 'c004', 'c006'], 'val': ['c005'], 'test': ['c002', 'c007']}
    elif data == "charville":
        fold0_coverslips = {'train': ['c001', 'c002'], 'val': ['c003'], 'test': ['c004']}
        fold1_coverslips = {'train': ['c003', 'c004'], 'val': ['c002'], 'test': ['c001']}
        fold2_coverslips = {'train': ['c002', 'c004'], 'val': ['c001'], 'test': ['c003']}

    if data in ['upmc', 'charville']:
        split_indices = {}
        split_indices[0] = fold0_coverslips
        split_indices[1] = fold1_coverslips
        split_indices[2] = fold2_coverslips
        fold_list = [0,1,2]
    else:
        fold_list = [0]

    fold_results = []
    for fold in fold_list:
        # Define Model kwargs
        model_kwargs = {
            'num_layer': args.num_layers,
            'num_feat': num_feat,
            'emb_dim': args.emb_dim,
            'num_node_tasks': 0,
            'num_graph_tasks': len(graph_tasks),
            'node_embedding_output': 'last',
            'drop_ratio': args.drop_ratio,
            'graph_pooling': args.pool,
            'attn_weight': args.attn_weight,
            'shared': args.shared,
        }
        # Define Train and Evaluate kwargs
        train_kwargs = {
            'batch_size': args.batch_size,
            'lr': args.lr,
            'graph_loss_weight': 1.0,
            'num_iterations': args.num_epochs,
            'early_stop': args.early_stop,
            'node_task_loss_fn': None,
            'graph_task_loss_fn': src.models.BinaryCrossEntropy() if args.task == 'classification' else src.models.CoxSGDLossFn(),
            'evaluate_fn': [src.train.evaluate_by_full_graph, src.train.save_model_weight],
            'evaluate_freq': args.eval_epoch,
        }

        evaluate_kwargs = {
            'shuffle': True,
            'full_graph_graph_task_evaluate_fn': src.inference.full_graph_graph_classification_evaluate_fn,
            'score_file': os.path.join(model_save_root, 'Mew-%s-%s-%d.txt' % (graph_tasks[0], '_'.join(use_node_features), fold)),
            'model_folder': os.path.join(model_save_root, 'Mew'),
        }
        
        train_kwargs.update(evaluate_kwargs)
        os.makedirs(evaluate_kwargs['model_folder'], exist_ok=True)

        train_inds = []
        valid_inds = []
        test_inds = []

        if data in ['upmc', 'charville']:
            for i, cs_id in enumerate(coverslip_ids):
                if cs_id in split_indices[fold]['train']:
                    train_inds.append(i)
                elif cs_id in split_indices[fold]['val']:
                    valid_inds.append(i)
                else:
                    test_inds.append(i)
        else:
            for i, region_id in enumerate(region_ids):
                if 'UPMC' in region_id:
                    if 'c007' in region_id:
                        valid_inds.append(i)    
                    else:
                        train_inds.append(i)
                else:
                    test_inds.append(i)
            
        start_time = time.time()
        model = src.models.SIGN_pred(**model_kwargs)
        model = model.to(device)

        model, fold_result = src.train.train_full_graph(
            model, dataset_yw, sign_xs, device, filename, fold, logger,
            train_inds=train_inds, valid_inds=valid_inds, test_inds=test_inds, task=args.task, name=data, save_model=args.save_model, **train_kwargs)

        if fold_result == -1:
            print(f'The learning rate: {args.lr} is too high, causing numerical instability. Please try using a lower learning rate')
            break

        performance_list = re.findall(r"[-+]?\d*\.\d+|\d+", fold_result)
        fold_results.append([float(performance) for performance in performance_list[-len(graph_tasks):]])
    
        if fold == fold_list[-1]:
            averages = [round(sum(pair)/len(pair), 3) for pair in zip(*fold_results)]
            std_devs = [round(np.std(pair), 3) for pair in zip(*fold_results)]
            
            averages_std_combined = [f"{avg} ± {std}" for avg, std in zip(averages, std_devs)]
            combined_str = ', '.join(averages_std_combined)
            logger.info(f'[*Fold Average*] Total Epoch: {args.num_epochs}, Valid, graph-score, [{combined_str}]')

        print(f"Fold: {fold}")
        print("Total time elapsed: {:.4f}s".format(time.time() - start_time))
        logger.info("Filename: {}, Total time elapsed: {:.4f}s, Args: {}".format(filename, time.time() - start_time, args))
        logger.info("")

if __name__ == '__main__':
    main()