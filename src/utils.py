import numpy as np
import pickle
import torch
import os
import sys
import random
import logging
import numpy as np
import pandas as pd

MB = 1024 ** 2
GB = 1024 ** 3

EDGE_TYPES = {
    "neighbor": 0,
    "distant": 1,
    "self": 2,
}

# Metadata for the example dataset
BIOMARKERS_UPMC = [
    "CD11b", "CD14", "CD15", "CD163", "CD20", "CD21", "CD31", "CD34", "CD3e",
    "CD4", "CD45", "CD45RA", "CD45RO", "CD68", "CD8", "CollagenIV", "HLA-DR",
    "Ki67", "PanCK", "Podoplanin", "Vimentin", "aSMA",
]

CELL_TYPE_MAPPING_UPMC = {
    'APC': 0,
    'B cell': 1,
    'CD4 T cell': 2,
    'CD8 T cell': 3,
    'Granulocyte': 4,
    'Lymph vessel': 5,
    'Macrophage': 6,
    'Naive immune cell': 7,
    'Stromal / Fibroblast': 8,
    'Tumor': 9,
    'Tumor (CD15+)': 10,
    'Tumor (CD20+)': 11,
    'Tumor (CD21+)': 12,
    'Tumor (Ki67+)': 13,
    'Tumor (Podo+)': 14,
    'Vessel': 15,
    'Unassigned': 16,
}

CELL_TYPE_FREQ_UPMC = {
    'APC': 0.038220815854819415,
    'B cell': 0.06635091324932002,
    'CD4 T cell': 0.09489001514723677,
    'CD8 T cell': 0.07824503590797544,
    'Granulocyte': 0.026886102677111563,
    'Lymph vessel': 0.006429085023448621,
    'Macrophage': 0.10251942892685563,
    'Naive immune cell': 0.033537398925429215,
    'Stromal / Fibroblast': 0.07692583870182068,
    'Tumor': 0.10921293560435145,
    'Tumor (CD15+)': 0.06106975782857908,
    'Tumor (CD20+)': 0.02098925720318548,
    'Tumor (CD21+)': 0.053892044158901406,
    'Tumor (Ki67+)': 0.13373768013421947,
    'Tumor (Podo+)': 0.06276108605978743,
    'Vessel': 0.034332604596958326,
    'Unassigned': 0.001,
}


def setup_logger(save_dir, text, filename = 'log.txt'):
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger(text)
    logger.setLevel(4)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    logger.info("======================================================================================")

    return logger

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_initials(string_list):
    initials = "".join(word[0] for word in string_list if word)  # Ensure the word is not empty
    return initials.lower() 

def generate_charville_label(df):
    ## recurrence_interval & recurrence_event 
    # Convert 'surgery_date' to datetime & Convert 'first_recurrence_date' to datetime, handling 'NONE' cases
    df['surgery_date'] = pd.to_datetime(df['surgery_date'])
    df['first_recurrence_date'] = pd.to_datetime(df['first_recurrence_date'], errors='coerce')
    df['last_contact_date'] = pd.to_datetime(df['last_contact_date'], errors='coerce')

    # Calculate the recurrence interval as the number of days from 'surgery_date' to 'first_recurrence_date'
    df['recurrence_interval'] = (df['first_recurrence_date'] - df['surgery_date']).dt.days

    # Recurrence event: 1 if there is a recurrence, 0 if 'NONE/DISEASE FREE' or 'NEVER DISEASE FREE'
    df['recurrence_event'] = df['type_of_first_recurrence'].apply(lambda x: 0 if 'FREE' in x else 1)
    df.loc[df['first_recurrence_date'].isna(), 'recurrence_interval'] = (df['last_contact_date'] - df['surgery_date']).dt.days

    df.loc[df['first_recurrence_date'].isna(), 'recurrence_interval'] = (df['last_contact_date'] - df['surgery_date']).dt.days

    ## survival_legnth & survival_status
    def calculate_length(row):
        # If the 'first_recurrence_date' is not missing, use it to calculate the length
        if pd.notna(row['first_recurrence_date']):
            return (row['first_recurrence_date'] - row['surgery_date']).days / 30.4375  # Average days per month
        # If 'first_recurrence_date' is missing, but 'last_contact_date' is not, use 'last_contact_date'
        elif pd.notna(row['last_contact_date']):
            return (row['last_contact_date'] - row['surgery_date']).days / 30.4375
        # If both are missing, return NaN
        else:
            return pd.NA

    # Apply the function to the rows where 'length_of_disease_free_survival' is missing
    df['survival_legnth'] = df.apply(calculate_length, axis=1)
    df['survival_legnth'] = np.round(pd.to_numeric(df['survival_legnth'], errors='coerce'))

    # Survival event: 1 if patient survives, 0 if dead
    df['survival_status'] = df['alive_or_deceased'].apply(lambda x: 1 if 'Dead' in x else 0)

    return df

def preprocess_generalization(root='./dataset'):
    upmc_biomarker_cols = set(pd.read_csv(f'{root}/upmc_data/raw_data/UPMC_c001_v001_r001_reg001.expression.csv').columns)
    dfci_biomarker_cols = set(pd.read_csv(f'{root}/dfci_data/raw_data/s271_c001_v001_r001_reg001.expression.csv').columns)

    common_biomarker_cols_list = list(upmc_biomarker_cols & dfci_biomarker_cols)
    common_cell_type_dict = {
        'APC': 'APC',
        'Dendritic cell': 'APC',
        'APC/macrophage': 'APC',
        'B cell': 'B cell',
        'CD4 T cell': 'CD4 T cell',
        'T cell (CD45RO+/FoxP3+/ICOS+)': 'CD4 T cell',
        'CD4 T cell (ICOS+/FoxP3+)': 'CD4 T cell',
        'CD8 T cell': 'CD8 T cell',
        'T cell (GranzymeB+/LAG3+)': 'CD8 T cell',
        'Granulocyte': 'Granulocyte',
        'Lymph vessel': 'Vessel cell',
        'Macrophage': 'Macrophage',
        'Naive immune cell': 'Naive immune cell',
        'Naive B cell': 'Naive immune cell',
        'Naive lymphocyte (CD45RA+/CD38+)': 'Naive immune cell',
        'Stromal / Fibroblast': 'Stromal cell',
        'Stroma': 'Stromal cell',
        'Tumor': 'Tumor',
        'Tumor (CD15+)': 'Tumor',
        'Tumor (CD20+)': 'Tumor',
        'Tumor (CD21+)': 'Tumor',
        'Tumor (Ki67+)': 'Tumor (Ki67+)',
        'Tumor (Podo+)': 'Tumor',
        'Tumor (PanCK hi)': 'Tumor',
        'Tumor (PanCK low)': 'Tumor',
        'Unassigned': 'Other cell',
        'NK cell': 'Other cell',
        'Mast cell': 'Other cell',
        'Unknown (TCF1+)': 'Other cell',
        'Unclassified': 'Other cell',
        'Vessel': 'Vessel cell',
        'Vessel endothelium': 'Vessel cell',
    }

    return common_cell_type_dict, common_biomarker_cols_list

def get_cell_type_metadata(nx_graph_files):
    """Find all unique cell types from a list of cellular graphs

    Args:
        nx_graph_files (list/str): path/list of paths to cellular graph files (gpickle)

    Returns:
        cell_type_mapping (dict): mapping of unique cell types to integer indices
        cell_type_freq (dict): mapping of unique cell types to their frequency
    """
    if isinstance(nx_graph_files, str):
        nx_graph_files = [nx_graph_files]
    cell_type_mapping = {}
    for g_f in nx_graph_files:
        try:
            G = pickle.load(open(g_f, 'rb'))
        except:
            print('Error detected! in file:', g_f)
        assert 'cell_type' in G['layer_1'].nodes[0]
        for n in G['layer_1'].nodes:
            ct = G['layer_1'].nodes[n]['cell_type']
            if ct not in cell_type_mapping:
                cell_type_mapping[ct] = 0
            cell_type_mapping[ct] += 1
    unique_cell_types = sorted(cell_type_mapping.keys())
    unique_cell_types_ct = [cell_type_mapping[ct] for ct in unique_cell_types]
    unique_cell_type_freq = [count / sum(unique_cell_types_ct) for count in unique_cell_types_ct]
    cell_type_mapping = {ct: i for i, ct in enumerate(unique_cell_types)}
    cell_type_freq = dict(zip(unique_cell_types, unique_cell_type_freq))
    return cell_type_mapping, cell_type_freq


def get_biomarker_metadata(nx_graph_files):
    """Load all biomarkers from a list of cellular graphs

    Args:
        nx_graph_files (list/str): path/list of paths to cellular graph files (gpickle)

    Returns:
        shared_bms (list): list of biomarkers shared by all cells (intersect)
        all_bms (list): list of all biomarkers (union)
    """
    if isinstance(nx_graph_files, str):
        nx_graph_files = [nx_graph_files]
    all_bms = set()
    shared_bms = None
    for g_f in nx_graph_files:
        G = pickle.load(open(g_f, 'rb'))
        for n in G['layer_1'].nodes:
            bms = sorted(G['layer_1'].nodes[n]["biomarker_expression"].keys())
            for bm in bms:
                all_bms.add(bm)
            valid_bms = [
                bm for bm in bms if G['layer_1'].nodes[n]["biomarker_expression"][bm] == G['layer_1'].nodes[n]["biomarker_expression"][bm]]
            shared_bms = set(valid_bms) if shared_bms is None else shared_bms & set(valid_bms)
    shared_bms = sorted(shared_bms)
    all_bms = sorted(all_bms)
    return shared_bms, all_bms


def get_graph_splits(dataset,
                     split='random',
                     cv_k=5,
                     seed=None,
                     fold_mapping=None):
    """ Define train/valid split

    Args:
        dataset (CellularGraphDataset): dataset to split
        split (str): split method, one of 'random', 'fold'
        cv_k (int): number of splits for random split
        seed (int): random seed
        fold_mapping (dict): mapping of region ids to folds,
            fold could be coverslip, patient, etc.

    Returns:
        split_inds (list): fold indices for each region in the dataset
    """
    splits = {}
    region_ids = set([dataset.get_full(i).region_id for i in range(dataset.N)])
    _region_ids = sorted(region_ids)
    if split == 'random':
        if seed is not None:
            np.random.seed(seed)
        if fold_mapping is None:
            fold_mapping = {region_id: region_id for region_id in _region_ids}
        # `_ids` could be sample ids / patient ids / certain properties
        _folds = sorted(set(list(fold_mapping.values())))
        np.random.shuffle(_folds)
        cv_shard_size = len(_folds) / cv_k
        for i, region_id in enumerate(_region_ids):
            splits[region_id] = _folds.index(fold_mapping[region_id]) // cv_shard_size
    elif split == 'fold':
        # Split into folds, one fold per group
        assert fold_mapping is not None
        _folds = sorted(set(list(fold_mapping.values())))
        for i, region_id in enumerate(_region_ids):
            splits[region_id] = _folds.index(fold_mapping[region_id])
    else:
        raise ValueError("split mode not recognized")

    split_inds = []
    for i in range(dataset.N):
        split_inds.append(splits[dataset.get_full(i).region_id])
    return split_inds

def get_memory_usage(gpu, print_info=False):
    """Get accurate gpu memory usage by querying torch runtime"""
    allocated = torch.cuda.memory_allocated(gpu)
    reserved = torch.cuda.memory_reserved(gpu)
    if print_info:
        print("allocated: %.2f MB" % (allocated / 1024 / 1024), flush=True)
        print("reserved:  %.2f MB" % (reserved / 1024 / 1024), flush=True)
    return allocated

def compute_tensor_bytes(tensors):
    """Compute the bytes used by a list of tensors"""
    if not isinstance(tensors, (list, tuple)):
        tensors = [tensors]

    ret = 0
    for x in tensors:
        if x.dtype in [torch.int64, torch.long]:
            ret += np.prod(x.size()) * 8
        if x.dtype in [torch.float32, torch.int, torch.int32]:
            ret += np.prod(x.size()) * 4
        elif x.dtype in [torch.bfloat16, torch.float16, torch.int16]:
            ret += np.prod(x.size()) * 2
        elif x.dtype in [torch.int8]:
            ret += np.prod(x.size())
        else:
            print(x.dtype)
            raise ValueError()
    return ret

