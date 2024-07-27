import os
import numpy as np
import pandas as pd
import json
import pickle
import matplotlib
import matplotlib.path as mplPath
import matplotlib.pyplot as plt
import networkx as nx
import warnings
from scipy.spatial import Delaunay
from itertools import combinations

RADIUS_RELAXATION = 0.1
NEIGHBOR_EDGE_CUTOFF = 55  # distance cutoff for neighbor edges, 55 pixels~20 um


def plot_voronoi_polygons(voronoi_polygons, voronoi_polygon_colors=None):
    """Plot voronoi polygons for the cellular graph

    Args:
        voronoi_polygons (nx.Graph/list): cellular graph or list of voronoi polygons
        voronoi_polygon_colors (list): list of colors for voronoi polygons
    """
    if isinstance(voronoi_polygons, nx.Graph):
        voronoi_polygons = [voronoi_polygons.nodes[n]['voronoi_polygon'] for n in voronoi_polygons.nodes]

    if voronoi_polygon_colors is None:
        voronoi_polygon_colors = ['w'] * len(voronoi_polygons)
    assert len(voronoi_polygon_colors) == len(voronoi_polygons)

    xmax = 0
    ymax = 0
    for polygon, polygon_color in zip(voronoi_polygons, voronoi_polygon_colors):
        x, y = polygon[:, 0], polygon[:, 1]
        plt.fill(x, y, facecolor=polygon_color, edgecolor='k', linewidth=0.5)
        xmax = max(xmax, x.max())
        ymax = max(ymax, y.max())

    plt.xlim(0, xmax)
    plt.ylim(0, ymax)
    return


def plot_graph(G, node_colors=None, cell_type=False):
    """Plot dot-line graph for the cellular graph

    Args:
        G (nx.Graph): full cellular graph of the region
        node_colors (list): list of node colors. Defaults to None.
    """
    # Extract basic node attributes
    node_coords = [G.nodes[n]['center_coord'] for n in G.nodes]
    node_coords = np.stack(node_coords, 0)

    if node_colors is None:
        unique_cell_types = sorted(set([G.nodes[n]['cell_type'] for n in G.nodes]))
        cell_type_to_color = {ct: matplotlib.cm.get_cmap("tab20")(i % 20) for i, ct in enumerate(unique_cell_types)}
        node_colors = [cell_type_to_color[G.nodes[n]['cell_type']] for n in G.nodes]
    assert len(node_colors) == node_coords.shape[0]

    for (i, j, edge_type) in G.edges.data():
        xi, yi = G.nodes[i]['center_coord']
        xj, yj = G.nodes[j]['center_coord']
        if cell_type:
            plotting_kwargs = {"c": "k",
                               "linewidth": 1,
                               "linestyle": '-'}
        else:
            if edge_type['edge_type'] == 'neighbor':
                plotting_kwargs = {"c": "k",
                                "linewidth": 1,
                                "linestyle": '-'}
            else:
                plotting_kwargs = {"c": (0.4, 0.4, 0.4, 1.0),
                                "linewidth": 0.3,
                                "linestyle": '--'}
        plt.plot([xi, xj], [yi, yj], zorder=1, **plotting_kwargs)

    plt.scatter(node_coords[:, 0],
                node_coords[:, 1],
                s=10,
                c=node_colors,
                linewidths=0.3,
                zorder=2)
    plt.xlim(0, node_coords[:, 0].max() * 1.01)
    plt.ylim(0, node_coords[:, 1].max() * 1.01)
    return


def load_cell_coords(cell_coords_file):
    """Load cell coordinates from file

    Args:
        cell_coords_file (str): path to csv file containing cell coordinates

    Returns:
        pd.DataFrame: dataframe containing cell coordinates, columns ['CELL_ID', 'X', 'Y']
    """
    df = pd.read_csv(cell_coords_file)
    df.columns = [c.upper() for c in df.columns]
    assert 'X' in df.columns, "Cannot find column for X coordinates"
    assert 'Y' in df.columns, "Cannot find column for Y coordinates"
    if 'CELL_ID' not in df.columns:
        warnings.warn("Cannot find column for cell id, using index as cell id")
        df['CELL_ID'] = df.index
    return df[['CELL_ID', 'X', 'Y']]


def load_cell_types(cell_types_file):
    """Load cell types from file

    Args:
        cell_types_file (str): path to csv file containing cell types

    Returns:
        pd.DataFrame: dataframe containing cell types, columns ['CELL_ID', 'CELL_TYPE']
    """
    df = pd.read_csv(cell_types_file)
    df.columns = [c.upper() for c in df.columns]

    cell_type_column = [c for c in df.columns if c != 'CELL_ID']
    if len(cell_type_column) == 1:
        cell_type_column = cell_type_column[0]
    elif 'CELL_TYPE' in cell_type_column:
        cell_type_column = 'CELL_TYPE'
    elif 'CELL_TYPES' in cell_type_column:
        cell_type_column = 'CELL_TYPES'
    else:
        raise ValueError("Please rename the column for cell type as 'CELL_TYPE'")

    if 'CELL_ID' not in df.columns:
        warnings.warn("Cannot find column for cell id, using index as cell id")
        df['CELL_ID'] = df.index
    _df = df[['CELL_ID', cell_type_column]]
    _df.columns = ['CELL_ID', 'CELL_TYPE']  # rename columns for clarity
    return _df


def load_cell_biomarker_expression(cell_biomarker_expression_file):
    """Load cell biomarker expression from file

    Args:
        cell_biomarker_expression_file (str): path to csv file containing cell biomarker expression

    Returns:
        pd.DataFrame: dataframe containing cell biomarker expression,
            columns ['CELL_ID', 'BM-<biomarker1_name>', 'BM-<biomarker2_name>', ...]
    """
    df = pd.read_csv(cell_biomarker_expression_file)
    df.columns = [c.upper() for c in df.columns]
    biomarkers = sorted([c for c in df.columns if c != 'CELL_ID'])
    for bm in biomarkers:
        if df[bm].dtype not in [np.dtype(int), np.dtype(float), np.dtype('float64')]:
            warnings.warn("Skipping column %s as it is not numeric" % bm)
            biomarkers.remove(bm)

    if 'CELL_ID' not in df.columns:
        warnings.warn("Cannot find column for cell id, using index as cell id")
        df['CELL_ID'] = df.index
    _df = df[['CELL_ID'] + biomarkers]
    _df.columns = ['CELL_ID'] + ['BM-%s' % bm for bm in biomarkers]
    return _df


def load_cell_features(cell_features_file):
    """Load additional cell features from file

    Args:
        cell_features_file (str): path to csv file containing additional cell features

    Returns:
        pd.DataFrame: dataframe containing cell features
            columns ['CELL_ID', '<feature1_name>', '<feature2_name>', ...]
    """
    df = pd.read_csv(cell_features_file)
    df.columns = [c.upper() for c in df.columns]

    feature_columns = sorted([c for c in df.columns if c != 'CELL_ID'])
    for feat in feature_columns:
        if df[feat].dtype not in [np.dtype(int), np.dtype(float), np.dtype('float64')]:
            warnings.warn("Skipping column %s as it is not numeric" % feat)
            feature_columns.remove(feat)

    if 'CELL_ID' not in df.columns:
        warnings.warn("Cannot find column for cell id, using index as cell id")
        df['CELL_ID'] = df.index

    return df[['CELL_ID'] + feature_columns]


def read_raw_voronoi(voronoi_file):
    """Read raw coordinates of voronoi polygons from file

    Args:
        voronoi_file (str): path to the voronoi polygon file

    Returns:
        voronoi_polygons (list): list of voronoi polygons,
            represented by the coordinates of their exterior vertices
    """
    if voronoi_file.endswith('json'):
        with open(voronoi_file) as f:
            raw_voronoi_polygons = json.load(f)
    elif voronoi_file.endswith('.pkl'):
        with open(voronoi_file, 'rb') as f:
            raw_voronoi_polygons = pickle.load(f)

    voronoi_polygons = []
    for i, polygon in enumerate(raw_voronoi_polygons):
        if isinstance(polygon, list):
            polygon = np.array(polygon).reshape((-1, 2))
        elif isinstance(polygon, dict):
            assert len(polygon) == 1
            polygon = list(polygon.values())[0]
            polygon = np.array(polygon).reshape((-1, 2))
        voronoi_polygons.append(polygon)
    return voronoi_polygons


def calcualte_voronoi_from_coords(x, y, xmax=None, ymax=None):
    """Calculate voronoi polygons from a set of points

    Points are assumed to have coordinates in ([0, xmax], [0, ymax])

    Args:
        x (array-like): x coordinates of points
        y (array-like): y coordinates of points
        xmax (float): maximum x coordinate
        ymax (float): maximum y coordinate

    Returns:
        voronoi_polygons (list): list of voronoi polygons,
            represented by the coordinates of their exterior vertices
    """
    from geovoronoi import voronoi_regions_from_coords
    from shapely import geometry
    xmax = 1.01 * max(x) if xmax is None else xmax
    ymax = 1.01 * max(y) if ymax is None else ymax
    boundary = geometry.Polygon([[0, 0], [xmax, 0], [xmax, ymax], [0, ymax]])
    coords = np.stack([
        np.array(x).reshape((-1,)),
        np.array(y).reshape((-1,))], 1)
    region_polys, _ = voronoi_regions_from_coords(coords, boundary)
    voronoi_polygons = [np.array(list(region_polys[k].exterior.coords)) for k in region_polys]
    return voronoi_polygons


def build_graph_from_cell_coords(cell_data, voronoi_polygons):
    """Construct a networkx graph based on cell coordinates

    Args:
        cell_data (pd.DataFrame): dataframe containing cell data,
            columns ['CELL_ID', 'X', 'Y', ...]
        voronoi_polygons (list): list of voronoi polygons,
            represented by the coordinates of their exterior vertices

    Returns:
        G (nx.Graph): full cellular graph of the region
    """
    save_polygon = True
    if not len(cell_data) == len(voronoi_polygons):
        warnings.warn("Number of cells does not match number of voronoi polygons")
        save_polygon = False

    coord_ar = np.array(cell_data[['CELL_ID', 'X', 'Y']])
    G = nx.Graph()
    node_to_cell_mapping = {}
    for i, row in enumerate(coord_ar):
        vp = voronoi_polygons[i] if save_polygon else None
        G.add_node(i, voronoi_polygon=vp)
        node_to_cell_mapping[i] = row[0]

    dln = Delaunay(coord_ar[:, 1:3])
    neighbors = [set() for _ in range(len(coord_ar))]
    for t in dln.simplices:
        for v in t:
            neighbors[v].update(t)
    
    for i, ns in enumerate(neighbors):
        for n in ns:
            G.add_edge(int(i), int(n))

    return G, node_to_cell_mapping


def build_graph_from_voronoi_polygons(voronoi_polygons, radius_relaxation=RADIUS_RELAXATION):
    """Construct a networkx graph based on voronoi polygons

    Args:
        voronoi_polygons (list): list of voronoi polygons,
            represented by the coordinates of their exterior vertices

    Returns:
        G (nx.Graph): full cellular graph of the region
    """
    G = nx.Graph()

    polygon_vertices = []
    vertice_identities = []
    for i, polygon in enumerate(voronoi_polygons):
        G.add_node(i, voronoi_polygon=polygon)
        polygon_vertices.append(polygon)
        vertice_identities.append(np.ones((polygon.shape[0],)) * i)

    polygon_vertices = np.concatenate(polygon_vertices, 0)
    vertice_identities = np.concatenate(vertice_identities, 0).astype(int)
    for i, polygon in enumerate(voronoi_polygons):
        path = mplPath.Path(polygon)
        points_inside = np.where(path.contains_points(polygon_vertices, radius=radius_relaxation) +
                                 path.contains_points(polygon_vertices, radius=-radius_relaxation))[0]
        id_inside = set(vertice_identities[points_inside])
        for j in id_inside:
            if j > i:
                G.add_edge(int(i), int(j))
    return G


def build_voronoi_polygon_to_cell_mapping(G, voronoi_polygons, cell_data):
    """Construct 1-to-1 mapping between voronoi polygons and cells

    Args:
        G (nx.Graph): full cellular graph of the region
        voronoi_polygons (list): list of voronoi polygons,
            represented by the coordinates of their exterior vertices
        cell_data (pd.DataFrame): dataframe containing cellular data

    Returns:
        voronoi_polygon_to_cell_mapping (dict): 1-to-1 mapping between
            polygon index (also node index in `G`) and cell id
    """
    cell_coords = np.array(list(zip(cell_data['X'], cell_data['Y']))).reshape((-1, 2))
    # Fetch all cells within each polygon
    cells_in_polygon = {}
    for i, polygon in enumerate(voronoi_polygons):
        path = mplPath.Path(polygon)
        _cell_ids = cell_data.iloc[np.where(path.contains_points(cell_coords))[0]]
        _cells = list(_cell_ids[['CELL_ID', 'X', 'Y']].values)
        cells_in_polygon[i] = _cells

    def get_point_reflection(c1, c2, c3):
        # Reflection of point c1 across line defined by c2 & c3
        x1, y1 = c1
        x2, y2 = c2
        x3, y3 = c3
        if x2 == x3:
            return (2 * x2 - x1, y1)
        m = (y3 - y2) / (x3 - x2)
        c = (x3 * y2 - x2 * y3) / (x3 - x2)
        d = (float(x1) + (float(y1) - c) * m) / (1 + m**2)
        x4 = 2 * d - x1
        y4 = 2 * d * m - y1 + 2 * c
        return (x4, y4)

    # Establish 1-to-1 mapping between polygons and cell ids
    voronoi_polygon_to_cell_mapping = {}
    for i, polygon in enumerate(voronoi_polygons):
        path = mplPath.Path(polygon)
        if len(cells_in_polygon[i]) == 1:
            # A single polygon contains a single cell centroid, assign cell id
            voronoi_polygon_to_cell_mapping[i] = cells_in_polygon[i][0][0]

        elif len(cells_in_polygon[i]) == 0:
            # Skipping polygons that do not contain any cell centroids
            continue

        else:
            # A single polygon contains multiple cell centroids
            polygon_edges = [(polygon[_i], polygon[_i + 1]) for _i in range(-1, len(polygon) - 1)]
            # Use the reflection of neighbor polygon's center cell
            neighbor_cells = sum([cells_in_polygon[j] for j in G.neighbors(i)], [])
            reflection_points = np.concatenate(
                [[get_point_reflection(cell[1:], edge[0], edge[1]) for edge in polygon_edges]
                    for cell in neighbor_cells], 0)
            reflection_points = reflection_points[np.where(path.contains_points(reflection_points))]
            # Reflection should be very close to the center cell
            dists = [((reflection_points - c[1:])**2).sum(1).min(0) for c in cells_in_polygon[i]]
            if not np.min(dists) < 0.01:
                warnings.warn("Cannot find the exact center cell for polygon %d" % i)
            voronoi_polygon_to_cell_mapping[i] = cells_in_polygon[i][np.argmin(dists)][0]
            
    return voronoi_polygon_to_cell_mapping


def assign_attributes(G, cell_data, node_to_cell_mapping, _assert=True):
    """Assign node and edge attributes to the cellular graph

    Args:
        G (nx.Graph): full cellular graph of the region
        cell_data (pd.DataFrame): dataframe containing cellular data
        node_to_cell_mapping (dict): 1-to-1 mapping between
            node index in `G` and cell id

    Returns:
        nx.Graph: populated cellular graph
    """
    if _assert:
        assert set(G.nodes) == set(node_to_cell_mapping.keys())
    biomarkers = sorted([c for c in cell_data.columns if c.startswith('BM-')])

    additional_features = sorted([
        c for c in cell_data.columns if c not in biomarkers + ['CELL_ID', 'X', 'Y', 'CELL_TYPE']])

    cell_to_node_mapping = {v: k for k, v in node_to_cell_mapping.items()}
    node_properties = {}
    for _, cell_row in cell_data.iterrows():
        cell_id = cell_row['CELL_ID']
        if cell_id not in cell_to_node_mapping:
            continue
        node_index = cell_to_node_mapping[cell_id]
        p = {"cell_id": cell_id}
        p["center_coord"] = (cell_row['X'], cell_row['Y'])
        if "CELL_TYPE" in cell_row:
            p["cell_type"] = cell_row["CELL_TYPE"]
        else:
            p["cell_type"] = "Unassigned"
        biomarker_expression_dict = {bm.split('BM-')[1]: cell_row[bm] for bm in biomarkers}
        p["biomarker_expression"] = biomarker_expression_dict
        for feat_name in additional_features:
            p[feat_name] = cell_row[feat_name]
        node_properties[node_index] = p
    
    G = G.subgraph(node_properties.keys())
    nx.set_node_attributes(G, node_properties)

    # Add distance, edge type (by thresholding) to edge feature
    edge_properties = get_edge_type(G)
    nx.set_edge_attributes(G, edge_properties)

    return G


def get_edge_type(G, neighbor_edge_cutoff=NEIGHBOR_EDGE_CUTOFF):
    """Define neighbor vs distant edges based on distance

    Args:
        G (nx.Graph): full cellular graph of the region
        neighbor_edge_cutoff (float): distance cutoff for neighbor edges.
            By default we use 55 pixels (~20 um)

    Returns:
        dict: edge properties
    """
    edge_properties = {}
    for (i, j) in G.edges:
        ci = G.nodes[i]['center_coord']
        cj = G.nodes[j]['center_coord']
        dist = np.linalg.norm(np.array(ci) - np.array(cj), ord=2)
        edge_properties[(i, j)] = {
            "distance": dist,
            "edge_type": "neighbor" if dist < neighbor_edge_cutoff else "distant"
        }
        '''
        if ('center_coord' in G.nodes[i].keys()) & ('center_coord' in G.nodes[j].keys()):
            ci = G.nodes[i]['center_coord']
            cj = G.nodes[j]['center_coord']
            dist = np.linalg.norm(np.array(ci) - np.array(cj), ord=2)
            edge_properties[(i, j)] = {
                "distance": dist,
                "edge_type": "neighbor" if dist < neighbor_edge_cutoff else "distant"
            }
        else:
            G.remove_edge(i, j)
            # if 'center_coord' not in G.nodes[i].keys():
            #     G.remove_node(i)
            # else:
            #     G.remove_node(j)

            # G.remove_edge(i, j)
        '''
    return edge_properties


def merge_cell_dataframes(df1, df2):
    """Merge two cell dataframes on shared rows (cells)"""
    if set(df2['CELL_ID']) != set(df1['CELL_ID']):
        warnings.warn("Cell ids in the two dataframes do not match")
    shared_cell_ids = set(df2['CELL_ID']).intersection(set(df1['CELL_ID']))
    df1 = df1[df1['CELL_ID'].isin(shared_cell_ids)]
    df1 = df1.merge(df2, on='CELL_ID')
    return df1


def construct_graph_for_region(region_id,
                               cell_coords_file=None,
                               cell_types_file=None,
                               cell_biomarker_expression_file=None,
                               cell_features_file=None,
                               voronoi_file=None,
                               graph_source='polygon',
                               graph_output=None,
                               voronoi_polygon_img_output=None,
                               graph_img_output=None,
                               common_cell_type_dict=None,
                               common_biomarker_list=None,
                               figsize=10):
    """Construct cellular graph for a region

    Args:
        region_id (str): region id
        cell_coords_file (str): path to csv file containing cell coordinates
        cell_types_file (str): path to csv file containing cell types/annotations
        cell_biomarker_expression_file (str): path to csv file containing cell biomarker expression
        cell_features_file (str): path to csv file containing additional cell features
            Note that features stored in this file can only be numeric and
            will be saved and used as is.
        voronoi_file (str): path to the voronoi coordinates file
        graph_source (str): source of edges in the graph, either "polygon" or "cell"
        graph_output (str): path for saving cellular graph as gpickle
        voronoi_polygon_img_output (str): path for saving voronoi image
        graph_img_output (str): path for saving dot-line graph image
        figsize (int): figure size for plotting

    Returns:
        G (nx.Graph): full cellular graph of the region
    """
    assert cell_coords_file is not None, "cell coordinates must be provided"
    cell_data = load_cell_coords(cell_coords_file)

    if voronoi_file is None:
        # Calculate voronoi polygons based on cell coordinates
        voronoi_polygons = calcualte_voronoi_from_coords(cell_data['X'], cell_data['Y'])
    else:
        # Load voronoi polygons from file
        voronoi_polygons = read_raw_voronoi(voronoi_file)

    if cell_types_file is not None:
        # Load cell types
        cell_types = load_cell_types(cell_types_file)
        if common_cell_type_dict != None:
            cell_types['CELL_TYPE'] = cell_types['CELL_TYPE'].map(common_cell_type_dict)
        cell_data = merge_cell_dataframes(cell_data, cell_types)

    if cell_biomarker_expression_file is not None:
        # Load cell biomarker expression
        cell_expression = load_cell_biomarker_expression(cell_biomarker_expression_file)
        if common_biomarker_list != None:
            columns_to_keep = ['CELL_ID'] + ['BM-' + suffix.upper() for suffix in common_biomarker_list if suffix != 'CELL_ID']
            cell_expression = cell_expression[columns_to_keep]
        cell_data = merge_cell_dataframes(cell_data, cell_expression)

    if cell_features_file is not None:
        # Load additional cell features
        additional_cell_features = load_cell_features(cell_features_file)
        cell_data = merge_cell_dataframes(cell_data, additional_cell_features)

    if graph_source == 'polygon':
        # Build initial cellular graph
        G = build_graph_from_voronoi_polygons(voronoi_polygons)
        # Construct matching between voronoi polygons and cells
        node_to_cell_mapping = build_voronoi_polygon_to_cell_mapping(G, voronoi_polygons, cell_data)
        # Prune graph to contain only voronoi polygons that have corresponding cells
        G = G.subgraph(node_to_cell_mapping.keys())
    elif graph_source == 'cell':
        G, node_to_cell_mapping = build_graph_from_cell_coords(cell_data, voronoi_polygons)
    else:
        raise ValueError("graph_source must be either 'polygon' or 'cell'")

    # Assign attributes to cellular graph
    G = assign_attributes(G, cell_data, node_to_cell_mapping)
    G.region_id = region_id

    # Build Multiplex Network
    cell_to_node_mapping = {value: key for key, value in node_to_cell_mapping.items()}
    cell_type_graph = nx.Graph()
    cell_type_graph.add_nodes_from(G.nodes(data=True))  # This copies nodes with their features

    grouped = cell_data.groupby('CELL_TYPE')

    for cell_type, group in grouped:
        nodes = group['CELL_ID']
        nodes = [cell_to_node_mapping[cell_id] for cell_id in nodes if cell_id in cell_to_node_mapping]  # Convert cell IDs to node indices
        edges = combinations(nodes, 2)  # Create combinations of nodes
        cell_type_graph.add_edges_from(edges)

    # Assign attributes to cell type graph
    cell_type_graph = assign_attributes(cell_type_graph, cell_data, node_to_cell_mapping, _assert=False)
    cell_type_graph.region_id = region_id

    multiplex_network = {
        'layer_1': G,
        'layer_2': cell_type_graph
    }

    # Visualization of cellular graph
    if voronoi_polygon_img_output is not None:
        plt.clf()
        plt.figure(figsize=(figsize, figsize))
        plot_voronoi_polygons(G)
        plt.axis('scaled')
        plt.savefig(voronoi_polygon_img_output, dpi=300, bbox_inches='tight')

    if graph_img_output is not None:
        plt.clf()
        plt.figure(figsize=(figsize, figsize))
        plot_graph(G)
        plt.axis('scaled')
        plt.savefig(graph_img_output[0], dpi=300, bbox_inches='tight')
        plt.clf()
        plt.figure(figsize=(figsize, figsize))
        plot_graph(cell_type_graph, cell_type=True)
        plt.axis('scaled')
        plt.savefig(graph_img_output[1], dpi=300, bbox_inches='tight')

    # Save graph to file
    if graph_output is not None:
        with open(graph_output, 'wb') as f:
            pickle.dump(multiplex_network, f)

    return multiplex_network


if __name__ == "__main__":
    raw_data_root = "data/voronoi/"
    nx_graph_root = "data/example_dataset/graph"
    fig_save_root = "data/example_dataset/fig"
    os.makedirs(nx_graph_root, exist_ok=True)
    os.makedirs(fig_save_root, exist_ok=True)

    region_ids = sorted(set(f.split('.')[0] for f in os.listdir(raw_data_root)))

    for region_id in region_ids:
        print("Processing %s" % region_id)
        cell_coords_file = os.path.join(raw_data_root, "%s.cell_data.csv" % region_id)
        cell_types_file = os.path.join(raw_data_root, "%s.cell_types.csv" % region_id)
        cell_biomarker_expression_file = os.path.join(raw_data_root, "%s.expression.csv" % region_id)
        cell_features_file = os.path.join(raw_data_root, "%s.cell_features.csv" % region_id)
        voronoi_file = os.path.join(raw_data_root, "%s.json" % region_id)

        voronoi_img_output = os.path.join(fig_save_root, "%s_voronoi.png" % region_id)
        graph_img_output = os.path.join(fig_save_root, "%s_graph.png" % region_id)
        graph_output = os.path.join(nx_graph_root, "%s.gpkl" % region_id)

        if not os.path.exists(graph_output):
            G = construct_graph_for_region(
                region_id,
                cell_coords_file=cell_coords_file,
                cell_types_file=cell_types_file,
                cell_biomarker_expression_file=cell_biomarker_expression_file,
                cell_features_file=cell_features_file,
                voronoi_file=voronoi_file,
                graph_output=graph_output,
                voronoi_polygon_img_output=voronoi_img_output,
                graph_img_output=graph_img_output,
                figsize=10)
