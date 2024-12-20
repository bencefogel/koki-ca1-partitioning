import networkx as nx
import pandas as pd
from pandas import DataFrame


def create_directed_graph(df_iax: DataFrame, tp: int) -> nx.DiGraph:
    df_iax_tp = df_iax[tp]
    df_iax_tp = df_iax_tp.reset_index()
    df_iax_tp.rename(columns={tp: "iax"}, inplace=True)  # has three columns: ref, par, iax

    # Create directed graph (add edges to the graph based on the sign of iax)
    dg = nx.DiGraph()
    for index, row in df_iax_tp.iterrows():
        if row['iax'] >= 0:
            dg.add_edge(row['par'], row['ref'], iax=row['iax'])  # par -> ref if 'iax_timepoint' is positive
        elif row['iax'] < 0:
            dg.add_edge(row['ref'], row['par'], iax=row['iax'])  # ref -> par if 'iax_timepoint' is negative
    return dg


def get_partitioning_order(dg: nx.DiGraph, target: str, direction: str) -> list[tuple[str, str]]:
    traversal_methods = {
        "out": get_traversal_order_out,
        "in": get_traversal_order_in
    }
    return traversal_methods[direction](dg, target)


def get_traversal_order_out(dg: nx.DiGraph, target: str) -> list[tuple[str, str]]:
    # Find subgraph using depth first search algorithm and copy iax values of edges
    dg_dfs_out = nx.dfs_tree(dg, source=target)
    for u, v in dg_dfs_out.edges():
        if dg.has_edge(u, v):
            dg_dfs_out[u][v]['iax'] = dg[u][v]['iax']

    # Extract traversal order starting from the leaf nodes
    edges_visited_out = list(nx.edge_dfs(dg_dfs_out, target))
    node_pairs_out = [(v, u) for (u, v) in edges_visited_out]  # switch nodes of each edge
    node_pairs_out.reverse()  # reverse node pairs order (to start from the leaf nodes)
    return node_pairs_out


def get_traversal_order_in(dg: nx.DiGraph, target: str) -> list[tuple[str, str]]:
    dg_reversed = nx.reverse(dg, copy=True)
    # Find subgraph using depth first search algorithm and copy iax values of edges
    dg_dfs_in = nx.dfs_tree(dg_reversed, source=target)
    for u, v in dg_dfs_in.edges():
        if dg.has_edge(v, u):
            dg_dfs_in[u][v]['iax'] = dg[v][u]['iax']

    # Extract traversal order starting from the leaf nodes
    edges_visited_in = list(nx.edge_dfs(dg_dfs_in, target))
    node_pairs_in = [(v, u) for (u, v) in edges_visited_in]  # switch nodes of each edge
    node_pairs_in.reverse()  # reverse node pairs order (to start from the leaf nodes)
    return node_pairs_in