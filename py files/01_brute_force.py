import itertools
from collections.abc import Mapping, Iterable


def is_dominating_set(graph: Mapping[int, Iterable[int]], candidate_nodes: set[int]) -> bool:
    for node, neighbours in graph.items():
        if node in candidate_nodes:
            continue
        if not any(neighbour in candidate_nodes for neighbour in neighbours):
            return False
    return True


def calculate_dominating_set(graph: Mapping[int, Iterable[int]]) -> tuple[list[int], int]:
    graph_nodes = list(graph.keys())
    num_nodes = len(graph_nodes)

    for current_set_size in range(1, num_nodes + 1):
        for candidate_dominating_set in itertools.combinations(graph_nodes, current_set_size):
            if is_dominating_set(graph, set(candidate_dominating_set)):
                return list(candidate_dominating_set), current_set_size
    return [], 0
