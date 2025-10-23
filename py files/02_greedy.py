from collections.abc import Mapping, Iterable


def calculate_dominating_set(graph: Mapping[int, Iterable[int]]) -> tuple[list[int], int]:
    graph_nodes = set(graph.keys())
    closed_neighbourhood = {node: set(neighbour) | {node} for node, neighbour in graph.items()}

    uncovered_nodes = set(closed_neighbourhood)
    dominating_set = set()

    while uncovered_nodes:
        best_node = None
        best_gain = -1

        for node in graph_nodes:
            current_gain = len(closed_neighbourhood[node] & uncovered_nodes)
            if current_gain > best_gain or (current_gain == best_gain and
                                            best_node is not None and
                                            len(closed_neighbourhood[node]) > len(closed_neighbourhood[best_node])):
                best_gain = current_gain
                best_node = node

        dominating_set.add(best_node)
        uncovered_nodes -= closed_neighbourhood[best_node]

    return list(dominating_set), len(dominating_set)
