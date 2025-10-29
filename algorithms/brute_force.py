import itertools
from collections.abc import Mapping, Iterable


def load_graph(input_path: str) -> dict[int, list[int]]:
    num_vertices: int | None = None
    edges: list[tuple[int, int]] = []

    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line[0] in ("c", "C"):
                continue

            if line.startswith("p "):
                parts = line.split()

                if len(parts) >= 3 and parts[1].lower() == "ds":
                    num_vertices = int(parts[2])
                else:
                    for tok in parts[1:]:
                        if tok.isdigit():
                            num_vertices = int(tok)
                            break
                continue

            if num_vertices is None:
                continue

            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = map(int, parts[:2])
            edges.append((u, v))

    if num_vertices is None:
        raise ValueError("p ds <n> [<m>]")

    graph: dict[int, list[int]] = {i: [] for i in range(1, num_vertices + 1)}
    for u, v in edges:
        if u == v:
            continue
        if v not in graph[u]:
            graph[u].append(v)
        if u not in graph[v]:
            graph[v].append(u)

    return graph


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
