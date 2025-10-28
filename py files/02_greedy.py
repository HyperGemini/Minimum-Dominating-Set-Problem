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
