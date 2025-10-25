import random
from typing import Dict, Set, Tuple, Optional


def load_graph(input_path: str) -> Dict[int, Set[int]]:
    num_vertices = None
    edges: list[Tuple[int, int]] = []

    with open(input_path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            line = line.strip()

            if not line or line[0] in ("c", "C"):
                continue

            if line.startswith("p "):
                parts = line.split()
                if parts[1] == 'ds':
                    num_vertices = int(parts[2])
                continue

            if num_vertices is None:
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            a, b = map(int, parts[:2])
            edges.append((a, b))

    adjacency_list: Dict[int, Set[int]] = {i: set() for i in range(num_vertices)}
    for u, v in edges:
        u -= 1
        v -= 1

        if u == v:
            continue

        adjacency_list[u].add(v)
        adjacency_list[v].add(u)

    return adjacency_list


def calculate_solution_value(solution_nodes: Set[int]) -> int:
    return len(solution_nodes)


def precompute_closed_neighbourhoods(graph: Dict[int, Set[int]]) -> Dict[int, Set[int]]:
    closed_neighbourhoods: Dict[int, Set[int]] = {}
    for v, neighbours in graph.items():
        s = set(neighbours)
        s.add(v)
        closed_neighbourhoods[v] = s
    return closed_neighbourhoods


def is_dominating_set(solution_nodes: Set[int],
                      graph: Dict[int, Set[int]],
                      closed_neighbourhoods: Dict[int, Set[int]]
                      ) -> bool:

    num_nodes = len(graph)
    if num_nodes == 0:
        return True
    if not solution_nodes:
        return False

    covered_nodes = bytearray(num_nodes)
    num_uncovered_nodes = num_nodes

    for v in solution_nodes:
        for u in closed_neighbourhoods[v]:
            if not covered_nodes[u]:
                covered_nodes[u] = 1
                num_uncovered_nodes -= 1

                if num_uncovered_nodes == 0:
                    return True

    return num_uncovered_nodes == 0


def create_initial_solution(graph: Dict[int, Set[int]],
                            init_method: str = "all",
                            closed_neighbourhoods: Optional[Dict[int, Set[int]]] = None
                            ) -> Set[int]:

    if init_method == "greedy":
        if closed_neighbourhoods is None:
            closed_neighbourhoods = precompute_closed_neighbourhoods(graph)

        uncovered_nodes = set(graph.keys())
        solution_nodes: Set[int] = set()

        while uncovered_nodes:
            v = max(graph.keys(), key=lambda x: len(closed_neighbourhoods[x] & uncovered_nodes))
            solution_nodes.add(v)
            uncovered_nodes -= closed_neighbourhoods[v]
        return solution_nodes

    return set(graph.keys())


def make_small_change(solution_nodes: Set[int],
                      graph: Dict[int, Set[int]],
                      all_nodes: Optional[Set[int]] = None,
                      closed: Optional[Dict[int, Set[int]]] = None
                      ) -> Set[int]:

    if all_nodes is None:
        all_nodes = set(graph.keys())

    if not solution_nodes:
        return solution_nodes

    move_type = "removal" if random.random() < 0.5 else "swap"

    if move_type == "removal":
        candidate_solution = solution_nodes.copy()
        node_to_remove = random.choice(tuple(candidate_solution))
        candidate_solution.remove(node_to_remove)
        return candidate_solution if is_dominating_set(candidate_solution, graph, closed) else solution_nodes

    # swap
    candidate_solution = solution_nodes.copy()
    node_to_remove = random.choice(tuple(candidate_solution))
    candidate_solution.remove(node_to_remove)

    candidates_for_addition = all_nodes - candidate_solution
    if not candidates_for_addition:
        return solution_nodes

    node_to_add = random.choice(tuple(candidates_for_addition))
    candidate_solution.add(node_to_add)

    return candidate_solution if is_dominating_set(candidate_solution, graph, closed) else solution_nodes


def local_search(graph: Dict[int, Set[int]],
                 max_iters: int,
                 init_method: str = "all"
                 ) -> Tuple[Set[int], int]:

    closed_neighbourhoods = precompute_closed_neighbourhoods(graph)
    all_nodes = set(graph.keys())

    solution_nodes = create_initial_solution(graph,
                                             init_method=init_method,
                                             closed_neighbourhoods=closed_neighbourhoods)

    best_solution_nodes = solution_nodes.copy()
    best_solution_size = calculate_solution_value(best_solution_nodes)

    for _ in range(max_iters):
        new_solution_nodes = make_small_change(solution_nodes, graph, all_nodes, closed_neighbourhoods)
        new_solution_size = len(new_solution_nodes)

        if new_solution_size < best_solution_size:
            best_solution_size = new_solution_size
            best_solution_nodes = new_solution_nodes

        solution_nodes = new_solution_nodes

    return best_solution_nodes, best_solution_size