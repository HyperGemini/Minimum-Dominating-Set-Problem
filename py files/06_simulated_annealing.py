import random
import math
from typing import Dict, Set, Tuple

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


def calculate_cost(solution_nodes: Set[int],
                   graph: Dict[int, Set[int]],
                   closed_neighbourhoods: Dict[int, Set[int]],
                   penalty_weight: int
                   ) -> int:

    num_nodes = len(graph)
    if num_nodes == 0:
        return 0

    covered_nodes = bytearray(num_nodes)
    num_uncovered_nodes = num_nodes

    for v in solution_nodes:
        for u in closed_neighbourhoods[v]:
            if not covered_nodes[u]:
                covered_nodes[u] = 1
                num_uncovered_nodes -= 1

    cost = len(solution_nodes) + (num_uncovered_nodes * penalty_weight)
    return cost


def create_initial_solution(graph: Dict[int, Set[int]], init_method="greedy", closed_neighbourhoods=None, top_k=3):
    if init_method == "greedy":
        if closed_neighbourhoods is None:
            closed_neighbourhoods = precompute_closed_neighbourhoods(graph)

        uncovered = set(graph.keys())
        solution = set()
        while uncovered:
            sorted_nodes = sorted(graph.keys(), key=lambda x: len(closed_neighbourhoods[x] & uncovered), reverse=True)
            choice_pool = sorted_nodes[:top_k]
            chosen = random.choice(choice_pool)
            solution.add(chosen)
            uncovered -= closed_neighbourhoods[chosen]
        return solution

    return set(graph.keys())


def repair_solution(solution: Set[int],
                    graph: Dict[int, Set[int]],
                    closed_neighbourhoods: Dict[int, Set[int]]
                    ) -> Set[int]:

    uncovered = set(graph.keys())
    for v in solution:
        uncovered -= closed_neighbourhoods[v]

    while uncovered:
        best_node = max(graph.keys(), key=lambda x: len(closed_neighbourhoods[x] & uncovered))
        solution.add(best_node)
        uncovered -= closed_neighbourhoods[best_node]

    return solution

def make_small_change(solution: Set[int], num_nodes: int, T: float, T0: float) -> Set[int]:
    candidate = solution.copy()
    move_type = random.choices(["multi_flip", "swap"], weights=[0.9, 0.1], k=1)[0]

    if move_type == "multi_flip":
        if num_nodes <= 200:
            num_flips = 1
        else:
            base_flips = math.ceil(num_nodes * 0.005)
            cooling_factor = max(0.1, T / T0)
            num_flips = max(1, int(base_flips * cooling_factor))

        for _ in range(num_flips):
            node = random.randrange(num_nodes)
            if node in candidate:
                candidate.remove(node)
            else:
                candidate.add(node)

    else:
        if candidate:
            node_to_remove = random.choice(tuple(candidate))

            outside_nodes = set(range(num_nodes)) - candidate
            if not outside_nodes:
                return candidate

            node_to_add = random.choice(tuple(outside_nodes))
            candidate.remove(node_to_remove)
            candidate.add(node_to_add)

    return candidate


def simulated_annealing(graph: Dict[int, Set[int]],
                        max_iters: int,
                        init_method: str = "greedy",
                        T0: float = 10.0,
                        alpha: float = 0.999,
                        T_min: float = 1e-6
                        ) -> Tuple[Set[int], int]:

    num_nodes = len(graph)
    if num_nodes == 0:
        return set(), 0

    closed_neighbourhoods = precompute_closed_neighbourhoods(graph)
    PENALTY = num_nodes + 1

    solution_nodes = create_initial_solution(graph,
                                             init_method=init_method,
                                             closed_neighbourhoods=closed_neighbourhoods)

    current_cost = calculate_cost(solution_nodes, graph, closed_neighbourhoods, PENALTY)
    best_solution_nodes = solution_nodes.copy()
    best_cost = current_cost

    T = T0

    for it in range(1, max_iters + 1):
        candidate = make_small_change(solution_nodes, num_nodes, T, T0)
        new_cost = calculate_cost(candidate, graph, closed_neighbourhoods, PENALTY)
        delta = new_cost - current_cost

        if delta <= 0 or random.random() < math.exp(-delta / T):
            solution_nodes = candidate
            current_cost = new_cost

            if new_cost < best_cost:
                best_cost = new_cost
                best_solution_nodes = candidate

        T = max(T * alpha, T_min)
        if T == T_min:
            break

    if not is_dominating_set(best_solution_nodes, graph, closed_neighbourhoods):
        print("\nFixing invalid solution...")
        best_solution_nodes = repair_solution(best_solution_nodes, graph, closed_neighbourhoods)

    return best_solution_nodes, len(best_solution_nodes)







