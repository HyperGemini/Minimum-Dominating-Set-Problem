from typing import List, Set, Tuple, Dict, Any, Optional
import numpy as np


def load_graph_and_preprocess(input_path: str) -> Tuple[List[List[int]], int]:
    num_vertices, num_edges = None, None
    edges = []

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

            a, b = map(int, parts)
            edges.append((a, b))

    adjacency_list = [set() for _ in range(num_vertices)]
    for u, v in edges:
        u -= 1
        v -= 1

        if u == v:
            continue

        adjacency_list[u].add(v)
        adjacency_list[v].add(u)

    closed_neighbourhoods = []
    for v in range(num_vertices):
        closed_neighbourhood = set(adjacency_list[v])
        closed_neighbourhood.add(v)
        closed_neighbourhoods.append(sorted(list(closed_neighbourhood)))

    return closed_neighbourhoods, num_vertices


def create_initial_solution(num_nodes: int, closed_neighbourhoods: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
    initial_solution = np.ones(num_nodes, dtype=np.uint8)
    coverage_count = np.zeros(num_nodes, dtype=np.int32)

    for v in range(num_nodes):
        for u in closed_neighbourhoods[v]:
            coverage_count[u] += 1

    return initial_solution, coverage_count


def calculate_solution_value(solution: np.ndarray) -> int:
    return int(solution.sum())


def build_coverers(closed_neighbourhoods: List[List[int]], n: int) -> List[Set[int]]:
    coverers = [set() for _ in range(n)]

    for v, neighbour in enumerate(closed_neighbourhoods):
        for u in neighbour:
            coverers[u].add(v)

    return coverers


def generate_neighbours(current_solution: np.ndarray,
                        coverage_count: np.ndarray,
                        closed_neighbourhoods: List[List[int]],
                        coverers: List[Set[int]],
                        solution_set: Set[int],
                        non_solution_set: Set[int],
                        swap_cap: Optional[int] = 64
                        ) -> List[Tuple[Tuple, int]]:

    neighbours_moves: List[Tuple[Tuple, int]] = []
    current_value = calculate_solution_value(current_solution)

    for node_out in solution_set:
        critical_nodes = [u for u in closed_neighbourhoods[node_out] if coverage_count[u] == 1]

        if not critical_nodes:
            neighbours_moves.append((('removal', node_out), current_value - 1))
            continue

        candidate_nodes: Set[int] = coverers[critical_nodes[0]].copy()
        for node in critical_nodes[1:]:
            candidate_nodes &= coverers[node]

        candidate_nodes &= non_solution_set

        if node_out in candidate_nodes:
            candidate_nodes.remove(node_out)

        if swap_cap is not None and len(candidate_nodes) > swap_cap:
            candidate_nodes = set(list(candidate_nodes)[:swap_cap])

        for node_in in candidate_nodes:
            neighbours_moves.append((('swap', node_out, node_in), current_value))

    return neighbours_moves


def tabu_search(graph_path: str, max_iters: int, tabu_tenure: int):
    closed_neighbourhoods, num_nodes = load_graph_and_preprocess(graph_path)
    coverers = build_coverers(closed_neighbourhoods, num_nodes)
    current_solution, coverage_count = create_initial_solution(num_nodes, closed_neighbourhoods)

    solution_set: Set[int] = set(np.flatnonzero(current_solution))
    non_solution_set: Set[int] = set(range(num_nodes)) - solution_set

    global_best_solution = current_solution.copy()
    global_best_value = calculate_solution_value(global_best_solution)

    tabu_list: Dict[Tuple, int] = {}
    long_term_memory: Dict[Tuple, int] = {}

    for it in range(max_iters):
        expired_moves = [move for move, expiry in tabu_list.items() if expiry <= it]
        for move in expired_moves:
            del tabu_list[move]

        neighbours_moves = generate_neighbours(
            current_solution=current_solution,
            coverage_count=coverage_count,
            closed_neighbourhoods=closed_neighbourhoods,
            coverers=coverers,
            solution_set=solution_set,
            non_solution_set=non_solution_set,
            swap_cap=256
        )

        candidate_moves_with_values: List[Tuple[Tuple, int]] = []

        for move, sol_value in neighbours_moves:
            is_tabu = move in tabu_list

            if is_tabu and sol_value >= global_best_value:
                continue

            candidate_moves_with_values.append((move, sol_value))

        if not candidate_moves_with_values:
            break

        candidate_moves_with_values.sort(key=lambda x: (x[1], long_term_memory.get(x[0], 0)))
        chosen_move, chosen_value = candidate_moves_with_values[0]
        move_type = chosen_move[0]

        if move_type == 'removal':
            node_to_remove = chosen_move[1]
            current_solution[node_to_remove] = 0

            for u in closed_neighbourhoods[node_to_remove]:
                coverage_count[u] -= 1

            solution_set.remove(node_to_remove)
            non_solution_set.add(node_to_remove)

        elif move_type == 'swap':
            node_out, node_in = chosen_move[1], chosen_move[2]

            current_solution[node_out] = 0
            current_solution[node_in] = 1

            for u_out in closed_neighbourhoods[node_out]:
                coverage_count[u_out] -= 1
            for u_in in closed_neighbourhoods[node_in]:
                coverage_count[u_in] += 1

            solution_set.remove(node_out)
            solution_set.add(node_in)
            non_solution_set.add(node_out)
            non_solution_set.remove(node_in)

        if chosen_value < global_best_value:
            global_best_solution = current_solution.copy()
            global_best_value = chosen_value

            print(f"Iteracija {it}: Novo najbolje rešenje = {global_best_value}")

        tabu_list[chosen_move] = it + tabu_tenure
        long_term_memory[chosen_move] = long_term_memory.get(chosen_move, 0) + 1

    final_solution_indices = np.flatnonzero(global_best_solution)
    final_solution_set = set(idx + 1 for idx in final_solution_indices)

    return final_solution_set, global_best_value
