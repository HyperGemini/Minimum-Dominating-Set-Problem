import random
from copy import deepcopy
from typing import List, Set, Tuple
from bitarray import bitarray
import numpy as np


def load_graph(input_path: str) -> List[Set[int]]:
    num_vertices, num_edges = None, None
    edges: List[Tuple[int, int]] = []

    with open(input_path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('c'):
                continue
            if line.startswith('p '):
                _, _, n, m = line.split()
                num_vertices, num_edges = int(n), int(m)
                continue

            parts = line.split()
            a, b = map(int, parts)
            edges.append((a, b))

    assert num_vertices is not None, "Invalid file: no 'p' line found"

    graph: List[Set[int]] = [set() for _ in range(num_vertices)]
    for u, v in edges:
        u -= 1
        v -= 1
        graph[u].add(v)
        graph[v].add(u)

    return graph


def build_bit_neighborhoods(graph: list[set[int]]) -> list[bitarray]:
    num_nodes = len(graph)
    neighborhoods = []

    for i in range(num_nodes):
        b = bitarray(num_nodes)
        b.setall(0)
        b[i] = 1
        for v in graph[i]:
            b[v] = 1
        neighborhoods.append(b)

    return neighborhoods


def get_node_degrees(graph: list[set[int]]) -> np.ndarray:
    return np.array([len(neighbors) for neighbors in graph], dtype=float)


class Individual:
    def __init__(self, genetic_code: np.ndarray, neighborhoods: list[bitarray]):
        self.genetic_code = genetic_code.astype(np.uint8)
        self.neighborhoods = neighborhoods
        self.fitness = self.calculate_fitness()

    def calculate_fitness(self) -> float:
        num_nodes = len(self.neighborhoods)
        selected = np.flatnonzero(self.genetic_code)
        if selected.size == 0:
            return 0.0

        covered = bitarray(num_nodes)
        covered.setall(0)

        for node in selected:
            covered |= self.neighborhoods[node]

        n = covered.count(1)
        gamma_x = selected.size

        return n / num_nodes + 1 / (num_nodes * gamma_x)


def initialize_population(pop_size: int, num_nodes: int, neighborhoods: list[bitarray]) -> List[Individual]:
    population = []
    for _ in range(pop_size):
        code = np.random.randint(0, 2, size=num_nodes)
        individual = Individual(code, neighborhoods)
        individual.fitness = individual.calculate_fitness()
        population.append(individual)
    return population


def build_mating_pool_linear_ranking(population: list[Individual],
                                     max_rank_value: float = 1.1,
                                     offspring_count: int | None = None) -> list[Individual]:

    N = len(population)
    if offspring_count is None:
        offspring_count = N
    if N == 1:
        return population[:]

    ranked = sorted(population, key=lambda ind: ind.fitness, reverse=True)

    MAX = float(max_rank_value)
    INC = 2.0 * (MAX - 1.0)

    expected = [MAX - r * INC for r in range(N)]

    total = sum(expected)
    probs = [e / total for e in expected]

    mating_pool = random.choices(ranked, weights=probs, k=offspring_count)

    return mating_pool


def crossover(parent1: Individual, parent2: Individual) -> tuple[Individual, Individual]:
    n = len(parent1.genetic_code)
    crossover_point = random.randrange(1, n)

    child1_code = np.concatenate((parent1.genetic_code[:crossover_point], parent2.genetic_code[crossover_point:]))
    child2_code = np.concatenate((parent2.genetic_code[:crossover_point], parent1.genetic_code[crossover_point:]))

    return (Individual(child1_code, parent1.neighborhoods),
            Individual(child2_code, parent1.neighborhoods))


def mutate(individual: Individual, mutation_rate: float):
    random_mask = np.random.rand(len(individual.genetic_code)) < mutation_rate
    individual.genetic_code[random_mask] = 1 - individual.genetic_code[random_mask]
    individual.fitness = individual.calculate_fitness()


def local_search(individual: Individual,
                 node_degrees: np.ndarray,
                 num_iters: int) -> Individual:

    best_individual = deepcopy(individual)

    for _ in range(num_iters):
        current = deepcopy(best_individual)
        current_fit = current.fitness

        if current_fit >= 1:
            ones = np.flatnonzero(current.genetic_code)
            if ones.size == 0:
                continue

            weights = 1.0 / node_degrees[ones].astype(float)
            weights /= weights.sum()
            chosen = np.random.choice(ones, p=weights)
            current.genetic_code[chosen] = 0

        else:
            zeros = np.flatnonzero(1 - current.genetic_code)
            if zeros.size == 0:
                continue

            weights = node_degrees[zeros]
            weights /= weights.sum()
            chosen = np.random.choice(zeros, p=weights)
            current.genetic_code[chosen] = 1

        current.fitness = current.calculate_fitness()

        if current.fitness > current_fit:
            best_individual = deepcopy(current)

    return best_individual


def filtering(individual: Individual) -> Individual:
    if individual.fitness < 1.0:
        return deepcopy(individual)

    improved = True
    best_individual = deepcopy(individual)

    while improved:
        improved = False
        ones = np.flatnonzero(best_individual.genetic_code)

        for node in list(ones):
            best_individual.genetic_code[node] = 0
            new_fitness = best_individual.calculate_fitness()

            if new_fitness > best_individual.fitness:
                best_individual.fitness = new_fitness
                improved = True
            else:
                best_individual.genetic_code[node] = 1

    return best_individual


def elite_inspiration(elite_sets: List[Individual],
                      node_degrees: np.ndarray,
                      n_core: int
                      ) -> Individual | None:

    if not elite_sets or len(elite_sets) < n_core:
        return None

    elites_sorted = sorted(elite_sets, key=lambda e: e.fitness, reverse=True)
    x_best = elites_sorted[0]
    n_F = int(np.sum(x_best.genetic_code))

    x_core = np.logical_and.reduce([e.genetic_code for e in elites_sorted[:n_core]]).astype(np.uint8)
    neighborhoods = elite_sets[0].neighborhoods
    x_core_ind = Individual(x_core, neighborhoods)

    while True:
        current_size = int(np.sum(x_core_ind.genetic_code))

        if current_size >= n_F - 1:
            break

        if x_core_ind.fitness >= 1:
            return x_core_ind

        zero_indices = np.where(x_core_ind.genetic_code == 0)[0]
        if zero_indices.size == 0:
            break

        best_idx = zero_indices[np.argmax(node_degrees[zero_indices])]
        x_core_ind.genetic_code[best_idx] = 1
        x_core_ind.fitness = x_core_ind.calculate_fitness()

    return x_core_ind if x_core_ind.fitness >= 1 else None


def hga_mds(
        graph: list[set[int]],
        neighborhoods: list[bitarray],
        node_degrees: np.ndarray,
        M: int = 40,
        g_max: int = 100,
        p_c: float = 0.8,
        p_m: float = 0.01,
        n_l: int = 2,
        n_DS: int = 10,
        n_core: int = 3,
        max_rank_value: float = 1.1
) -> Individual:

    num_nodes = len(graph)

    DS: List[Individual] = []
    P_t = initialize_population(M, num_nodes, neighborhoods)

    best_in_P0_idx = max(range(len(P_t)), key=lambda i: P_t[i].fitness)
    global_best_individual = deepcopy(P_t[best_in_P0_idx])
    improved_best = local_search(global_best_individual, node_degrees, n_l)

    P_t[best_in_P0_idx] = improved_best
    global_best_individual = improved_best

    for t in range(g_max):
        best_in_P_t = deepcopy(max(P_t, key=lambda ind: ind.fitness))
        P_t_prime = build_mating_pool_linear_ranking(P_t, max_rank_value, M)

        offspring = []
        random.shuffle(P_t_prime)

        i = 0
        while len(offspring) < M:
            parent1 = P_t_prime[i]
            parent2 = P_t_prime[i + 1]

            if random.random() < p_c:
                child1, child2 = crossover(parent1, parent2)
                offspring.append(child1)
                if len(offspring) < M:
                    offspring.append(child2)
            else:
                offspring.append(deepcopy(parent1))
                if len(offspring) < M:
                    offspring.append(deepcopy(parent2))
            i += 2

        for child in offspring:
            mutate(child, p_m)

        P_t_plus_1 = offspring
        best_in_P_t_plus_1 = max(P_t_plus_1, key=lambda ind: ind.fitness)

        if best_in_P_t_plus_1.fitness < best_in_P_t.fitness:
            worst_index = min(range(len(P_t_plus_1)), key=lambda i: P_t_plus_1[i].fitness)
            P_t_plus_1[worst_index] = best_in_P_t

        P_t = P_t_plus_1

        current_best_in_pop = max(P_t, key=lambda ind: ind.fitness)
        if current_best_in_pop.fitness > global_best_individual.fitness:
            global_best_individual = deepcopy(current_best_in_pop)

        improved_best = local_search(global_best_individual, node_degrees, n_l)
        if improved_best.fitness > global_best_individual.fitness:
            global_best_individual = improved_best

        if global_best_individual.fitness >= 1.0:
            DS.append(deepcopy(global_best_individual))
            DS.sort(key=lambda ind: ind.fitness, reverse=True)
            DS = DS[:n_DS]

        if global_best_individual.fitness >= 1.0:
            filtered_best = filtering(global_best_individual)
            if filtered_best.fitness > global_best_individual.fitness:
                global_best_individual = filtered_best

            if global_best_individual.fitness >= 1.0:
                DS.append(deepcopy(global_best_individual))
                DS.sort(key=lambda ind: ind.fitness, reverse=True)
                DS = DS[:n_DS]

    x_core_ind = elite_inspiration(DS, node_degrees, n_core)

    final_best = global_best_individual

    if x_core_ind is not None:
        if x_core_ind.fitness > final_best.fitness:
            final_best = x_core_ind

    return final_best
