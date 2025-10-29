import random
import heapq
from copy import deepcopy
from typing import List, Set, Tuple

import numpy as np


def load_graph(input_path: str) -> List[Set[int]]:
    num_vertices, num_edges = None, None
    edges = []

    with open(input_path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            line = line.strip()

            if not line or line[0] in ("c", "C"):
                continue

            if line.startswith("p "):
                parts = line.split()
                num_vertices = int(parts[2])
                continue

            a, b = map(int, line.split())
            edges.append((a, b))

    assert num_vertices is not None, "Did not find 'p ds n m' line in input"

    adjacency_list = [set() for _ in range(num_vertices)]
    for u, v in edges:
        u -= 1
        v -= 1

        if u == v:
            continue

        adjacency_list[u].add(v)
        adjacency_list[v].add(u)

    return adjacency_list


def build_closed_neighbourhoods(adjacency_list: List[Set[int]]) -> Tuple[List[List[int]], List[int]]:
    num_nodes = len(adjacency_list)

    closed_neighbourhoods = []
    closed_masks = []

    for v in range(num_nodes):
        closed_neighbourhood = set(adjacency_list[v])
        closed_neighbourhood.add(v)

        sorted_neighbourhood = sorted(closed_neighbourhood)
        closed_neighbourhoods.append(sorted_neighbourhood)

        mask = 0
        for u in sorted_neighbourhood:
            mask |= (1 << u)
        closed_masks.append(mask)

    return closed_neighbourhoods, closed_masks


class Individual:
    __slots__ = ("closed_neighbourhoods", "closed_masks", "genetic_code",
                 "num_nodes", "fitness", "solution_size", "num_uncovered_nodes")

    def __init__(self,
                 closed_neighbourhoods: List[List[int]],
                 closed_masks: List[int],
                 genetic_code: np.ndarray | None = None,
                 compute_fitness: bool = True):

        self.num_nodes = len(closed_neighbourhoods)
        self.closed_neighbourhoods = closed_neighbourhoods
        self.closed_masks = closed_masks

        if genetic_code is None:
            self.genetic_code = (np.random.rand(self.num_nodes) < 0.05).astype(np.uint8)
        else:
            self.genetic_code = genetic_code.astype(np.uint8, copy=True)

        if compute_fitness:
            self.recompute_fitness()
        else:
            self.fitness = 0
            self.solution_size = int(self.genetic_code.sum())
            self.num_uncovered_nodes = self.num_nodes

    def calculate_mask_of_covered_nodes(self) -> int:
        mask_covered = 0
        for u in np.flatnonzero(self.genetic_code):
            mask_covered |= self.closed_masks[u]
        return mask_covered

    def calculate_fitness(self) -> Tuple[int, int, int]:
        mask_covered = self.calculate_mask_of_covered_nodes()
        num_covered_nodes = mask_covered.bit_count()
        num_uncovered_nodes = self.num_nodes - num_covered_nodes

        solution_size = int(self.genetic_code.sum())

        coverage_weight = self.num_nodes + 1
        fitness = num_covered_nodes * coverage_weight - solution_size

        return fitness, solution_size, num_uncovered_nodes

    def recompute_fitness(self):
        self.fitness, self.solution_size, self.num_uncovered_nodes = self.calculate_fitness()

    def repair(self):
        mask_all_nodes = (1 << self.num_nodes) - 1
        mask_covered_nodes = self.calculate_mask_of_covered_nodes()
        mask_uncovered_nodes = mask_all_nodes ^ mask_covered_nodes

        while mask_uncovered_nodes:
            first_uncovered_node = (mask_uncovered_nodes & -mask_uncovered_nodes).bit_length() - 1

            best_node = None
            best_gain = -1

            for v in self.closed_neighbourhoods[first_uncovered_node]:
                current_gain = (self.closed_masks[v] & mask_uncovered_nodes).bit_count()
                if current_gain > best_gain:
                    best_gain = current_gain
                    best_node = v

            self.genetic_code[best_node] = 1
            mask_covered_nodes |= self.closed_masks[best_node]
            mask_uncovered_nodes = mask_all_nodes ^ mask_covered_nodes

        coverage_count = [0] * self.num_nodes
        for v in np.flatnonzero(self.genetic_code):
            for u in self.closed_neighbourhoods[v]:
                coverage_count[u] += 1

        covered_nodes = list(np.flatnonzero(self.genetic_code))
        random.shuffle(covered_nodes)

        for v in covered_nodes:
            can_drop = True
            for u in self.closed_neighbourhoods[v]:
                if coverage_count[u] == 1:
                    can_drop = False
                    break
            if can_drop:
                self.genetic_code[v] = 0
                for u in self.closed_neighbourhoods[v]:
                    coverage_count[u] -= 1

        self.recompute_fitness()


def initialize_population(population_size: int, closed_neighbourhoods: List[List[int]], closed_masks: List[int]) -> \
        List[Individual]:
    population = []
    for _ in range(population_size):
        individual = Individual(closed_neighbourhoods, closed_masks)
        individual.repair()
        population.append(individual)
    return population


def tournament_selection(population: List[Individual], tournament_size: int) -> Individual:
    contestants = random.sample(population, k=tournament_size)
    return max(contestants, key=lambda individual: individual.fitness)


def crossover(parent1: Individual, parent2: Individual, p: float = 0.5):
    mask = (np.random.rand(parent1.num_nodes) < p)
    c1 = np.where(mask, parent1.genetic_code, parent2.genetic_code).astype(np.uint8)
    c2 = np.where(mask, parent2.genetic_code, parent1.genetic_code).astype(np.uint8)

    child1 = Individual(parent1.closed_neighbourhoods, parent1.closed_masks, c1, compute_fitness=False)
    child2 = Individual(parent1.closed_neighbourhoods, parent1.closed_masks, c2, compute_fitness=False)

    return child1, child2


def mutate(individual: Individual, mutation_rate: float) -> bool:
    if random.random() < mutation_rate:
        index = random.randrange(individual.num_nodes)
        individual.genetic_code[index] ^= 1

        mask_covered_nodes = individual.calculate_mask_of_covered_nodes()
        if mask_covered_nodes.bit_count() < individual.num_nodes:
            individual.repair()

        individual.recompute_fitness()
        return True
    return False


def ga(closed_neighbourhoods: List[List[int]],
       closed_masks: List[int],
       population_size: int = 300,
       tournament_size: int = 5,
       mutation_rate: float = 0.03,
       elitism_size: int = 12,
       max_iters: int = 1500,
       p_repair_start: float = 0.30,
       p_repair_end: float = 1.00
       ) -> Individual:
    population = initialize_population(population_size, closed_neighbourhoods, closed_masks)
    best_solution_ever = deepcopy(max(population, key=lambda ind: ind.fitness))

    for it in range(max_iters):
        elites = heapq.nlargest(elitism_size, population, key=lambda ind: ind.fitness)
        if elites[0].fitness > best_solution_ever.fitness:
            best_solution_ever = elites[0]

        new_population = elites

        p_repair = p_repair_start + (p_repair_end - p_repair_start) * (it / max_iters)

        while len(new_population) < population_size:
            parent1 = tournament_selection(population, tournament_size)
            parent2 = tournament_selection(population, tournament_size)

            while parent2 is parent1 and len(population) > 1:
                parent2 = tournament_selection(population, tournament_size)

            child1, child2 = crossover(parent1, parent2)

            did_repair1 = False
            did_repair2 = False

            if random.random() < p_repair:
                child1.repair()
                did_repair1 = True
            if random.random() < p_repair:
                child2.repair()
                did_repair2 = True

            changed = mutate(child1, mutation_rate)
            if not changed and not did_repair1:
                child1.recompute_fitness()

            changed = mutate(child2, mutation_rate)
            if not changed and not did_repair2:
                child2.recompute_fitness()

            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)

        population = new_population

    if best_solution_ever.num_uncovered_nodes > 0:
        best_solution_ever.repair()

    return best_solution_ever
