# Minimal Dominating Set

**Kratak opis**

Ovaj repozitorijum sadrži implementacije više pristupa za rešavanje problema **Minimal Dominating Set (MDS)**. Cilj projekta je da se istraže kvalitativne i vremenske performanse različitih heurističkih i egzaktnih pristupa (brute force, greedy, local search, simulated annealing, tabu search, genetic algorithm i hibridni genetski algoritam - HGA). U projektu su takođe priloženi test primeri grafova i izveštaj sa eksperimentima.

---

## Requirements

- **Python 3.13** (projekt je rađen i testiran na verziji 3.13)
- Eksterne biblioteke:
  - `numpy`
  - `bitarray`

`requirements.txt`:

```
numpy
bitarray
```

---

## Instalacija

1. Napravi virtualno okruženje (preporučeno):

```bash
python3.13 -m venv .venv
source .venv/bin/activate    # Linux / macOS
.venv\Scripts\activate      # Windows (PowerShell)
```

2. Instaliraj zavisnosti:

```bash
pip install -r requirements.txt
```

---

## Kako pokrenuti — primeri upotrebe

Svaki modul u `algorithms/` sadrži funkciju `load_graph(...)` (ili sličnu) za čitanje `.gr` fajla i glavnu funkciju koja pokreće algoritam. Primeri su generički - pogledaj potpise funkcija u početku svakog fajla za tačne parametre.

### Simulated Annealing (primer)

```python
from algorithms.simulated_annealing import load_graph, simulated_annealing

graph = load_graph('tests/bremen_subgraph_20.gr')
solution, size = simulated_annealing(graph, max_iters=10000, init_method='greedy')
print('Velicina MDS:', size)
print('Cvorovi:', sorted(solution))
```

### Tabu Search (primer)

```python
from algorithms.tabu_search import load_graph_and_preprocess, tabu_search

graph = load_graph_and_preprocess('tests/test.gr')
solution, size = tabu_search(graph, max_iters=5000, tabu_tenure=7)
print('Velicina MDS (Tabu):', size)
```