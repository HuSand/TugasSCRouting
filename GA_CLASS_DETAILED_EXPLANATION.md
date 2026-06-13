# Penjelasan Lengkap GeneticAlgorithm Class

## 1. Class Definition & Purpose

```python
class GeneticAlgorithm(BaseRoutingAlgorithm):
    """
    Genetic Algorithm for routing — balanced time + distance fitness.
    """
```

**Apa itu GA class?**
- Subclass dari `BaseRoutingAlgorithm` (parent class untuk semua algorithms)
- Implement dua mode routing: point-to-point dan multi-stop TSP
- Menggunakan genetic operators (selection, crossover, mutation) untuk evolve solutions

**Dua mode:**
1. **Point-to-point** (`find_route`): Cari jalan terbaik dari A ke B
   - Fitness = 50% travel_time + 50% physical distance
   - Chromosome = actual road path
   
2. **Multi-stop TSP** (`_route_multi_stop`): Cari urutan kunjungan optimal untuk multiple stops
   - Fitness = total travel_time
   - Chromosome = permutation (visit order)

---

## 2. Class Attributes & Hyperparameters

### Point-to-Point Hyperparameters

```python
name        = "ga"
description = "Genetic Algorithm — balanced time+distance (50/50), TSP-aware multi-stop"

# Point-to-point hyperparameters
POPULATION_SIZE = 30       # 30 individu per generasi
GENERATIONS     = 600      # Max 600 generasi untuk convergence
CROSSOVER_RATE  = 0.85     # 85% chance ada crossover
MUTATION_RATE   = 0.9      # 90% chance mutation terjadi
TOURNAMENT_SIZE = 3        # Tournament selection pilih 3 individu
RANDOM_SEED     = 42       # Seed untuk reproducibility
```

**Penjelasan hyperparameters:**
```
POPULATION_SIZE = 30:
  - Setiap generation = 30 paths berbeda
  - Lebih besar = explore lebih, tapi slower
  - 30 = balance antara quality dan speed

GENERATIONS = 600:
  - Maksimal 600 generasi untuk evolve
  - Bisa stop lebih awal jika converge

CROSSOVER_RATE = 0.85:
  - 85% offspring hasil mixing 2 parents
  - 15% offspring = copy parent (mutation only)

MUTATION_RATE = 0.9:
  - 90% offspring dimutasi
  - 10% offspring copy tanpa change

TOURNAMENT_SIZE = 3:
  - Selection: pilih 3 random, ambil yang terbaik
  - 3 = good balance untuk diversity

RANDOM_SEED = 42:
  - Fixed seed → hasil reproducible
  - Good untuk testing & debugging
```

### Multi-Stop TSP Hyperparameters

```python
TSP_POPULATION_SIZE = 50    # 50 individu per generation
TSP_GENERATIONS     = 600   # Max 600 generasi
TSP_PATIENCE        = 20    # Early stop jika 20 gen no improvement
TSP_WORKERS         = 5     # 5 thread untuk parallel Dijkstra
```

**Kenapa separate dari point-to-point?**
```
Reason:
- Point-to-point: search space = semua possible paths (BESAR)
- TSP: search space = semua possible permutations (KECIL)
- Contoh: 30 nodes = 30! permutations
  vs point-to-point = exponential paths

Jadi TSP bisa:
- Gunakan population lebih besar (50 vs 30)
- Generations sama tapi lebih efisien
- Early stopping lebih aggressive (patience=20)
```

---

## 3. _fitness() Method - Bagaimana Menilai Kualitas Path

```python
def _fitness(self, G, path: list) -> float:
    """
    Balanced fitness untuk point-to-point path.
    
    Combines:
    - 50% travel_time (seberapa cepat)
    - 50% physical distance (seberapa pendek)
    """
    REF_SPEED_MS = 40 / 3.6   # Reference: 40 km/h = 11.1 m/s
    
    time_s = 0.0
    dist_m = 0.0
    
    # Traverse setiap edge di path
    for u, v in zip(path[:-1], path[1:]):
        # Get edge data dari graph
        data = G.get_edge_data(u, v)
        if data is None:
            return float("inf")  # Invalid path
        
        # Find best edge quality (minimum travel_time)
        best = min(data.values(), key=lambda d: float(d.get("travel_time", 9999)))
        
        # Accumulate travel_time dan distance
        time_s += float(best.get("travel_time", 9999))
        dist_m += float(best.get("length", 0))
    
    # Convert distance to "time equivalent" (distance in seconds at 40 km/h)
    dist_as_time = dist_m / REF_SPEED_MS
    
    # Final fitness = 50% speed + 50% distance
    return 0.5 * time_s + 0.5 * dist_as_time
```

**Contoh perhitungan:**

```
Path: Node 1 → 2 → 3

Edge (1,2):
  travel_time = 100 seconds
  length = 800 meters

Edge (2,3):
  travel_time = 150 seconds
  length = 1200 meters

Calculation:
  time_s = 100 + 150 = 250 seconds
  dist_m = 800 + 1200 = 2000 meters
  dist_as_time = 2000 / 11.1 = 180 seconds
  
  fitness = 0.5 * 250 + 0.5 * 180
          = 125 + 90
          = 215
```

**Mengapa 50-50 split?**
```
Dijkstra: hanya minimize travel_time
          hasil: bisa very long distance

GA: 50-50 compromise
    hasil: balance antara speed dan distance
    
Contoh:
- Dijkstra: Fast route, 250 sec, 5 km jarak
- GA: Balanced route, 260 sec, 3.5 km jarak
     → GA prefer karena jarak lebih pendek (physical)
```

---

## 4. _crossover() Method - Mixing 2 Parents

```python
def _crossover(self, p1: list, p2: list, rng: random.Random) -> list:
    """
    Mid-pivot common-node crossover untuk point-to-point paths.
    """
    # Find common nodes antara p1 dan p2
    set1 = set(p1[1:-1])  # semua nodes di p1 (skip start & end)
    common = [n for n in p2[1:-1] if n in set1]
    
    if not common:
        return p1[:]  # Jika tidak ada common nodes, return p1
    
    mid = len(p1) // 2
    
    # Find common node paling dekat ke middle
    min_dist = min(abs(p1.index(n) - mid) for n in common)
    best = [n for n in common if abs(p1.index(n) - mid) == min_dist]
    
    pivot = rng.choice(best)  # Choose one of the best nodes
    
    # Build child: first half dari p1, second half dari p2
    i1 = p1.index(pivot)
    i2 = p2.index(pivot)
    
    return p1[:i1] + p2[i2:]
```

**Contoh:**

```
p1 = [start, A, B, C, D, end]  (6 nodes)
p2 = [start, D, C, B, A, end]

Step 1: Find common nodes (skip start & end)
        p1[1:-1] = [A, B, C, D]
        p2[1:-1] = [D, C, B, A]
        common = [A, B, C, D]

Step 2: Find node paling dekat ke middle
        mid = 6 // 2 = 3
        
        Node A di p1: index 1, distance = |1-3| = 2
        Node B di p1: index 2, distance = |2-3| = 1 ← TERKECIL!
        Node C di p1: index 3, distance = |3-3| = 0 ← PALING KECIL!
        Node D di p1: index 4, distance = |4-3| = 1
        
        pivot = C (distance 0, paling dekat ke middle)

Step 3: Find pivot positions
        i1 = p1.index(C) = 3
        i2 = p2.index(C) = 2
        
        child = p1[:3] + p2[2:]
              = [start, A, B] + [C, B, A, end]
              = [start, A, B, C, B, A, end]

Result: child = [start, A, B, C, B, A, end]
        ↑ Mixing dari p1 (first half) dan p2 (second half)
```

---

## 5. _mutate() Method - Random Change

```python
def _mutate(self, G, path: list, rng: random.Random) -> list:
    """Standard segment re-route mutation"""
    return _ga_mutate(G, path, rng)
```

**Ini simple delegate ke helper function `_ga_mutate`.**

Mutation untuk point-to-point:
- Pilih random segment dari path
- Re-route segment itu (find alternative path)
- Hasil: path yang slightly different

---

## 6. _route_multi_stop() - THE MAIN TSP METHOD

Ini adalah core logic untuk multi-stop routing. Mari break down ke steps:

### Step 1: Setup & Validation

```python
t0 = time.perf_counter()

# Parse nodes: separate source, target, intermediates
start, end, intermediates = _split_multi_stop_nodes(
    nodes, source_node, target_node, round_trip
)
```

**Contoh:**
```
Input nodes: [1, 4, 2, 3]
source_node: 1
target_node: 1 (round trip)

Result:
  start = 1 (starting point)
  end = 1 (ending point, sama karena round trip)
  intermediates = [4, 2, 3] (nodes yang harus dikunjungi)
```

### Step 2: Precompute Pairwise Costs (CRITICAL!)

```python
# Run parallel Dijkstra dari setiap node
def _dijkstra_row(src):
    """One Dijkstra sweep dari src ke semua nodes"""
    return src, dict(nx.single_source_dijkstra_path_length(
        G, src, weight="travel_time"
    ))

pair_cost: dict = {}

# ThreadPoolExecutor: run multiple Dijkstra concurrently
with ThreadPoolExecutor(max_workers=self.TSP_WORKERS) as pool:
    for src, lengths in pool.map(_dijkstra_row, nodes):
        for dst in nodes:
            if dst != src:
                pair_cost[(src, dst)] = lengths.get(dst, float("inf"))
```

**Apa yang terjadi:**

```
Nodes: [1, 4, 2, 3]
TSP_WORKERS = 5 threads

Thread 1: Dijkstra dari 1
          Result: distances = {4: 100, 2: 150, 3: 200}
          Store: pair_cost[(1,4)]=100, pair_cost[(1,2)]=150, pair_cost[(1,3)]=200

Thread 2: Dijkstra dari 4
          Result: distances = {1: 110, 2: 80, 3: 150}
          Store: pair_cost[(4,1)]=110, pair_cost[(4,2)]=80, pair_cost[(4,3)]=150

Thread 3: Dijkstra dari 2
          Result: distances = {1: 160, 4: 85, 3: 120}
          Store: pair_cost[(2,1)]=160, pair_cost[(2,4)]=85, pair_cost[(2,3)]=120

Thread 4: Dijkstra dari 3
          Result: distances = {1: 210, 4: 160, 2: 130}
          Store: pair_cost[(3,1)]=210, pair_cost[(3,4)]=160, pair_cost[(3,2)]=130

Final pair_cost = {
    (1,4): 100,  (1,2): 150,  (1,3): 200,
    (4,1): 110,  (4,2): 80,   (4,3): 150,
    (2,1): 160,  (2,4): 85,   (2,3): 120,
    (3,1): 210,  (3,4): 160,  (3,2): 130,
}

Keuntungan:
- Parallel: 4 threads = ~4x faster (vs sequential)
- Reuse: Setiap generation gunakan same pair_cost (O(1) lookup)
- vs running Dijkstra per edge per individual per generation = SLOW
```

### Step 3: Tour Cost Helper

```python
def tour_cost(perm: list) -> float:
    """Calculate total cost untuk satu permutation"""
    full_tour = [start] + perm + [end]
    # perm = [4, 2, 3]
    # full_tour = [1, 4, 2, 3, 1]
    
    return sum(
        pair_cost.get((a, b), float("inf"))
        for a, b in zip(full_tour[:-1], full_tour[1:])
    )
    # cost = pair_cost[(1,4)] + pair_cost[(4,2)] + pair_cost[(2,3)] + pair_cost[(3,1)]
    #      = 100 + 80 + 120 + 210
    #      = 510
```

**Contoh:**
```
perm = [4, 2, 3]

full_tour = [1] + [4, 2, 3] + [1]
          = [1, 4, 2, 3, 1]

Pairs:
  (1,4): 100
  (4,2): 80
  (2,3): 120
  (3,1): 210

total = 100 + 80 + 120 + 210 = 510 minutes
```

### Step 4: Order Crossover (OX)

```python
def ox_crossover(p1: list, p2: list) -> list:
    """Order Crossover - standard TSP operator"""
    size = len(p1)
    
    # Step 1: Pilih 2 cut points random
    a, b = sorted(rng.sample(range(size), 2))
    # Contoh: a=0, b=2
    
    # Step 2: Copy segment dari p1
    child = [None] * size
    child[a:b + 1] = p1[a:b + 1]
    # child = [4, 2, 3]
    # child[0:3] = p1[0:3] = [4, 2, 3]
    # child = [4, 2, 3]
    
    # Step 3: Fill remaining dengan p2 (skip yang sudah ada)
    fill = [x for x in p2 if x not in child]
    # p2 = [2, 4, 3]
    # fill = [x for x in [2,4,3] if x not in [4,2,3]]
    #      = [] (semua sudah ada)
    
    j = 0
    for i in range(size):
        if child[i] is None:
            child[i] = fill[j]
            j += 1
    
    return child
```

**Step-by-step contoh:**
```
p1 = [4, 2, 3]  (indices: 0=4, 1=2, 2=3)
p2 = [2, 4, 3]  (indices: 0=2, 1=4, 2=3)

Step 1: Pick 2 cut points
        a, b = 0, 2
        Segment: p1[0:3] = [4, 2, 3]

Step 2: Copy segment ke child
        child = [4, 2, 3]
        (semua positions filled, no None)

Step 3: Fill remaining
        fill = [x for x in p2 if x not in child]
             = [x for x in [2,4,3] if x not in [4,2,3]]
             = []  (semua 2,4,3 sudah di child)

Result: child = [4, 2, 3]
        ↑ Valid permutation!
        
        Kalau p2 berbeda:
        p2 = [3, 4, 2]
        
        Step 1: Same, copy p1[0:3] = [4, 2, 3]
        Step 3: fill = [x for x in [3,4,2] if x not in [4,2,3]] = []
        Result: child = [4, 2, 3]
        
        Kalau cut points berbeda:
        a, b = 0, 1  (copy hanya p1[0:2])
        
        child = [None, None, None]
        child[0:2] = p1[0:2] = [4, 2]
        child = [4, 2, None]
        
        fill = [x for x in p2 if x not in [4,2]]
             = [x for x in [3,4,2] if x not in [4,2]]
             = [3]
        
        Step 3: Fill None dengan [3]
        child[2] = 3
        child = [4, 2, 3]
        
        Result: child = [4, 2, 3]
```

**Kenapa OX baik untuk TSP?**
```
OX = Order Crossover

Constraints untuk TSP:
- Setiap node harus dikunjungi EXACTLY ONCE
- No duplicates
- Valid permutation

OX guarantee:
- Copy segment dari p1 → no duplicates dalam segment
- Fill dari p2 (skip yang sudah ada) → no duplicates overall
- Result ALWAYS valid permutation

Vs simple crossover (cut & paste):
- p1 = [4, 2, 3]
- p2 = [2, 4, 3]
- cut at position 2: [4, 2] + [4, 3] = [4, 2, 4, 3]
- ↑ INVALID! Node 4 duplicate!

OX avoids ini dengan clever filling.
```

### Step 5: Swap Mutation

```python
def swap_mutate(perm: list) -> list:
    """Randomly swap 2-3 stops"""
    if len(perm) < 2:
        return perm[:]
    
    p = perm[:]
    num_swaps = rng.randint(2, 3)  # Random 2 atau 3 swaps
    
    for _ in range(num_swaps):
        i, j = rng.sample(range(len(p)), 2)  # Pick 2 random indices
        p[i], p[j] = p[j], p[i]  # Swap
    
    return p
```

**Contoh:**
```
perm = [4, 2, 3]  (indices: 0=4, 1=2, 2=3)
num_swaps = 2

Swap 1: Pick i=0, j=2
        p[0] ↔ p[2]
        [4, 2, 3] → [3, 2, 4]

Swap 2: Pick i=1, j=2
        p[1] ↔ p[2]
        [3, 2, 4] → [3, 4, 2]

Result: [3, 4, 2]

Kenapa 2-3 swaps (bukan 1)?
- 1 swap: small change, explore terlalu limited
- 2-3 swaps: bigger change, dapat explore lebih luas
- Contoh: [4,2,3] dengan 1 swap hanya bisa jadi:
          [2,4,3], [3,2,4], dll (24 kemungkinan dari 6 total)
          Tapi 2-3 swaps explore lebih banyak permutation
```

### Step 6: Evolution Loop

```python
for gen_idx in range(self.TSP_GENERATIONS):  # Max 600 gen
    # ── Evaluate current population ──
    fitness = [tour_cost(p) for p in population]
    best_idx = min(range(len(population)), key=lambda i: fitness[i])
    elite = population[best_idx]
    
    # ── Check improvement ──
    if fitness[best_idx] < best_cost - 1e-6:  # Improved!
        best_perm = elite[:]
        best_cost = fitness[best_idx]
        no_improve = 0  # Reset counter
    else:
        no_improve += 1
    
    # ── Logging ──
    full_order = [start] + best_perm + [end]
    gen_history.append(recorder.frame(gen_idx, full_order))
    
    # ── Early stopping ──
    if no_improve >= self.TSP_PATIENCE:  # 20 gen no improvement
        log.info(f"Early stop at gen {gen_idx+1}")
        break
    
    # ── Build next generation ──
    new_pop = [elite[:]]  # Keep best (elitism)
    
    while len(new_pop) < self.TSP_POPULATION_SIZE:
        # Tournament selection
        p1 = _ga_tournament(population, fitness, self.TOURNAMENT_SIZE, rng)
        
        # Crossover 80%
        if rng.random() < self.CROSSOVER_RATE:
            p2 = _ga_tournament(population, fitness, self.TOURNAMENT_SIZE, rng)
            child = ox_crossover(p1, p2)
        else:
            child = p1[:]
        
        # Mutation 90%
        if rng.random() < self.MUTATION_RATE:
            child = swap_mutate(child)
        
        new_pop.append(child)
    
    population = new_pop
```

**Alur setiap generasi:**
```
Gen N:
  ├─ Evaluate: Calculate fitness semua individu
  ├─ Update best: Jika ada improvement, update best_perm
  ├─ Check: Jika no_improve >= patience, BREAK (early stop)
  ├─ Log: Save gen_history untuk evolution log
  └─ Build next generation:
      ├─ Elite: Keep terbaik (elitism)
      ├─ Loop (untuk 49 more individuals):
      │   ├─ Select: Tournament pick p1
      │   ├─ Crossover: 80% mix p2, 20% copy p1
      │   ├─ Mutation: 90% swap, 10% no mutation
      │   └─ Add ke new_population
      └─ population = new_population

Gen N+1: Repeat
```

### Step 7: Expand Best Permutation to Road Path

```python
full_tour_stops = [start] + best_perm + [end]
# best_perm = [4, 2, 3]
# full_tour_stops = [1, 4, 2, 3, 1]

full_route: list = []

for src, dst in zip(full_tour_stops[:-1], full_tour_stops[1:]):
    # For each leg: (1,4), (4,2), (2,3), (3,1)
    try:
        leg = nx.shortest_path(G, src, dst, weight="travel_time")
        # leg = [1, 10, 20, 30, 4]  (actual road path)
    except ...:
        return RouteResult.failure(...)
    
    if not full_route:
        full_route.extend(leg)
    else:
        # Skip first node to avoid duplication
        full_route.extend(leg[1:])

# Final full_route: complete road path dari start sampai end
```

**Contoh:**
```
full_tour_stops = [1, 4, 2, 3, 1]

Legs:
  (1,4): shortest_path = [1, 10, 20, 4]
  (4,2): shortest_path = [4, 50, 2]
  (2,3): shortest_path = [2, 60, 70, 3]
  (3,1): shortest_path = [3, 80, 1]

Building full_route:
  leg 1: [1, 10, 20, 4]
         full_route = [1, 10, 20, 4]
  
  leg 2: [4, 50, 2]
         Skip 4 (already in full_route)
         full_route.extend([50, 2]) = [1, 10, 20, 4, 50, 2]
  
  leg 3: [2, 60, 70, 3]
         Skip 2
         full_route.extend([60, 70, 3]) = [1, 10, 20, 4, 50, 2, 60, 70, 3]
  
  leg 4: [3, 80, 1]
         Skip 3
         full_route.extend([80, 1]) = [1, 10, 20, 4, 50, 2, 60, 70, 3, 80, 1]

Final: full_route = [1, 10, 20, 4, 50, 2, 60, 70, 3, 80, 1]
       ↑ Complete road network path!
```

### Step 8: Build RouteResult

```python
ms = (time.perf_counter() - t0) * 1000

return RouteResult.build(
    G, self.name, scenario_name,
    start, end,
    full_route, ms,
    metadata={
        "algorithm_variant": "tsp_ga",
        "order_objective": "ga_stop_permutation_travel_time",
        "order_score": best_cost,  # Cost dalam minutes
        "generations": len(gen_history),
        "population": self.TSP_POPULATION_SIZE,
        "crossover_rate": self.CROSSOVER_RATE,
        "mutation_rate": self.MUTATION_RATE,
        "stop_count": len(nodes),
        "round_trip": round_trip,
        "visit_order": full_tour_stops,  # [1, 4, 2, 3, 1]
        "visit_order_nodes": full_tour_stops,
        "gen_history": gen_history,  # Per-generation info
    },
)
```

---

## 7. find_route() Method

```python
def find_route(self, G, source_node, target_node, scenario_name=""):
    return _ga_run(self, G, source_node, target_node, scenario_name)
```

**Simple delegate ke helper function `_ga_run`**

Ini untuk point-to-point routing (cari jalan dari A ke B).

---

## 8. Complete Flow Summary

```
┌─────────────────────────────────────────────┐
│ Multi-Stop Routing Problem                   │
│ Nodes: [Hospital, School, Police, Park]    │
│ Need: Find optimal visit order               │
└─────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────┐
│ _route_multi_stop() START                    │
│                                               │
│ Input: nodes=[1,4,2,3], start=1, end=1     │
│ intermediates = [4,2,3]                     │
└─────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────┐
│ STEP 1: Precompute Pairwise Costs            │
│                                               │
│ 4 parallel Dijkstra sweeps                  │
│ Result: pair_cost dict dengan semua costs   │
│         pair_cost[(1,4)]=100, etc           │
└─────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────┐
│ STEP 2: Initialize Population                │
│                                               │
│ 1 greedy NN: [4, 2, 3]                      │
│ 49 random shuffles                          │
│ Total: 50 individuals                       │
└─────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────┐
│ STEP 3-6: Evolution Loop (Gen 1-600)        │
│                                               │
│ For each generation:                        │
│   1. Evaluate: calculate tour_cost semuanya │
│   2. Update best: if improved, update       │
│   3. Check: if no improve >= 20, STOP       │
│   4. Log: save gen_history                  │
│   5. Breed: tournament + crossover + mutate │
│   6. New population                         │
│                                               │
│ Result: best_perm = [visit_order]           │
└─────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────┐
│ STEP 7: Expand to Full Road Path            │
│                                               │
│ best_perm = [4, 2, 3]                       │
│ For each leg: run shortest_path             │
│ Result: full_route = [actual node sequence] │
└─────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────┐
│ STEP 8: Build RouteResult                    │
│                                               │
│ Return:                                      │
│   - full_route (actual path nodes)          │
│   - visit_order (optimal sequence)          │
│   - travel_time, distance, ms               │
│   - metadata (generations, population, etc) │
└─────────────────────────────────────────────┘
```

---

## 9. Key Design Decisions Explained

### A. Parallel Pairwise Dijkstra

**Why?**
```
N nodes = N² pairs

Approach 1 (Naive):
  For each individual:
    For each pair (i,j):
      Run Dijkstra
  Total: POPULATION_SIZE × TSP_GENERATIONS × N²  runs!

Approach 2 (Smart - yang dipakai):
  Once: Run N Dijkstra (one per source)
  Store: pair_cost dict
  Every individual: O(N) lookup
  Total: N Dijkstra + POPULATION_SIZE × TSP_GENERATIONS × N lookups!
  
Speedup: Dijkstra expensive, lookup cheap
         1000x faster atau lebih!
```

### B. Greedy NN Initialization

**Why?**
```
Pure random:
  Gen 1: [4,2,3]=510 min, [2,3,4]=520 min, ...
  Problem: Starting dari bad solutions

Hybrid (1 greedy + 49 random):
  Gen 1: [4,2,3]=310 min (greedy), [2,3,4]=520 min, ...
  Benefit: Start dengan good baseline
           GA evolve dari situ ke lebih baik
           Faster convergence
```

### C. Early Stopping (Patience)

**Why?**
```
Without early stop:
  Run semua 600 generasi bahkan jika sudah converge
  Waste time jika no improvement 100 gen

With early stop (patience=20):
  If 20 gen berturut-turut no improvement:
    population sudah converge
    fitness plateau
    Stop dan save time!
  
Contoh:
  Gen 1-150: steady improvement
  Gen 151-170: no improvement (stop here!)
  Saves: 430 generasi × 50 evaluasi = 21500 tour_cost calculations
```

### D. Elitism

**Why?**
```python
new_pop = [elite[:]]  # Keep best!
while len(new_pop) < TSP_POPULATION_SIZE:
    # Breed rest from tournament
```

Benefit:
```
Without elitism:
  Best individual bisa di-eliminate
  GA bisa go backward (fitness increase)

With elitism:
  Best individual guaranteed survive
  GA monotonically improve (non-decreasing)
  Fitness history always trending down (better)
```

---

## Summary: Why This Design Works

```
1. PARALLEL PREPROCESSING (Dijkstra once, reuse forever)
   ↓ Massive speedup

2. HYBRID INITIALIZATION (good seed + exploration)
   ↓ Fast convergence from good baseline

3. OX CROSSOVER (valid permutations guaranteed)
   ↓ No invalid solutions, pure evolution

4. MULTIPLE SWAPS MUTATION (2-3 vs 1)
   ↓ Larger neighborhood exploration

5. TOURNAMENT SELECTION + ELITISM
   ↓ Convergence dengan maintain best

6. EARLY STOPPING (patience=20)
   ↓ Stop when converged, save time

Result:
  - Fast: parallel Dijkstra + early stopping
  - Good: greedy seed + 50-50 hyperparameters
  - Valid: OX + swap mutation guarantee permutations
  - Robust: tournament selection + elitism
```
