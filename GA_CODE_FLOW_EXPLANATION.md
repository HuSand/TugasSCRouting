# GA Code Flow - Penjelasan Alur Kode

## 1. Entry Point: find_route() Method

Ketika GA dipanggil untuk solve routing problem, ini adalah entry point:

```python
def find_route(self, G, source_node, target_node, scenario_name):
    """
    Entry point untuk GA routing
    """
    # Check apakah ini multi-stop problem
    if hasattr(self, "_route_multi_stop"):
        # Jika ada waypoints, gunakan multi-stop version
        return self._route_multi_stop(G, source_node, target_node, scenario_name)
    else:
        # Jika hanya point-to-point, gunakan standard shortest path
        return self._find_shortest_path_dijkstra(G, source_node, target_node)
```

**Alur:**
```
find_route() dipanggil
    ↓
Check: Apakah ini multi-stop (punya waypoints)?
    ↓
YA → Panggil _route_multi_stop() [GA TSP mode]
TIDAK → Gunakan Dijkstra [Simple point-to-point]
```

---

## 2. Multi-Stop TSP Mode: _route_multi_stop()

Ini adalah core logic untuk GA ketika menghadapi Traveling Salesman Problem (mencari urutan optimal untuk visit multiple locations).

### 2.1 Input yang diterima

```python
def _route_multi_stop(self, G, source_node, target_node, scenario_name):
    """
    Args:
        G: NetworkX graph dengan nodes/edges
        source_node: Starting location (e.g., hospital node)
        target_node: End location (biasanya sama dengan source untuk round trip)
        scenario_name: Nama scenario (untuk logging)
    """
```

**Input contoh:**
```
source_node = 1 (Hospital)
target_node = 1 (kembali ke Hospital)
intermediates = [4, 2, 3] ← Nodes yang harus dikunjungi
scenario_name = "emergency_patrol_circuit"
```

### 2.2 Phase 1: Prepare Pairwise Costs

```python
# Step 1: Build cost matrix (distance antar nodes)
intermediates = [n for n in scenario.waypoints if n != source_node]

# Step 2: Precompute pairwise dijkstra (parallel)
pair_cost = {}
# Dijkstra dari setiap node ke setiap node lainnya
# Hasilnya: pair_cost[(1,2)] = 150 min, pair_cost[(1,3)] = 200 min, dll
```

**Diagram alur:**
```
intermediates = [4, 2, 3]
source_node = 1

Step 1: Hitung cost dari 1 ke semuanya:
        1→4: 100 min
        1→2: 150 min
        1→3: 200 min

Step 2: Hitung cost dari 4 ke semuanya:
        4→1: 110 min
        4→2: 80 min
        4→3: 150 min

Step 3: Hitung cost dari 2 ke semuanya:
        2→1: 160 min
        2→4: 85 min
        2→3: 120 min

Step 4: Hitung cost dari 3 ke semuanya:
        3→1: 210 min
        3→4: 160 min
        3→2: 130 min

Hasil: pair_cost = {
    (1,4): 100, (1,2): 150, (1,3): 200,
    (4,1): 110, (4,2): 80,  (4,3): 150,
    (2,1): 160, (2,4): 85,  (2,3): 120,
    (3,1): 210, (3,4): 160, (3,2): 130,
}
```

### 2.3 Phase 2: Initialize Population

```python
# Step 1: Create greedy nearest-neighbor ordering
def greedy_nn_order(start_node, targets, pair_cost_dict):
    remaining = set(targets)
    order = []
    current = start_node
    while remaining:
        nearest = min(remaining, 
                     key=lambda n: pair_cost_dict.get((current, n), float("inf")))
        order.append(nearest)
        remaining.remove(nearest)
        current = nearest
    return order

# Step 2: Build population
population = []

# 1 Greedy NN individual
greedy_perm = greedy_nn_order(source_node, intermediates, pair_cost)
population.append(greedy_perm)
# Contoh: greedy_perm = [4, 2, 3]

# 19 Random shuffle individuals
for _ in range(TSP_POPULATION_SIZE - 1):  # TSP_POPULATION_SIZE = 20
    perm = intermediates[:]  # Copy [4, 2, 3]
    random.shuffle(perm)     # Random shuffle → [2, 4, 3] atau [3, 2, 4], dll
    population.append(perm)

# Final population (20 individu):
population = [
    [4, 2, 3],      # ← greedy NN seed
    [2, 4, 3],      # ← random
    [3, 2, 4],      # ← random
    [4, 3, 2],      # ← random
    ...
]
```

**Alur visualisasi:**
```
Greedy NN untuk [4, 2, 3] dari source 1:
  Start: current=1
  nearest(4,2,3 dari 1)? → 4 (cost 100)
  order = [4], current=4, remaining={2,3}
  
  nearest(2,3 dari 4)? → 2 (cost 80)
  order = [4,2], current=2, remaining={3}
  
  nearest(3 dari 2)? → 3 (cost 120)
  order = [4,2,3], remaining={}
  
  RESULT: greedy_perm = [4,2,3]
  Cost: 100 + 80 + 120 = 300 min ← GOOD!
  
Vs Random:
  [2,4,3]: 150 + 150 + 130 = 430 min
  [3,2,4]: 200 + 130 + 85 = 415 min
  [4,3,2]: 100 + 150 + 130 = 380 min
  
Greedy lebih baik! Inilah kenapa seed dengan greedy.
```

### 2.4 Phase 3: Evaluate Generation 1

```python
# Hitung fitness untuk setiap individu di populasi
def tour_cost(perm):
    """Total cost untuk traverse permutation + kembali ke source"""
    cost = 0
    current = source_node
    for next_node in perm:
        cost += pair_cost.get((current, next_node), float("inf"))
        current = next_node
    cost += pair_cost.get((current, source_node), float("inf"))  # Return ke source
    return cost

fitness = [tour_cost(p) for p in population]

# fitness = [300, 420, 410, 380, 430, ...]
# best_idx = 0 (index dengan fitness terendah)
# best_perm = [4, 2, 3]
# best_cost = 300

best_idx = min(range(len(population)), key=lambda i: fitness[i])
best_perm = population[best_idx][:]
best_cost = fitness[best_idx]
```

**Contoh perhitungan tour_cost:**
```
tour_cost([4, 2, 3]):
  current = 1 (source)
  cost = 0
  
  Visit 4: cost += pair_cost[(1,4)] = 100, current = 4, cost = 100
  Visit 2: cost += pair_cost[(4,2)] = 80,  current = 2, cost = 180
  Visit 3: cost += pair_cost[(2,3)] = 120, current = 3, cost = 300
  Return:  cost += pair_cost[(3,1)] = 210, current = 1, cost = 510

WAIT! Ini 510, bukan 300?
Tergantung pair_cost[(3,1)]. Jika actual cost dari 3→1 adalah 210, 
maka total adalah 100 + 80 + 120 + 210 = 510 min.

Tapi bisa juga lebih murah tergantung road network.
```

### 2.5 Phase 4: Evolution Loop (Generasi 2-30)

```python
best_cost = fitness[best_idx]
no_improve = 0
patience = 5

for generation in range(1, GENERATIONS):  # 30 generasi
    new_population = []
    
    # ============================================
    # STEP 1: SELECTION (Tournament)
    # ============================================
    while len(new_population) < TSP_POPULATION_SIZE:  # 20 individu
        # Tournament: Pilih 3 individu random, ambil yang terbaik
        candidates = rng.sample(range(TSP_POPULATION_SIZE), TOURNAMENT_SIZE)
        # candidates = [5, 12, 8]
        
        winner_idx = min(candidates, key=lambda i: fitness[i])
        # Misal fitness[5]=400, fitness[12]=350, fitness[8]=380
        # winner_idx = 12 (terendah = 350)
        
        parent = population[winner_idx][:]
        # parent = population[12] = [4, 3, 2]
        
        # ============================================
        # STEP 2: CROSSOVER (80% chance)
        # ============================================
        if rng.random() < CROSSOVER_RATE:  # 80%
            # Pilih parent2 juga pake tournament
            candidates2 = rng.sample(range(TSP_POPULATION_SIZE), TOURNAMENT_SIZE)
            parent2_idx = min(candidates2, key=lambda i: fitness[i])
            parent2 = population[parent2_idx][:]
            
            # Order Crossover (OX)
            child = order_crossover(parent, parent2)
            # parent  = [4, 3, 2]
            # parent2 = [2, 4, 3]
            # child  = [4, 2, 3] atau [2, 3, 4] (kombinasi valid)
        else:
            # 20% chance: tidak ada crossover, child = copy parent
            child = parent[:]
        
        # ============================================
        # STEP 3: MUTATION (90% chance)
        # ============================================
        if rng.random() < MUTATION_RATE:  # 90%
            child = swap_mutate(child)
            # child sebelum = [4, 2, 3]
            # swap 2-3 kali:
            #   swap(0,2): [3, 2, 4]
            #   swap(1,2): [3, 4, 2]
            # child sesudah = [3, 4, 2]
        
        new_population.append(child)
    
    # ============================================
    # STEP 4: EVALUATE NEW GENERATION
    # ============================================
    fitness = [tour_cost(p) for p in new_population]
    best_current_idx = min(range(len(new_population)), 
                          key=lambda i: fitness[i])
    
    if fitness[best_current_idx] < best_cost:
        # Ada improvement!
        best_cost = fitness[best_current_idx]
        best_perm = new_population[best_current_idx][:]
        no_improve = 0
    else:
        # Tidak ada improvement
        no_improve += 1
    
    # ============================================
    # STEP 5: EARLY STOPPING
    # ============================================
    if no_improve >= patience:  # 5 generasi tanpa improvement
        # Stop early, tidak perlu lanjut ke gen 30
        break
    
    population = new_population

# ============================================
# STEP 6: FINAL RESULT
# ============================================
# best_perm = [4, 2, 3]
# best_cost = 285 min
# final_route = source → 4 → 2 → 3 → source
```

### 2.6 Order Crossover (OX) Detail

```python
def order_crossover(parent1, parent2):
    """
    Crossover untuk permutasi yang maintain validity
    
    parent1 = [4, 2, 3]
    parent2 = [2, 4, 3]
    """
    n = len(parent1)
    # Pilih 2 crossover points random
    point1, point2 = sorted(rng.sample(range(n), 2))
    
    # point1 = 0, point2 = 2
    
    # Step 1: Copy segment dari parent1
    child = [None] * n
    child[point1:point2] = parent1[point1:point2]
    # child = [None, 2, 3]
    
    # Step 2: Fill remaining dengan parent2 (maintain order)
    pointer = point2 % n
    parent2_pointer = point2 % n
    
    while None in child:
        if parent2[parent2_pointer] not in child:
            child[pointer] = parent2[parent2_pointer]
            pointer = (pointer + 1) % n
        parent2_pointer = (parent2_pointer + 1) % n
    
    # child = [4, 2, 3] atau [2, 2, 3] (tergantung kondisi)
    
    return child
```

**Contoh OX step-by-step:**
```
parent1 = [4, 2, 3]
parent2 = [2, 4, 3]
point1 = 0, point2 = 2

Step 1: Copy parent1[0:2] ke child
        child = [4, 2, None]

Step 2: Fill None dengan parent2 (skip yang sudah ada)
        Iterate parent2 dari index 2:
        parent2[2] = 3: 3 not in [4,2], add → child[2] = 3
        
        child = [4, 2, 3]

Result: child = [4, 2, 3] ← Valid permutation!
```

### 2.7 Swap Mutate Detail

```python
def swap_mutate(perm: list) -> list:
    """
    Mutasi dengan 2-3 random swaps
    
    perm = [4, 2, 3]
    """
    if len(perm) < 2:
        return perm[:]
    
    p = perm[:]
    num_swaps = rng.randint(2, 3)  # Random 2 atau 3 swaps
    
    for _ in range(num_swaps):
        i, j = rng.sample(range(len(p)), 2)  # Pick 2 random indices
        p[i], p[j] = p[j], p[i]  # Swap
    
    return p

# Contoh:
# perm = [4, 2, 3]
# num_swaps = 2
# 
# Swap 1: i=0, j=2 → swap(4,3) → p = [3, 2, 4]
# Swap 2: i=1, j=2 → swap(2,4) → p = [3, 4, 2]
# 
# Result: [3, 4, 2] ← Permutation berubah!
```

---

## 3. Alur Keseluruhan Dari Awal Sampai Akhir

```
┌─────────────────────────────────────────────────────────────┐
│ find_route() dipanggil dengan scenario multi-stop            │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ _route_multi_stop() START                                    │
│                                                               │
│ Input:                                                        │
│  - source_node = 1 (Hospital)                                │
│  - intermediates = [4, 2, 3] (Schools, Stations)             │
│  - pair_cost = {(1,4): 100, (4,2): 80, ...}                │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: Initialize Population (20 individuals)              │
│                                                               │
│  - greedy_nn_order([4,2,3]) → [4, 2, 3]                     │
│  - 19 random shuffle                                         │
│                                                               │
│  Population = [                                              │
│    [4, 2, 3],  ← greedy (Gen1 cost=300)                     │
│    [2, 4, 3],  ← random (Gen1 cost=420)                     │
│    [3, 2, 4],  ← random (Gen1 cost=410)                     │
│    ...                                                        │
│  ]                                                            │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: Evaluate Generation 1                               │
│                                                               │
│  fitness = [300, 420, 410, 430, ...]                        │
│  best_perm = [4, 2, 3]                                      │
│  best_cost = 300                                            │
│  no_improve = 0                                              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 3-4: Evolution Loop (Generation 2-30)                 │
│                                                               │
│ For gen = 2 to 30:                                           │
│   ┌──────────────────────────────────────────────────┐      │
│   │ Create new_population (20 individuals)            │      │
│   │                                                    │      │
│   │ For each individual:                              │      │
│   │   1. SELECTION: Tournament pick (size=3)          │      │
│   │      → parent = population[tournament_winner]     │      │
│   │                                                    │      │
│   │   2. CROSSOVER: 80% chance                        │      │
│   │      if random() < 0.8:                           │      │
│   │          parent2 = tournament_pick()              │      │
│   │          child = order_crossover(parent, parent2) │      │
│   │      else:                                         │      │
│   │          child = parent                           │      │
│   │                                                    │      │
│   │   3. MUTATION: 90% chance                         │      │
│   │      if random() < 0.9:                           │      │
│   │          child = swap_mutate(child)  # 2-3 swaps │      │
│   │                                                    │      │
│   │   4. Add child ke new_population                  │      │
│   │                                                    │      │
│   └──────────────────────────────────────────────────┘      │
│                                                               │
│   Evaluate new_population:                                   │
│   fitness = [tour_cost(p) for p in new_population]          │
│   best_current = min(fitness)                                │
│                                                               │
│   Check improvement:                                         │
│   if best_current < best_cost:                              │
│       best_cost = best_current                              │
│       best_perm = new_population[best_current_idx]          │
│       no_improve = 0                                         │
│   else:                                                      │
│       no_improve += 1                                        │
│                                                               │
│   Early stopping:                                            │
│   if no_improve >= 5:  # patience=5                          │
│       BREAK (stop generation loop)                           │
│                                                               │
│   population = new_population  # update untuk next gen       │
│                                                               │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 5: Final Result                                        │
│                                                               │
│ best_perm = [4, 2, 3]  (optimal visit order)                │
│ best_cost = 285 min    (total travel time)                  │
│                                                               │
│ Build final route:                                           │
│ route = dijkstra_path(1 → 4) +                              │
│         dijkstra_path(4 → 2) +                              │
│         dijkstra_path(2 → 3) +                              │
│         dijkstra_path(3 → 1)                                │
│                                                               │
│ Return RouteResult(                                          │
│   route=route_nodes,                                        │
│   travel_time_s=best_cost*60,                               │
│   distance_m=total_distance,                                │
│   computation_ms=elapsed_time,                              │
│   metadata={'visit_order': [4,2,3], 'generation': 28}       │
│ )                                                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Key Concepts

### A. Population vs Generation

```
Generation 1:
  Population = [
    [4, 2, 3],    ← Individu 1
    [2, 4, 3],    ← Individu 2
    [3, 2, 4],    ← Individu 3
    ...           ← 20 individu total
  ]

Generation 2:
  Population = [
    [4, 3, 2],    ← Individu baru (dari crossover/mutasi)
    [2, 3, 4],    ← Individu baru
    [4, 2, 3],    ← Individu baru
    ...           ← 20 individu baru
  ]

Setiap generation = 20 individu baru (hasil crossover + mutasi dari gen sebelumnya)
```

### B. Fitness Function

```python
def tour_cost(perm):
    """
    Cost untuk permutation = total travel time untuk visit semua dalam urutan
    """
    cost = 0
    current = source_node
    
    for next_node in perm:
        cost += pair_cost[(current, next_node)]
        current = next_node
    
    cost += pair_cost[(current, source_node)]  # Return ke start
    
    return cost
```

**Contoh:**
```
perm = [4, 2, 3]
source = 1

Path:
  1 → 4: cost = 100
  4 → 2: cost = 80
  2 → 3: cost = 120
  3 → 1: cost = 210
  
Total cost = 100 + 80 + 120 + 210 = 510 min

Fitness = 510 min (terendah = terbaik)
```

### C. Selection Mechanism

```
Tournament Selection:
  - Pick 3 individu random dari populasi
  - Ambil yang punya fitness terendah (best cost)
  - Individu itu menjadi parent untuk breeding
  
Contoh:
  Population fitness = [300, 420, 410, 350, 380, ...]
  
  Tournament 1: pick indices [5, 12, 8]
               fitness = [380, 350, 410]
               winner = index 12 (fitness 350) ← paling bagus
  
  Tournament 2: pick indices [2, 15, 7]
               fitness = [410, 390, 400]
               winner = index 15 (fitness 390) ← paling bagus
  
  Keduanya menjadi parent untuk crossover
```

### D. Crossover (Perkawinan)

```
Order Crossover (OX) - maintain permutation validity

parent1 = [4, 2, 3]
parent2 = [2, 4, 3]

Step 1: Pilih 2 crossover points
        point1 = 0, point2 = 2
        
Step 2: Copy segment dari parent1
        child[0:2] = parent1[0:2] = [4, 2]
        child = [4, 2, None]
        
Step 3: Fill None dengan order dari parent2
        remaining dari parent2 = [4, 3] (skip 2 karena sudah ada)
        Tapi 4 sudah ada, jadi ambil 3
        child = [4, 2, 3]
        
Result: child = [4, 2, 3] ← Valid permutation!
```

### E. Mutation

```
Swap Mutation - random swap 2-3 kali

perm = [4, 2, 3]
num_swaps = 2

Swap 1: random pick i=0, j=2
        perm[0] ↔ perm[2] → [3, 2, 4]

Swap 2: random pick i=1, j=2
        perm[1] ↔ perm[2] → [3, 4, 2]

Result: [3, 4, 2] ← Berbeda dari parent!
```

---

## 5. Why Each Component Matters

### Selection (Tournament Size = 3)
```
Fungsi: Pilih individu terbaik untuk breeding
Alasan: Balance antara:
  - Memilih yang bagus (fitness pressure)
  - Maintain diversitas populasi
  
Terlalu besar (size=5): Best always win → kurang diversitas
Terlalu kecil (size=1): Random pick → tidak ada fitness pressure
```

### Crossover (Rate = 80%)
```
Fungsi: Combine 2 parent menjadi 1 child
Alasan: 
  - Inherit good traits dari kedua parents
  - Combine best features
  
80% chance: Mostly children have mixing
20% chance: Some children copy parent (maintain diversity)
```

### Mutation (Rate = 90%, 2-3 swaps)
```
Fungsi: Random change pada permutation
Alasan:
  - Explore neighborhood solution
  - Escape local optima
  - Introduce variation jika populasi converge
  
90% rate: Banyak exploration (vs pure exploitation)
2-3 swaps: Perubahan signifikan (vs 1 swap)
```

---

## 6. Hyperparameters & What They Do

```
TSP_POPULATION_SIZE = 20
  → Berapa banyak individu per generation
  → Lebih besar = lebih explore, tapi lebih slow

TSP_GENERATIONS = 30
  → Berapa generasi maksimal
  → Lebih banyak = lebih lama evolve, tapi diminish returns

MUTATION_RATE = 0.9
  → Probability setiap individu dimutasi
  → 0.9 = 90% mutasi, 10% copy
  → Tinggi = banyak variation, rendah = converge cepat

CROSSOVER_RATE = 0.8
  → Probability crossover (vs copy parent)
  → 0.8 = 80% ada crossover, 20% copy

TOURNAMENT_SIZE = 3
  → Berapa banyak pick untuk tournament selection
  → 3 = moderate selection pressure
  → Lebih besar = prefer top individuals more
```

---

## 7. Debug Points - Cara Cek Apakah GA Bekerja

### Check 1: Population Diversity Gen 1

```python
print("Generation 1 Population:")
for i, perm in enumerate(population):
    print(f"  Individual {i}: {perm} -> cost = {fitness[i]}")

Expected:
  - Ada 1 yang jauh lebih baik (greedy seed, ~300)
  - 19 lainnya random dengan fitness ~400-600
  - Jangan semua sama!
```

### Check 2: Fitness Improvement Over Generations

```python
print(f"Gen {gen}: best_cost = {best_cost}, visit_order = {best_perm}")

Expected:
  Gen 1:  best_cost = 300, visit_order = [4, 2, 3]
  Gen 2:  best_cost = 295, visit_order = [4, 3, 2]
  Gen 3:  best_cost = 290, visit_order = [2, 4, 3]
  ...
  Gen 30: best_cost = 280, visit_order = [3, 4, 2]
  
  ✓ Trend downward (improvement)
  ✓ Visit order changing setiap generasi (evolving!)
  ✗ Jika semua Gen 1-30 sama visit order = PROBLEM!
```

### Check 3: Mutation Actually Works

```python
# Test swap_mutate:
perm = [4, 2, 3]
for i in range(10):
    mutated = swap_mutate(perm)
    print(f"  Mutation {i}: {perm} → {mutated}")

Expected:
  Mutation 0: [4, 2, 3] → [3, 4, 2]
  Mutation 1: [4, 2, 3] → [2, 3, 4]
  Mutation 2: [4, 2, 3] → [4, 3, 2]
  ...
  ✓ Setiap mutasi hasilkan permutasi berbeda
  ✗ Jika semua hasil sama = mutation tidak jalan
```

### Check 4: Early Stopping Working

```python
print(f"no_improve = {no_improve}, patience = {patience}")
if no_improve >= patience:
    print("EARLY STOP at generation", generation)

Expected:
  - Jika improvement terus ada, run sampai generation 30
  - Jika 5 generasi berturut-turut no improvement, stop
  - Check apakah early stop reasonable (tidak too early)
```

---

## Summary: GA Code Flow

```
1. find_route() → detect multi-stop → call _route_multi_stop()

2. _route_multi_stop():
   a. Compute pairwise costs (Dijkstra)
   b. Initialize population (1 greedy + 19 random)
   c. Evaluate Gen 1 (tour_cost untuk setiap)
   d. Loop Gen 2-30:
      - Selection: tournament pick
      - Crossover: 80% chance mix 2 parents
      - Mutation: 90% chance swap 2-3 times
      - Evaluate new population
      - Check improvement, early stop if no progress
   e. Return best_perm + best_cost
   
3. Build final route dari best_perm order
   
4. Return RouteResult dengan route + stats
```
