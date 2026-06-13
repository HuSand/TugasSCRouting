# Penjelasan Lengkap Benchmark.py Flow

## 1. Overview: Apa Itu Benchmark.py?

Benchmark.py adalah **orchestrator** untuk menjalankan semua algorithms pada multiple scenarios dan mengumpulkan hasil.

**High-level flow:**
```
Registry
  ↓ (daftar semua algorithms)
Scenarios  ← Built dari facility data
  ↓ (daftar semua routing problems)
For each scenario:
  For each algorithm:
    Run algorithm → RouteResult
    Log evolution (if GA)
  ↓
Aggregate results → CSV + Charts + Maps
```

---

## 2. AlgorithmRegistry - Mendaftar Algorithms

### Apa itu Registry?

```python
class AlgorithmRegistry:
    def __init__(self):
        self._algos: Dict[str, BaseRoutingAlgorithm] = {}
    
    def register(self, algo: BaseRoutingAlgorithm):
        """Tambah algorithm ke registry"""
        if algo.name in self._algos:
            log.warning(f"Overwriting '{algo.name}'")
        self._algos[algo.name] = algo
    
    def all(self) -> List[BaseRoutingAlgorithm]:
        """Return semua algorithms yang terdaftar"""
        return list(self._algos.values())
```

**Contoh penggunaan:**
```python
registry = AlgorithmRegistry()
registry.register(DijkstraTime())           # Dijkstra minimize time
registry.register(DijkstraDistance())       # Dijkstra minimize distance
registry.register(AStarTime())              # A* minimize time
registry.register(AStarDistance())          # A* minimize distance
registry.register(GeneticAlgorithm())       # GA (our focus!)
registry.register(SimulatedAnnealingAlgorithm())
registry.register(ChristofidesAlgorithm())
registry.register(AntColonyElitePro())
registry.register(GeraldSimulatedAnnealing())
registry.register(ParticleSwarmRouting())

# Sekarang registry punya 10 algorithms
registry.names()  # ['dijkstra_time', 'dijkstra_dist', 'astar_time', ...]
```

---

## 3. Scenarios - Mendifinisikan Routing Problems

### Apa itu Scenario?

```python
class Scenario:
    name: str                    # e.g., "emergency_patrol_circuit"
    description: str             # Purpose & insights
    source_node: int             # Starting node
    target_node: int             # Ending node
    source_label: str            # Name (e.g., "Polda Jatim")
    target_label: str            # Name
    route_nodes: list            # All nodes to visit
    route_labels: list           # Names for each node
    optimize_order: bool = False # Should benchmark compute best order?
    round_trip: bool = False     # Return to start?
```

**Contoh Scenario 1: Emergency Patrol Circuit**
```python
emergency_scenario = Scenario(
    name="emergency_patrol_circuit",
    description="Police + fire stations circuit",
    source_node=9156956728,       # Polda Jatim (starting point)
    target_node=9156956728,       # Same (round trip)
    source_label="Polda Jatim",
    target_label="Polda Jatim",
    route_nodes=[9156956728, 1234567, 2345678, ..., 9999999],  # 50 stations
    route_labels=["Polda Jatim", "Polda Surabaya", "Fire Station A", ...],
    optimize_order=True,          # Benchmark will compute optimal order!
    round_trip=True,              # Return to Polda Jatim
)

# This scenario asks:
# "What's the best order to visit all 50 emergency stations,
#  starting and ending at Polda Jatim?"
```

**Contoh Scenario 2: Terminal Circuit**
```python
terminal_scenario = Scenario(
    name="terminal_circuit",
    description="Bus terminals + ports + gas stations circuit",
    source_node=1111111,
    target_node=1111111,
    source_label="Terminal A",
    target_label="Terminal A",
    route_nodes=[1111111, 2222222, 3333333, ...],  # ~20 terminals
    route_labels=["Terminal A", "Terminal B", ...],
    optimize_order=True,
    round_trip=True,
)
```

### Scenario Builder: build_category_scenarios()

```python
def build_category_scenarios(G, fac, max_emergency=50) -> List[Scenario]:
    """
    Build 2 scenarios dari facility GeoDataFrame.
    
    Input:
      G = NetworkX graph dengan road network
      fac = GeoDataFrame dengan facilities (hospitals, police, etc)
      max_emergency = maksimal facilities per scenario
    
    Output:
      [emergency_scenario, terminal_scenario]
    """
```

**Proses:**

```
Step 1: Load facilities dari GeoDataFrame
        fac = facilities dengan columns:
          - category (emergency, terminal, gas_station)
          - nearest_node (snapped ke road network)
          - lat, lon (coordinates)
          - name (facility name)

Step 2: Filter facilities per category
        emg = fac[fac["category"] == "emergency"]  # ~100 facilities
        trm = fac[fac["category"] == "terminal"]   # ~20 facilities

Step 3: Dedup (one facility per nearest_node)
        emg = dedup(emg)  # Remove duplicate nodes
        trm = dedup(trm)

Step 4: Geographic sampling (if > max_emergency)
        if len(emg) > 50:
            emg = diverse_subset(emg, 50)
        # Pilih 50 facilities yang tersebar di Surabaya

Step 5: Order dengan nearest-neighbor
        emg = nn_order(emg, start_node=POLDA_NODE)
        # Urutkan dengan geographic proximity
        # Start dari Polda Jatim (police headquarters)
        # Result: [Polda, Station1, Station2, ...] (geographic circuit)

Step 6: Create scenario objects
        scenario1 = make_scenario(emg, "emergency_patrol_circuit", ...)
        scenario2 = make_scenario(trm, "terminal_circuit", ...)

Return: [scenario1, scenario2]
```

**Nearest-Neighbor Geographic Ordering:**
```python
def _nn_order(df, start_node=None):
    """Order facilities by geographic proximity"""
    
    # Start dari starting point
    if start_node:
        start_idx = df.index[df["nearest_node"] == start_node][0]
    else:
        start_idx = np.argmin(lons)  # Westernmost facility
    
    visited = [False] * n
    order = [start_idx]
    visited[start_idx] = True
    
    # Greedy nearest-neighbor
    for _ in range(n-1):
        cur = order[-1]
        # Find nearest unvisited facility
        nearest = argmin(distance(cur, all_unvisited))
        order.append(nearest)
        visited[nearest] = True
    
    return df.iloc[order]  # Facilities in NN order
```

**Example:**
```
Facilities (dengan coordinates):
  Polda Jatim:      (-7.234, 112.745)
  Police Station A: (-7.245, 112.760)
  Police Station B: (-7.210, 112.755)
  Fire Station:     (-7.235, 112.730)

NN Order starting dari Polda:
  1. Polda Jatim      (start)
  2. Police Station A (nearest: 2.2 km)
  3. Police Station B (nearest: 3.8 km)
  4. Fire Station     (nearest: 1.5 km)
  → Return ke Polda (round trip)

Result: Geographic circuit yang masuk akal!
```

---

## 4. BenchmarkRunner - Main Orchestrator

### Struktur

```python
class BenchmarkRunner:
    def __init__(self, registry: AlgorithmRegistry, log_dir: Path = None):
        self.registry = registry           # List of algorithms
        self.log_dir = log_dir             # Where to save evolution logs
        self.scenarios: List[Scenario] = []
        self.results: List[RouteResult] = []
    
    def add_scenario(self, scenario: Scenario):
        """Add scenario to benchmark"""
        self.scenarios.append(scenario)
    
    def run(self, G: nx.MultiDiGraph, parallel_legs: bool = False) -> pd.DataFrame:
        """Run all algorithms on all scenarios"""
        ...
    
    def summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute benchmark summary statistics"""
        ...
```

### The run() Method - Main Benchmark Loop

```python
def run(self, G, parallel_legs=False):
    """
    Main benchmark loop:
      For each scenario:
        For each algorithm:
          Run algorithm → RouteResult
          Log evolution (if GA)
    
    Return: DataFrame dengan semua hasil
    """
    self.results = []
    algos = self.registry.all()  # Get all algorithms
    n_workers = min(len(algos), cpu_count())
    
    mode = "algo+leg" if parallel_legs else "algo"
    log.info(f"Benchmark: {len(algos)} algorithms × {len(self.scenarios)} "
             f"scenarios (mode={mode})")
    
    # ─── LOOP: Setiap scenario ────────────────────
    for scenario in self.scenarios:
        route_label = " -> ".join(scenario.label_sequence)
        log.info(f"  Scenario [{scenario.name}]: {route_label}")
        
        # ─── RUN: Semua algorithms pada scenario ini ─
        if parallel_legs:
            # Parallel: every (algo × leg) is independent
            results_by_name = self._run_scenario_parallel_legs(
                G, scenario, algos, n_leg_workers
            )
        else:
            # Parallel: every algorithm in own process
            results_by_name = {}
            g_bytes = pickle.dumps(G)
            tasks = [(algo, scenario) for algo in algos]
            with Pool(processes=n_workers) as pool:
                for algo, result in zip(algos, pool.map(_algo_task, tasks)):
                    results_by_name[algo.name] = result
        
        # ─── LOG: Print results ────────────────────
        for algo in algos:
            result = results_by_name[algo.name]
            self.results.append(result)
            
            status = "OK   " if result.found else "FAIL "
            log.info(f"    {status} [{algo.name:<22}]  "
                     f"time={result.total_time_s/60:5.1f}min  "
                     f"dist={result.total_distance_m/1000:5.2f}km  "
                     f"cpu={result.computation_ms:6.1f}ms")
            
            # Write evolution log (only for GA)
            if self.log_dir and "gen_history" in result.metadata:
                _write_evolution_log(result, self.log_dir)
    
    return self._to_dataframe()
```

**Alur visualisasi:**
```
┌─────────────────────────────────────────┐
│ Benchmark Start                          │
│ Algorithms: GA, Dijkstra, A*, ACO, SA   │
│ Scenarios: emergency_patrol, terminal    │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ Scenario 1: emergency_patrol_circuit    │
│ Route: Polda → 50 stations → Polda      │
└─────────────────────────────────────────┘
              ↓
    ┌───────────┬───────────┬────────────┐
    ↓           ↓           ↓            ↓
  GA_result  Dijkstra_  A*_result  ACO_result
  (find)     result     (find)     (find)
  Time: 310   (find)     Time: 295  Time: 304
  Min         Time: 290  Min        Min
    ↓           ↓           ↓            ↓
    └───────────┴───────────┴────────────┘
              ↓
    Log: gen_history for GA
    ├─ Gen 1:  300 min, visit=[...]
    ├─ Gen 2:  295 min, visit=[...]
    ...
    └─ Gen 30: 310 min, visit=[...]
              ↓
┌─────────────────────────────────────────┐
│ Scenario 2: terminal_circuit             │
│ Route: Terminal A → 20 terminals → A    │
└─────────────────────────────────────────┘
              ↓
    ┌───────────┬───────────┬────────────┐
    ↓           ↓           ↓            ↓
  GA_result  Dijkstra_  A*_result  ACO_result
  (find)     result     (find)     (find)
    ↓           ↓           ↓            ↓
    └───────────┴───────────┴────────────┘
              ↓
┌─────────────────────────────────────────┐
│ All Results Aggregated                   │
│ self.results = [                         │
│   GA_emg, Dijkstra_emg, A*_emg, ...,    │
│   GA_term, Dijkstra_term, A*_term, ..., │
│ ]                                        │
└─────────────────────────────────────────┘
              ↓
        Return: DataFrame
```

---

## 5. Running Algorithm on Scenario - Core Method

### _run_algorithm_on_scenario()

```python
@staticmethod
def _run_algorithm_on_scenario(algo, G, scenario):
    """
    Run ONE algorithm on ONE scenario.
    Handle 2 cases:
    1. Multi-stop with _route_multi_stop (GA, ACO, etc)
    2. Multi-stop without _route_multi_stop (simple routing per leg)
    """
    nodes = scenario.node_sequence
    labels = scenario.label_sequence
    
    # ─── CASE 1: Algorithm punya _route_multi_stop ────
    if scenario.is_multi_stop and hasattr(algo, "_route_multi_stop"):
        # GA, ACO, SA, Christofides decide visit order themselves
        return algo._route_multi_stop(
            G,
            nodes,
            scenario.name,
            source_node=scenario.source_node,
            target_node=scenario.target_node,
            round_trip=scenario.round_trip,
        )
```

**Decision Point: hasattr(algo, "_route_multi_stop")**

```
Algorithms dengan _route_multi_stop:
  ✓ GeneticAlgorithm
  ✓ AntColonyElitePro
  ✓ GeraldSimulatedAnnealing
  ✓ ParticleSwarmRouting
  ✓ ChristofidesAlgorithm
  ✓ SimulatedAnnealingAlgorithm

Algorithms tanpa _route_multi_stop:
  ✗ DijkstraTime
  ✗ DijkstraDistance
  ✗ AStarTime
  ✗ AStarDistance

Kenapa?
- Algorithms WITH: dapat optimize visit order (evolutionary/heuristic)
- Algorithms WITHOUT: hanya point-to-point shortest path

Untuk yang WITHOUT, benchmark akan:
1. Compute optimal visit order (exhaustive atau greedy)
2. Route per leg sequential
3. Combine results
```

### CASE 2: Multi-Stop Without _route_multi_stop

```python
# Compute optimal visit order (if scenario.optimize_order)
if scenario.optimize_order:
    ordered_nodes, order_objective, order_score = \
        BenchmarkRunner._best_visit_order(G, algo, nodes, scenario.round_trip)
    nodes = ordered_nodes
    labels = [labels[i] for i in order_idx]

# Route setiap leg
full_route = []
for src, dst in zip(nodes[:-1], nodes[1:]):
    leg_name = f"{scenario.name}_leg_{idx}"
    result = algo.safe_run(G, src, dst, leg_name)  # Dijkstra/A* run
    
    if not result.found:
        return RouteResult.failure(...)
    
    if not full_route:
        full_route.extend(result.route)
    else:
        full_route.extend(result.route[1:])  # Skip start node (duplicate)

# Round trip?
if scenario.round_trip:
    ret = algo.safe_run(G, nodes[-1], nodes[0], "return_leg")
    full_route.extend(ret.route[1:])
```

**Contoh untuk Dijkstra:**
```
Scenario: 4 nodes [A, B, C, D], round_trip=True

Step 1: Compute order (greedy or exhaustive)
        order = [A, C, B, D]  (optimal order)

Step 2: Route legs
        Leg 1: A → C (Dijkstra)
               route = [A, x, y, C]
               full_route = [A, x, y, C]
        
        Leg 2: C → B (Dijkstra)
               route = [C, z, B]
               full_route.extend([z, B]) = [A, x, y, C, z, B]
        
        Leg 3: B → D (Dijkstra)
               route = [B, w, D]
               full_route.extend([w, D]) = [A, x, y, C, z, B, w, D]

Step 3: Round trip
        Return: D → A (Dijkstra)
                route = [D, v, A]
                full_route.extend([v, A]) = [A, x, y, C, z, B, w, D, v, A]

Result: Full circuit route!
```

---

## 6. Evolution Log Writing

### _write_evolution_log()

```python
def _write_evolution_log(result: RouteResult, log_dir: Path):
    """
    Write per-generation stats untuk GA algorithms.
    
    Only called if result.metadata punya "gen_history".
    Output: logs/evolution_<algo>_<scenario>.txt
    """
    history = result.metadata.get("gen_history")
    if not history:
        return  # Only GA results have gen_history
    
    fname = log_dir / f"evolution_{result.algorithm_name}_{result.scenario_name}.txt"
    
    with open(fname, "w") as f:
        # Header
        f.write(f"GA Evolution Log\n")
        f.write(f"Algorithm: {result.algorithm_name}\n")
        f.write(f"Scenario:  {result.scenario_name}\n")
        f.write(f"Config:    pop={m['population']}  gen={m['generations']}  "
                f"xover={m['crossover_rate']}  mut={m['mutation_rate']}\n")
        f.write(f"CPU time:  {result.computation_ms:.1f} ms\n\n")
        
        # Per-generation
        for frame in history:
            gen = frame["gen"]
            t = frame["min"]  # Best fitness di gen ini
            dist = frame.get("dist")
            impr = (first_min - t) / first_min * 100  # Improvement %
            delta = prev - t  # Improvement vs prev gen
            
            tag = "initial" if gen == 1 else ""
            if delta > 1e-6:
                tag = f"IMPROVED -{delta:.3f} min"
            if gen == len(history):
                tag = (tag + " [FINAL]").strip() if tag else "[FINAL]"
            
            f.write(f"Gen {gen:>3}  |  {t:.4f} min  |  {impr:.2f}% improved  |  {tag}\n")
            
            if frame.get("streets"):
                f.write(f"  Route: {' -> '.join(frame['streets'])}\n")
            
            prev = t
        
        # Summary
        f.write(f"\nSUMMARY\n")
        f.write(f"  Gen 1: {first_min:.4f} min\n")
        f.write(f"  Final: {final_min:.4f} min\n")
        f.write(f"  Total: {total_impr:.2f}% improvement\n")
```

**Output example:**
```
GA Evolution Log
Generated: 2026-04-29 14:32:00
============================================================
Algorithm: ga
Scenario:  emergency_patrol_circuit
Config:    pop=50  gen=600  xover=0.85  mut=0.9
CPU time:  12345.6 ms
============================================================

Gen   1  |  618.4000 min  |  0.00% improved  |  initial
Gen   2  |  612.3000 min  |  0.98% improved  |  IMPROVED -6.100 min
Gen   3  |  610.5000 min  |  1.28% improved  |  IMPROVED -1.800 min
...
Gen  30  |  310.2000 min  |  49.84% improved  |  FINAL

============================================================
SUMMARY
  Gen 1 best:  618.4000 min
  Final best:  310.2000 min
  Total impr:  49.84%
  # improved:  28 generations
```

---

## 7. Result Aggregation - _to_dataframe()

```python
def _to_dataframe(self) -> pd.DataFrame:
    """Convert all results to DataFrame"""
    rows = []
    for r in self.results:
        row = {
            "scenario":       r.scenario_name,
            "algorithm":      r.algorithm_name,
            "found":          r.found,
            "travel_time_s":  r.total_time_s,
            "travel_time_min": r.total_time_s / 60,
            "distance_m":     r.total_distance_m,
            "distance_km":    r.total_distance_m / 1000,
            "nodes_in_route": r.nodes_in_route,
            "computation_ms": r.computation_ms,
            "error":          r.error,
        }
        # Add metadata columns
        row.update({f"meta_{k}": v for k, v in r.metadata.items()})
        rows.append(row)
    
    return pd.DataFrame(rows)
```

**Result DataFrame:**
```
   scenario                algorithm  found  travel_time_min  distance_km  computation_ms
0  emergency_patrol_circuit  ga              True           310.2          245.3          12345
1  emergency_patrol_circuit  dijkstra_time  True           320.1          255.2          8900
2  emergency_patrol_circuit  astar_time    True           318.5          252.1          9100
3  emergency_patrol_circuit  aco            True           303.8          238.5          14200
4  terminal_circuit          ga              True           85.5           67.2          3200
5  terminal_circuit          dijkstra_time  True           92.3           72.5          2100
...

Metadata columns:
   meta_population  meta_generations  meta_crossover_rate  meta_mutation_rate  ...
0  50              600               0.85                 0.9
1  None            None              None                None
2  None            None              None                None
3  50              600               0.85                 0.85
4  50              600               0.85                 0.9
...
```

---

## 8. Summary Statistics

```python
def summary(self, df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-algorithm summary"""
    ok = df[df["found"] == True]
    
    summary = ok.groupby("algorithm").agg(
        solved=("scenario", "count"),
        avg_time_min=("travel_time_min", "mean"),
        avg_dist_km=("distance_km", "mean"),
        avg_cpu_ms=("computation_ms", "mean"),
        best_time_min=("travel_time_min", "min"),
        worst_time_min=("travel_time_min", "max"),
    ).round(3)
    
    return summary
```

**Output:**
```
BENCHMARK SUMMARY
==============================================================
                    solved  avg_time_min  avg_dist_km  best_time_min
algorithm
ga                       2        195.85       156.25           85.5
dijkstra_time            2        206.20       163.85           92.3
astar_time               2        201.45       160.35           88.9
aco                      2        189.30       150.30           85.2
simulated_annealing      2        192.15       154.60           86.7
christofides             2        198.40       158.20           91.5
```

---

## 9. Complete Benchmark Flow Diagram

```
┌────────────────────────────────────────────────────────┐
│ main.py: python main.py compare                        │
└────────────────────────────────────────────────────────┘
              ↓
┌────────────────────────────────────────────────────────┐
│ LOAD DATA                                               │
│  - G = NetworkX graph (road_network.graphml)          │
│  - fac = GeoDataFrame (facilities_with_network.csv)   │
└────────────────────────────────────────────────────────┘
              ↓
┌────────────────────────────────────────────────────────┐
│ BUILD REGISTRY                                          │
│  - Create AlgorithmRegistry()                          │
│  - Register 10 algorithms (GA, Dijkstra, A*, etc)     │
└────────────────────────────────────────────────────────┘
              ↓
┌────────────────────────────────────────────────────────┐
│ BUILD SCENARIOS                                         │
│  - build_category_scenarios(G, fac)                    │
│  → [emergency_patrol_circuit, terminal_circuit]        │
└────────────────────────────────────────────────────────┘
              ↓
┌────────────────────────────────────────────────────────┐
│ CREATE BENCHMARK RUNNER                                │
│  - BenchmarkRunner(registry, log_dir)                  │
│  - Add 2 scenarios                                      │
└────────────────────────────────────────────────────────┘
              ↓
┌────────────────────────────────────────────────────────┐
│ RUN BENCHMARK                                           │
│  runner.run(G, parallel_legs=True)                     │
│                                                        │
│  For each scenario:                                    │
│    For each algorithm:                                 │
│      Run algorithm → RouteResult                       │
│      Write evolution log (if GA)                       │
└────────────────────────────────────────────────────────┘
              ↓
    ┌─────────────────────────────┐
    │ Scenario 1: Emergency       │
    │ (2 runs shown for clarity)  │
    ├─────────────────────────────┤
    │                             │
    │ Algorithm 1: GA             │
    │ ├─ Precompute costs         │
    │ ├─ Gen 1: cost=618          │
    │ ├─ Gen 2: cost=612 (improve)│
    │ ├─ ...                      │
    │ ├─ Gen 30: cost=310         │
    │ └─ Write evolution log ✓    │
    │ Result: OK 310.2 min        │
    │                             │
    │ Algorithm 2: Dijkstra       │
    │ ├─ Run shortest path        │
    │ ├─ No evolution (no gen)    │
    │ Result: OK 320.1 min        │
    │                             │
    └─────────────────────────────┘
              ↓
    ┌─────────────────────────────┐
    │ Scenario 2: Terminal        │
    │ (GA & Dijkstra)             │
    │                             │
    │ GA Result: OK 85.5 min      │
    │ Dijkstra:  OK 92.3 min      │
    │                             │
    └─────────────────────────────┘
              ↓
┌────────────────────────────────────────────────────────┐
│ AGGREGATE RESULTS                                       │
│  - Combine all RouteResults                            │
│  - Convert to DataFrame                                │
│  - 10 algorithms × 2 scenarios = 20 rows               │
└────────────────────────────────────────────────────────┘
              ↓
┌────────────────────────────────────────────────────────┐
│ OUTPUT RESULTS                                          │
│  - CSV: data/comparison_results.csv                    │
│  - Charts: data/comparison_chart.png                   │
│  - Maps: data/comparison_map_*.html                    │
│  - Logs: logs/evolution_*.txt (for GA only)           │
└────────────────────────────────────────────────────────┘
              ↓
┌────────────────────────────────────────────────────────┐
│ SUMMARY STATISTICS                                      │
│  Per-algorithm average metrics:                        │
│    - avg time, avg distance, avg CPU ms               │
│    - best/worst times                                  │
└────────────────────────────────────────────────────────┘
```

---

## 10. Parallel Execution Strategy

### Two Modes

```
Mode 1: Parallel Algorithms (default, parallel_legs=False)
┌────────────────────────────────┐
│ Scenario 1                     │
├────────────────────────────────┤
│  Process 1: Algorithm 1        │
│  Process 2: Algorithm 2        │
│  Process 3: Algorithm 3        │
│  Process 4: Algorithm 4        │  ← Run in parallel
│  ...                           │
│  Process N: Algorithm N        │
└────────────────────────────────┘
     All algorithms run simultaneously on same scenario

Mode 2: Parallel Legs (parallel_legs=True)
┌────────────────────────────────┐
│ Scenario 1 with 50 facilities  │
├────────────────────────────────┤
│ Algorithm 1:                   │
│   Process 1: Leg 1 → 2         │
│   Process 2: Leg 2 → 3         │
│   Process 3: Leg 3 → 4         │
│   ...                          │  ← Run in parallel
│   Process 49: Leg 49 → 50      │
│                                │
│ Algorithm 2:                   │
│   Process 50: Leg 1 → 2        │
│   Process 51: Leg 2 → 3        │
│   ...                          │
│                                │
│ All (N_ALGOS × N_LEGS) tasks   │
└────────────────────────────────┘
```

### Code Implementation

```python
def run(self, G, parallel_legs=False):
    ...
    
    if parallel_legs:
        # Flat pool: every (algo × leg) is task
        results_by_name = self._run_scenario_parallel_legs(
            G, scenario, algos, n_leg_workers
        )
    else:
        # Simple pool: every algorithm is task
        g_bytes = pickle.dumps(G)
        tasks = [(algo, scenario) for algo in algos]
        with Pool(processes=n_workers) as pool:
            for algo, result in zip(algos, pool.map(_algo_task, tasks)):
                results_by_name[algo.name] = result
```

### When to Use Which Mode?

```
Use Mode 1 (parallel algorithms):
  ✓ Few facilities (≤5 nodes)
  ✓ Algorithms are CPU-intensive
  ✓ Want simplicity
  
Use Mode 2 (parallel legs):
  ✓ Many facilities (>20 nodes)
  ✓ Multi-leg routing (sequential legs)
  ✓ Want max CPU utilization
  ✓ Legs are independent → great parallelism
```

---

## Summary: Key Decision Points

```
1. HashasOne  hasattr(algo, "_route_multi_stop")?
   ├─ YES → algo decides visit order (GA, ACO, etc)
   └─ NO  → benchmark computes order, routes per leg

2. optimize_order = True?
   ├─ YES → compute best visit order (exhaustive ≤9, greedy >9)
   └─ NO  → use fixed order

3. round_trip = True?
   ├─ YES → add return leg (last → first)
   └─ NO  → stop at last node

4. parallel_legs = True?
   ├─ YES → parallel (algo × leg) tasks
   └─ NO  → parallel algorithms only

5. gen_history in result.metadata?
   ├─ YES → write evolution log
   └─ NO  → skip evolution log (non-GA algorithms)
```

---

## Complete Code Flow

```
main.py compare
  ↓
run_platform(G):
  ├─ Load G from graphml
  ├─ Load fac from csv
  ├─ Create registry
  │   ├─ register(DijkstraTime)
  │   ├─ register(DijkstraDistance)
  │   ├─ register(AStarTime)
  │   ├─ register(AStarDistance)
  │   ├─ register(GeneticAlgorithm) ← Our focus!
  │   ├─ register(SimulatedAnnealingAlgorithm)
  │   ├─ register(ChristofidesAlgorithm)
  │   ├─ register(AntColonyElitePro)
  │   ├─ register(GeraldSimulatedAnnealing)
  │   └─ register(ParticleSwarmRouting)
  │
  ├─ Build scenarios
  │   ├─ build_category_scenarios(G, fac)
  │   │   ├─ emergency_scenario (50 stations)
  │   │   └─ terminal_scenario (20 terminals)
  │
  ├─ Create runner
  │   ├─ BenchmarkRunner(registry, log_dir)
  │   ├─ add_scenario(emergency_scenario)
  │   └─ add_scenario(terminal_scenario)
  │
  ├─ Run benchmark
  │   └─ runner.run(G, parallel_legs=True)
  │       ├─ For emergency_scenario:
  │       │   ├─ GA._route_multi_stop() → gen_history
  │       │   ├─ Dijkstra.safe_run() per leg
  │       │   ├─ A*.safe_run() per leg
  │       │   ├─ ACO._route_multi_stop()
  │       │   └─ ... (other algos)
  │       │
  │       └─ For terminal_scenario:
  │           ├─ GA._route_multi_stop()
  │           ├─ Dijkstra per leg
  │           ├─ ... (other algos)
  │
  ├─ Write evolution logs
  │   ├─ logs/evolution_ga_emergency_patrol_circuit.txt
  │   └─ logs/evolution_ga_terminal_circuit.txt
  │
  ├─ Aggregate to DataFrame
  │   └─ 10 algos × 2 scenarios = 20 rows
  │
  ├─ Save CSV
  │   └─ data/comparison_results.csv
  │
  ├─ Generate charts & maps
  │   ├─ data/comparison_chart.png
  │   ├─ data/comparison_map_emergency.html
  │   └─ data/comparison_map_terminal.html
  │
  └─ Compute summary
      └─ Per-algorithm statistics
```
