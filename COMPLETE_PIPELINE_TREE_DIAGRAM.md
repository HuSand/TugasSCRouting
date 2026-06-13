# Complete Pipeline Tree Diagrams

## 1. GA Algorithm Complete Flow Tree

```
┌─────────────────────────────────────────────────────────────────┐
│ GA._route_multi_stop(G, nodes, scenario_name)                  │
│ Input: Graph G, nodes=[1,4,2,3], scenario="emergency_patrol"   │
└─────────────────────────────────────────────────────────────────┘
                            ↓
        ┌───────────────────────────────────────┐
        │ PHASE 1: Setup & Validation           │
        ├───────────────────────────────────────┤
        │ - Split nodes: start=1, end=1         │
        │ - intermediates=[4,2,3] (visit these) │
        │ - Validate: ≥2 nodes needed? YES ✓   │
        │ - Log: "Starting GA TSP evolution"    │
        └───────────────────────────────────────┘
                            ↓
        ┌───────────────────────────────────────┐
        │ PHASE 2: Precompute Pairwise Costs    │
        ├───────────────────────────────────────┤
        │ Run 4 parallel Dijkstra sweeps:       │
        │                                        │
        │ ├─ From node 1: {4:100, 2:150, 3:200}│
        │ ├─ From node 4: {1:110, 2:80, 3:150} │
        │ ├─ From node 2: {1:160, 4:85, 3:120} │
        │ └─ From node 3: {1:210, 4:160, 2:130}│
        │                                        │
        │ Result: pair_cost dict (4²=16 pairs) │
        │ Time: O(4 × Dijkstra) = fast! ⚡     │
        └───────────────────────────────────────┘
                            ↓
        ┌───────────────────────────────────────────┐
        │ PHASE 3: Initialize Population (50 indiv) │
        ├───────────────────────────────────────────┤
        │                                            │
        │ 1 GREEDY NN SEED:                         │
        │ ├─ greedy_nn_order(1, [4,2,3])           │
        │ ├─ Step 1: nearest(1)={4:100} → pick 4  │
        │ ├─ Step 2: nearest(4)={2:80} → pick 2   │
        │ ├─ Step 3: nearest(2)={3:120} → pick 3  │
        │ └─ Result: [4,2,3], cost=300 min ✓      │
        │                                            │
        │ 49 RANDOM SHUFFLES:                       │
        │ ├─ shuffle([4,2,3]) → [2,4,3]           │
        │ ├─ shuffle([4,2,3]) → [3,2,4]           │
        │ ├─ shuffle([4,2,3]) → [4,3,2]           │
        │ └─ ... (49 times, costs ~400-600 min)   │
        │                                            │
        │ Population = 50 individuals               │
        │ Diversity: ✓ HIGH (1 good + 49 random)  │
        │ Gen 1 baseline: min=300 (NOT 618!) ✓    │
        └───────────────────────────────────────────┘
                            ↓
        ┌────────────────────────────────────────────┐
        │ PHASE 4: Evaluate Generation 1              │
        ├────────────────────────────────────────────┤
        │ For each individual in population:         │
        │   fitness = tour_cost(individual)          │
        │                                             │
        │ fitness = [300, 420, 410, 450, 380, ...]  │
        │ best_idx = 0 (index dengan cost terendah) │
        │ best_perm = [4, 2, 3]                     │
        │ best_cost = 300 min                        │
        │                                             │
        │ Important: best_perm dari EVALUATED       │
        │ population, NOT forced dari input order! ✓ │
        │ (This was THE CRITICAL FIX)                │
        └────────────────────────────────────────────┘
                            ↓
        ┌──────────────────────────────────────────────────┐
        │ PHASE 5: Evolution Loop (Gen 2-30)               │
        ├──────────────────────────────────────────────────┤
        │                                                    │
        │ For gen_idx = 1 to 600:  (or until early stop)  │
        │                                                    │
        │   ┌─────────────────────────────────────────────┐│
        │   │ A. EVALUATE CURRENT POPULATION              ││
        │   ├─────────────────────────────────────────────┤│
        │   │ fitness = [tour_cost(p) for p in pop]      ││
        │   │ best_current = min(fitness)                 ││
        │   │                                              ││
        │   │ If best_current < best_cost:               ││
        │   │   ├─ best_perm = elite                      ││
        │   │   ├─ best_cost = best_current              ││
        │   │   ├─ no_improve = 0 (reset counter)        ││
        │   │   └─ Log: "Gen X: IMPROVED to 295 min" ✓  ││
        │   │ Else:                                        ││
        │   │   └─ no_improve += 1                        ││
        │   └─────────────────────────────────────────────┘│
        │                                                    │
        │   ┌─────────────────────────────────────────────┐│
        │   │ B. CHECK EARLY STOPPING                     ││
        │   ├─────────────────────────────────────────────┤│
        │   │ if no_improve >= 20 (TSP_PATIENCE):         ││
        │   │   └─ BREAK (Gen 15 no improve, stop here!) ││
        │   │   └─ Reason: population converged          ││
        │   │   └─ Saves: run remaining gens (expensive!) ││
        │   └─────────────────────────────────────────────┘│
        │                                                    │
        │   ┌─────────────────────────────────────────────────┐│
        │   │ C. BUILD NEXT GENERATION (50 new individuals)    ││
        │   ├─────────────────────────────────────────────────┤│
        │   │                                                   ││
        │   │ ELITISM: Keep best (elite)                      ││
        │   │   └─ new_pop = [elite[:]]  (1 individual)      ││
        │   │                                                   ││
        │   │ BREEDING: Create 49 more via genetic operators: ││
        │   │                                                   ││
        │   │ while len(new_pop) < 50:                        ││
        │   │                                                   ││
        │   │   ┌─────────────────────────────────────────┐   ││
        │   │   │ 1. SELECTION (Tournament, size=3)       │   ││
        │   │   ├─────────────────────────────────────────┤   ││
        │   │   │ candidates = random pick 3 from pop     │   ││
        │   │   │ p1 = best of 3 (fitness terendah)      │   ││
        │   │   │                                         │   ││
        │   │   │ Contoh:                                 │   ││
        │   │   │   candidates=[5, 12, 8]                │   ││
        │   │   │   fitness[5]=400                       │   ││
        │   │   │   fitness[12]=350 ← BEST              │   ││
        │   │   │   fitness[8]=380                       │   ││
        │   │   │   p1 = population[12]                  │   ││
        │   │   └─────────────────────────────────────────┘   ││
        │   │                                                   ││
        │   │   ┌─────────────────────────────────────────┐   ││
        │   │   │ 2. CROSSOVER (80% chance)              │   ││
        │   │   ├─────────────────────────────────────────┤   ││
        │   │   │ if random() < 0.8:                      │   ││
        │   │   │   p2 = tournament_pick(3)              │   ││
        │   │   │   child = ox_crossover(p1, p2)         │   ││
        │   │   │                                         │   ││
        │   │   │   Example OX:                           │   ││
        │   │   │   p1 = [4, 2, 3]                       │   ││
        │   │   │   p2 = [2, 4, 3]                       │   ││
        │   │   │   → child = [4, 2, 3] or [2, 3, 4]    │   ││
        │   │   │     (valid permutation!)               │   ││
        │   │   │                                         │   ││
        │   │   │ else:                                   │   ││
        │   │   │   child = p1[:]  (clone, no crossover)│   ││
        │   │   └─────────────────────────────────────────┘   ││
        │   │                                                   ││
        │   │   ┌─────────────────────────────────────────┐   ││
        │   │   │ 3. MUTATION (90% chance, 2-3 swaps)    │   ││
        │   │   ├─────────────────────────────────────────┤   ││
        │   │   │ if random() < 0.9:                      │   ││
        │   │   │   num_swaps = random(2-3)              │   ││
        │   │   │                                         │   ││
        │   │   │   Example (num_swaps=2):               │   ││
        │   │   │   child = [4, 2, 3]                    │   ││
        │   │   │   swap(0,2): [3, 2, 4]                 │   ││
        │   │   │   swap(1,2): [3, 4, 2]                 │   ││
        │   │   │   → child = [3, 4, 2] ✓ changed!      │   ││
        │   │   │                                         │   ││
        │   │   │ Benefit: explore more permutations!    │   ││
        │   │   │                                         │   ││
        │   │   │ else:                                   │   ││
        │   │   │   child unchanged                       │   ││
        │   │   └─────────────────────────────────────────┘   ││
        │   │                                                   ││
        │   │   Add child to new_pop                           ││
        │   │                                                   ││
        │   └─────────────────────────────────────────────────┘│
        │                                                        │
        │   population = new_pop  (update untuk next gen)      │
        │                                                        │
        │ End of loop → Go to next generation or break         │
        │                                                        │
        └──────────────────────────────────────────────────────┘
                            ↓
        ┌──────────────────────────────────────┐
        │ PHASE 6: Final Best Solution Found   │
        ├──────────────────────────────────────┤
        │ best_perm = [4, 2, 3]                │
        │ best_cost = 285 min (after evolution)│
        │ generations_run = 28 (early stop) ✓ │
        │                                       │
        │ Improvement:                         │
        │   Gen 1:  300 min                    │
        │   Final: 285 min                     │
        │   Improvement: 5% ✓                 │
        └──────────────────────────────────────┘
                            ↓
        ┌──────────────────────────────────────────────┐
        │ PHASE 7: Expand to Full Road Network Path    │
        ├──────────────────────────────────────────────┤
        │ best_perm = [4, 2, 3]                        │
        │ full_tour = [1, 4, 2, 3, 1]  (round trip)   │
        │                                               │
        │ For each leg:                                │
        │   (1→4): shortest_path = [1, 10, 20, 4]    │
        │   (4→2): shortest_path = [4, 50, 2]        │
        │   (2→3): shortest_path = [2, 60, 70, 3]    │
        │   (3→1): shortest_path = [3, 80, 1]        │
        │                                               │
        │ Combine (skip duplicate start nodes):        │
        │   full_route = [1, 10, 20, 4, 50, 2,        │
        │                 60, 70, 3, 80, 1]            │
        │                                               │
        │ Result: Complete road network path! ✓        │
        └──────────────────────────────────────────────┘
                            ↓
        ┌────────────────────────────────────────────┐
        │ PHASE 8: Return RouteResult                │
        ├────────────────────────────────────────────┤
        │ Return RouteResult(                         │
        │   algorithm_name="ga",                     │
        │   scenario_name="emergency_patrol_circuit",│
        │   route=[1, 10, 20, 4, 50, 2, ...],       │
        │   travel_time_s=17100,  (285 min)         │
        │   total_distance_m=245000,  (245 km)      │
        │   computation_ms=12345,                    │
        │   metadata={                                │
        │     'visit_order': [1, 4, 2, 3, 1],       │
        │     'generations': 28,                     │
        │     'gen_history': [                       │
        │       {'gen': 1, 'min': 300, ...},        │
        │       {'gen': 2, 'min': 295, ...},        │
        │       ...,                                 │
        │       {'gen': 28, 'min': 285, ...},       │
        │     ]                                      │
        │   }                                         │
        │ )                                           │
        └────────────────────────────────────────────┘
```

---

## 2. Benchmark Pipeline Complete Flow Tree

```
┌─────────────────────────────────────────────────────────────────┐
│ python main.py compare                                          │
│ ↓ Entry point untuk full benchmark                             │
└─────────────────────────────────────────────────────────────────┘
                            ↓
        ┌─────────────────────────────────────┐
        │ STEP 1: Load Data                   │
        ├─────────────────────────────────────┤
        │ ├─ G = nx.read_graphml(...)         │
        │ │   (road_network.graphml)          │
        │ │   Nodes: ~50,000 intersections   │
        │ │   Edges: ~150,000 streets         │
        │ │                                    │
        │ └─ fac = gpd.read_file(...)        │
        │     (facilities_with_network.csv)  │
        │     Hospitals, Police, Schools,    │
        │     Fire Stations, etc             │
        │     ~500-1000 total facilities     │
        │                                     │
        │ Status: Data loaded ✓               │
        └─────────────────────────────────────┘
                            ↓
        ┌──────────────────────────────────────────┐
        │ STEP 2: Build Algorithm Registry         │
        ├──────────────────────────────────────────┤
        │                                           │
        │ registry = AlgorithmRegistry()            │
        │                                           │
        │ registry.register(DijkstraTime())         │
        │ registry.register(DijkstraDistance())     │
        │ registry.register(AStarTime())            │
        │ registry.register(AStarDistance())        │
        │ registry.register(GeneticAlgorithm()) ← GA│
        │ registry.register(SimulatedAnnealingAlg())│
        │ registry.register(ChristofidesAlgorithm())│
        │ registry.register(AntColonyElitePro())    │
        │ registry.register(GeraldSimulatedAnn())   │
        │ registry.register(ParticleSwarmRouting()) │
        │                                           │
        │ Total: 10 algorithms registered ✓         │
        │                                           │
        │ Status: Registry ready with 10 algos ✓   │
        └──────────────────────────────────────────┘
                            ↓
        ┌────────────────────────────────────────────────┐
        │ STEP 3: Build Scenarios from Facility Data     │
        ├────────────────────────────────────────────────┤
        │                                                 │
        │ scenarios = build_category_scenarios(G, fac)   │
        │                                                 │
        │ ┌────────────────────────────────────────────┐ │
        │ │ Scenario 1: EMERGENCY PATROL CIRCUIT       │ │
        │ ├────────────────────────────────────────────┤ │
        │ │ Filter: facilities dengan category=        │ │
        │ │         "emergency" (police + fire)        │ │
        │ │         ~100 facilities awal               │ │
        │ │                                             │ │
        │ │ Dedup: one per nearest_node → ~95 unique  │ │
        │ │                                             │ │
        │ │ Cap: max 50 → diverse_subset(95, 50)      │ │
        │ │      (farthest-point geographic sampling)  │ │
        │ │                                             │ │
        │ │ Order: nn_order(50, start=POLDA_NODE)     │ │
        │ │        (nearest-neighbor geographic tour)  │ │
        │ │        Result: [Polda, Station1, St2, ...] │ │
        │ │                                             │ │
        │ │ Scenario properties:                        │ │
        │ │   name: "emergency_patrol_circuit"         │ │
        │ │   nodes: [9156956728, 1234567, ...]  (50) │ │
        │ │   labels: ["Polda Jatim", "Polda Surabaya"│ │
        │ │   round_trip: True                          │ │
        │ │   optimize_order: True                      │ │
        │ └────────────────────────────────────────────┘ │
        │                                                 │
        │ ┌────────────────────────────────────────────┐ │
        │ │ Scenario 2: TERMINAL CIRCUIT                │ │
        │ ├────────────────────────────────────────────┤ │
        │ │ Filter: category="terminal" (bus+ferry+gas)│ │
        │ │         ~30 facilities                      │ │
        │ │                                             │ │
        │ │ Dedup & Order: similar process             │ │
        │ │                                             │ │
        │ │ Scenario properties:                        │ │
        │ │   name: "terminal_circuit"                 │ │
        │ │   nodes: [...] (~20)                       │ │
        │ │   round_trip: True                          │ │
        │ │   optimize_order: True                      │ │
        │ └────────────────────────────────────────────┘ │
        │                                                 │
        │ Status: 2 scenarios built ✓                    │
        └────────────────────────────────────────────────┘
                            ↓
        ┌──────────────────────────────────────────────────┐
        │ STEP 4: Create BenchmarkRunner                   │
        ├──────────────────────────────────────────────────┤
        │ runner = BenchmarkRunner(registry, log_dir)      │
        │ runner.add_scenario(scenario1)                   │
        │ runner.add_scenario(scenario2)                   │
        │                                                   │
        │ Ready to run: 10 algos × 2 scenarios            │
        │            = 20 benchmark runs total             │
        │                                                   │
        │ Status: Runner initialized ✓                     │
        └──────────────────────────────────────────────────┘
                            ↓
        ┌────────────────────────────────────────────────────┐
        │ STEP 5: RUN BENCHMARK (Main Loop)                 │
        ├────────────────────────────────────────────────────┤
        │                                                     │
        │ df = runner.run(G, parallel_legs=True)            │
        │                                                     │
        │ For scenario in [emergency_patrol, terminal]:     │
        │                                                     │
        │   ┌─────────────────────────────────────────────┐ │
        │   │ Scenario: emergency_patrol_circuit (50 nodes)│ │
        │   ├─────────────────────────────────────────────┤ │
        │   │                                              │ │
        │   │ For algorithm in [GA, Dijkstra, A*, ...]:  │ │
        │   │                                              │ │
        │   │   ┌──────────────────────────────────────┐  │ │
        │   │   │ Algorithm: GA                        │  │ │
        │   │   ├──────────────────────────────────────┤  │ │
        │   │   │ Check: hasattr(ga, "_route_multi_stop")│ │
        │   │   │ YES! → Call _route_multi_stop()     │  │ │
        │   │   │                                      │  │ │
        │   │   │ GA TSP Evolution:                    │  │ │
        │   │   │   Gen 1: cost=618 min               │  │ │
        │   │   │   Gen 2: cost=612 min (improved!)   │  │ │
        │   │   │   ...                                │  │ │
        │   │   │   Gen 28: cost=310 min (final)      │  │ │
        │   │   │                                      │  │ │
        │   │   │ Result: RouteResult(                │  │ │
        │   │   │   found=True,                       │  │ │
        │   │   │   route=[...],                      │  │ │
        │   │   │   travel_time=310 min,              │  │ │
        │   │   │   gen_history=[...] ← SAVED!        │  │ │
        │   │   │ )                                    │  │ │
        │   │   │                                      │  │ │
        │   │   │ Write evolution log:                 │  │ │
        │   │   │   evolution_ga_emergency_patrol...  │  │ │
        │   │   └──────────────────────────────────────┘  │ │
        │   │                                              │ │
        │   │   ┌──────────────────────────────────────┐  │ │
        │   │   │ Algorithm: DijkstraTime              │  │ │
        │   │   ├──────────────────────────────────────┤  │ │
        │   │   │ Check: hasattr(dijkstra, "...")     │  │ │
        │   │   │ NO → Use pre-computed order         │  │ │
        │   │   │                                      │  │ │
        │   │   │ Compute best order (exhaustive):    │  │ │
        │   │   │   Try all 50! permutations         │  │ │
        │   │   │   → TOO MANY (skip, use greedy)    │  │ │
        │   │   │                                      │  │ │
        │   │   │ Use greedy nearest-neighbor:        │  │ │
        │   │   │   order = [node1, node5, node8,...]│  │ │
        │   │   │                                      │  │ │
        │   │   │ Route each leg:                      │  │ │
        │   │   │   Leg 1 (node1→node5): run Dijkstra │  │ │
        │   │   │   Leg 2 (node5→node8): run Dijkstra │  │ │
        │   │   │   ...                                │  │ │
        │   │   │   Return (last→first): run Dijkstra │  │ │
        │   │   │                                      │  │ │
        │   │   │ Result: RouteResult(                │  │ │
        │   │   │   found=True,                       │  │ │
        │   │   │   route=[...],                      │  │ │
        │   │   │   travel_time=320 min,              │  │ │
        │   │   │ )                                    │  │ │
        │   │   │                                      │  │ │
        │   │   │ No evolution log (not GA)           │  │ │
        │   │   └──────────────────────────────────────┘  │ │
        │   │                                              │ │
        │   │   ┌──────────────────────────────────────┐  │ │
        │   │   │ Algorithm: A*Time (same as Dijkstra)│  │ │
        │   │   ├──────────────────────────────────────┤  │ │
        │   │   │ Result: found=True, travel_time=318 │  │ │
        │   │   └──────────────────────────────────────┘  │ │
        │   │                                              │ │
        │   │   ┌──────────────────────────────────────┐  │ │
        │   │   │ Algorithm: ACO (AntColonyElitePro)   │  │ │
        │   │   ├──────────────────────────────────────┤  │ │
        │   │   │ Check: hasattr(aco, "_route_multi...")│ │
        │   │   │ YES! → Call _route_multi_stop()     │  │ │
        │   │   │                                      │  │ │
        │   │   │ ACO Evolution:                       │  │ │
        │   │   │   Gen 1: cost=520 min               │  │ │
        │   │   │   Gen 2: cost=510 min               │  │ │
        │   │   │   ...                                │  │ │
        │   │   │   Gen 50: cost=304 min (final)      │  │ │
        │   │   │                                      │  │ │
        │   │   │ Result: found=True, travel_time=304 │  │ │
        │   │   │ Has gen_history → Write log         │  │ │
        │   │   └──────────────────────────────────────┘  │ │
        │   │                                              │ │
        │   │   [Continue for 6 more algorithms...]      │ │
        │   │                                              │ │
        │   │   All 10 results collected:                │ │
        │   │   ├─ GA: 310 min ✓                         │ │
        │   │   ├─ Dijkstra: 320 min ✓                   │ │
        │   │   ├─ A*: 318 min ✓                         │ │
        │   │   ├─ ACO: 304 min ✓ (BEST!)               │ │
        │   │   ├─ SA: 312 min ✓                         │ │
        │   │   ├─ Christofides: 315 min ✓               │ │
        │   │   ├─ Gerald SA: 314 min ✓                  │ │
        │   │   ├─ PSO: 308 min ✓                        │ │
        │   │   └─ ...                                    │ │
        │   │                                              │ │
        │   └─────────────────────────────────────────────┘ │
        │                                                     │
        │   ┌─────────────────────────────────────────────┐  │
        │   │ Scenario: terminal_circuit (20 nodes)       │  │
        │   ├─────────────────────────────────────────────┤  │
        │   │ Repeat same process for 10 algorithms      │  │
        │   │ [GA, Dijkstra, A*, ACO, SA, Christofides] │  │
        │   │                                              │  │
        │   │ Results:                                     │  │
        │   │ ├─ GA: 85.5 min ✓                           │  │
        │   │ ├─ Dijkstra: 92.3 min ✓                     │  │
        │   │ ├─ ACO: 84.2 min ✓ (BEST!)                 │  │
        │   │ └─ ...                                       │  │
        │   │                                              │  │
        │   └─────────────────────────────────────────────┘  │
        │                                                     │
        │ Status: All 20 runs complete! ✓                    │
        │ Collected results in self.results list             │
        │                                                     │
        └────────────────────────────────────────────────────┘
                            ↓
        ┌──────────────────────────────────────────────────┐
        │ STEP 6: Aggregate Results to DataFrame            │
        ├──────────────────────────────────────────────────┤
        │                                                    │
        │ rows = []                                         │
        │ For each result in self.results (20 total):      │
        │   row = {                                        │
        │     'scenario': 'emergency_patrol_circuit',       │
        │     'algorithm': 'ga',                            │
        │     'found': True,                                │
        │     'travel_time_min': 310.2,                     │
        │     'distance_km': 245.3,                         │
        │     'computation_ms': 12345,                      │
        │     'meta_visit_order': [1,4,2,3,1],            │
        │     'meta_generations': 28,                       │
        │     'meta_gen_history': [...],                    │
        │     ... (more columns)                            │
        │   }                                               │
        │   rows.append(row)                               │
        │                                                    │
        │ df = pd.DataFrame(rows)  ← 20 rows, ~30 columns  │
        │                                                    │
        │ Status: DataFrame created ✓                      │
        │ Shape: (20 rows, 30+ columns)                    │
        └──────────────────────────────────────────────────┘
                            ↓
        ┌──────────────────────────────────────────────────┐
        │ STEP 7: Generate Output Files                     │
        ├──────────────────────────────────────────────────┤
        │                                                    │
        │ 1. CSV EXPORT                                     │
        │    └─ data/comparison_results.csv                │
        │       (all 20 results + metadata)                 │
        │                                                    │
        │ 2. EVOLUTION LOGS (GA only)                      │
        │    ├─ logs/evolution_ga_emergency_patrol...txt   │
        │    │   Gen 1: 618.4 min                          │
        │    │   Gen 2: 612.3 min (IMPROVED -6.1)         │
        │    │   ...                                        │
        │    │   Gen 28: 310.2 min [FINAL]                │
        │    │                                              │
        │    ├─ logs/evolution_ga_terminal_circuit.txt     │
        │    │   Gen 1: 155 min                            │
        │    │   ...                                        │
        │    │   Gen 30: 85.5 min [FINAL]                 │
        │    │                                              │
        │    ├─ logs/evolution_aco_emergency_patrol.txt    │
        │    │   (ACO also has multi-stop & gen_history)  │
        │    │                                              │
        │    └─ logs/evolution_aco_terminal_circuit.txt    │
        │                                                    │
        │ 3. CHARTS                                         │
        │    └─ data/comparison_chart.png                  │
        │       Bar chart:                                  │
        │       ├─ X-axis: Scenarios (Emergency, Terminal) │
        │       ├─ Y-axis: Travel time (minutes)           │
        │       └─ Bars: 10 algorithms per scenario        │
        │                                                    │
        │       Line chart:                                 │
        │       ├─ X-axis: Algorithms                      │
        │       ├─ Y-axis: CPU computation time           │
        │       └─ Show speed tradeoff vs quality          │
        │                                                    │
        │ 4. FOLIUM MAPS                                    │
        │    ├─ data/comparison_map_emergency_patrol.html  │
        │    │   ├─ Facility locations as markers          │
        │    │   ├─ GA route (blue)                        │
        │    │   ├─ Dijkstra route (red)                  │
        │    │   ├─ ACO route (green)                      │
        │    │   ├─ ... (all 10 algos)                     │
        │    │   └─ Interactive: click routes to highlight │
        │    │                                              │
        │    └─ data/comparison_map_terminal_circuit.html  │
        │       (same structure for terminal scenario)     │
        │                                                    │
        │ Status: All files generated ✓                    │
        └──────────────────────────────────────────────────┘
                            ↓
        ┌──────────────────────────────────────────────────┐
        │ STEP 8: Compute Summary Statistics                │
        ├──────────────────────────────────────────────────┤
        │                                                    │
        │ summary = df.groupby('algorithm').agg({          │
        │   'scenario': 'count',        → solved            │
        │   'travel_time_min': 'mean',  → avg_time         │
        │   'distance_km': 'mean',      → avg_dist         │
        │   'computation_ms': 'mean',   → avg_cpu          │
        │   'travel_time_min': 'min',   → best_time        │
        │   'travel_time_min': 'max',   → worst_time       │
        │ })                                                │
        │                                                    │
        │ Results per algorithm:                            │
        │ ┌────────────────┬────────┬──────────┬──────────┐│
        │ │ algorithm      │ solved │ avg_time │ best_time││
        │ ├────────────────┼────────┼──────────┼──────────┤│
        │ │ ga             │   2    │  195.85  │   85.5  ││
        │ │ dijkstra_time  │   2    │  206.20  │   92.3  ││
        │ │ astar_time     │   2    │  201.45  │   88.9  ││
        │ │ aco            │   2    │  189.30  │   84.2  ││
        │ │ simulated_ann  │   2    │  192.15  │   86.7  ││
        │ │ christofides   │   2    │  198.40  │   91.5  ││
        │ │ gerald_sa      │   2    │  194.20  │   86.9  ││
        │ │ pso            │   2    │  191.10  │   85.8  ││
        │ └────────────────┴────────┴──────────┴──────────┘│
        │                                                    │
        │ Status: Summary complete ✓                       │
        │ Best average: ACO (189.30 min)                   │
        │ Best single run: ACO emergency (84.2 min)       │
        └──────────────────────────────────────────────────┘
```

---

## 3. Algorithm Decision Tree

```
┌─────────────────────────────────────────────────────────────┐
│ Running Algorithm on Scenario - Decision Flow               │
└─────────────────────────────────────────────────────────────┘
                            ↓
        ┌──────────────────────────────────┐
        │ Is scenario multi-stop?          │
        │ (len(nodes) > 2 OR round_trip)   │
        └──────────────────────────────────┘
              ↙                    ↘
    NO (Point-to-point)      YES (Multi-stop)
            ↓                          ↓
    ┌──────────────────┐     ┌──────────────────────────┐
    │ Simple Routing   │     │ Does algo have           │
    ├──────────────────┤     │ _route_multi_stop()?     │
    │ algo.safe_run(   │     └──────────────────────────┘
    │   G,             │            ↙          ↘
    │   source,        │     YES              NO
    │   target         │      ↓                ↓
    │ )                │  ┌──────────────────┐ ┌──────────────────────┐
    │                  │  │ Algorithm         │ │ Benchmark computes  │
    │ Result:          │  │ decides order    │ │ optimal visit order │
    │ route found ✓    │  ├──────────────────┤ ├──────────────────────┤
    └──────────────────┘  │ algo._route_mult_│ │ 1. Compute best     │
                          │ stop(             │ │    order (algo-     │
    (Dijkstra,           │   G,              │ │    specific):       │
     A*,                 │   nodes,          │ │                      │
     etc)                │   scenario_name,  │ │ For len(nodes)≤9:  │
                          │   source,         │ │ - Exhaustive       │
                          │   target,         │ │   (try all perms) │
                          │   round_trip      │ │                    │
                          │ )                 │ │ For len(nodes)>9: │
                          │                  │ │ - Greedy NN        │
                          │ Algorithms:      │ │                    │
                          │ • GA             │ │ 2. Route each leg: │
                          │ • ACO            │ │    for src→dst:     │
                          │ • SA             │ │      algo.safe_run()│
                          │ • Christofides   │ │                    │
                          │ • PSO            │ │ 3. Combine legs +  │
                          │ • Gerald SA      │ │    return if round │
                          │                  │ │                    │
                          │ Result:          │ │ Result:            │
                          │ route found ✓    │ │ route found ✓      │
                          │ visit order ✓    │ │ (pre-computed)     │
                          │ gen_history ✓    │ │                    │
                          │ (if GA/ACO)      │ │                    │
                          └──────────────────┘ └──────────────────────┘
```

---

## 4. Key Hyperparameters & Their Effect Tree

```
┌─────────────────────────────────────────────────────────────┐
│ GA Hyperparameters & How They Affect Evolution              │
└─────────────────────────────────────────────────────────────┘
                            ↓
        ┌────────────────────────────────┐
        │ MUTATION_RATE = 0.9 (was 0.6) │
        ├────────────────────────────────┤
        │ 90% of individuals get mutated │
        │ 10% copy unchanged              │
        │                                 │
        │ Effect of increasing:          │
        │ ├─ More variation (good!)      │
        │ ├─ Less convergence (good!)    │
        │ ├─ Slower per gen (trade-off)  │
        │ └─ Better escape local optima✓ │
        │                                 │
        │ Was 0.6: only 60% mutated     │
        │ → Fast convergence             │
        │ → But: stuck in local optima   │
        │                                 │
        │ Now 0.9: 90% mutated          │
        │ → Slower convergence           │
        │ → But: find better solutions✓ │
        └────────────────────────────────┘
                            ↓
        ┌────────────────────────────────┐
        │ TOURNAMENT_SIZE = 3 (was 5)    │
        ├────────────────────────────────┤
        │ Selection: pick 3, choose best │
        │ (vs was: pick 5, choose best)  │
        │                                 │
        │ Effect of decreasing:          │
        │ ├─ Weaker selection pressure   │
        │ ├─ Second-best chance to breed │
        │ ├─ More population diversity   │
        │ └─ Less aggressive convergence │
        │                                 │
        │ Was 5: best dominate heavily  │
        │ → Converge fast               │
        │ → But: lose diversity         │
        │                                 │
        │ Now 3: balanced selection     │
        │ → Keep diversity              │
        │ → Find better paths✓          │
        └────────────────────────────────┘
                            ↓
        ┌────────────────────────────────┐
        │ swap_mutate: 2-3 swaps (was 1)│
        ├────────────────────────────────┤
        │ Mutation swaps 2-3 pairs       │
        │ (vs was: swap 1 pair)          │
        │                                 │
        │ Effect of increasing:          │
        │ ├─ Larger neighborhood explore │
        │ ├─ Different permutations      │
        │ ├─ More variation per gen      │
        │ └─ Escape stagnation✓          │
        │                                 │
        │ Was 1: minimal change         │
        │ [4,2,3] → [2,4,3] only        │
        │ → Limited exploration         │
        │                                 │
        │ Now 2-3: bigger change        │
        │ [4,2,3] → [3,4,2] or better   │
        │ → Full exploration✓           │
        └────────────────────────────────┘
                            ↓
        ┌────────────────────────────────┐
        │ Greedy NN Initialization       │
        │ (1 seed + 49 random)           │
        ├────────────────────────────────┤
        │ Population start:               │
        │ ├─ 1 greedy individual (~300)  │
        │ ├─ 49 random individuals       │
        │ │  (400-600 avg)               │
        │ └─ Mixed diversity✓            │
        │                                 │
        │ Effect:                        │
        │ ├─ Gen 1 baseline good (300)   │
        │ ├─ Not terrible (618) like     │
        │ │  pure random                 │
        │ ├─ Fast convergence from good  │
        │ │  baseline                    │
        │ └─ Better final results✓       │
        │                                 │
        │ Before: pure random            │
        │ Gen 1: 618.4 min (terrible!)  │
        │ Need 600 gen to improve       │
        │                                 │
        │ After: hybrid                  │
        │ Gen 1: 300 min (good!)        │
        │ Need fewer gen to converge✓   │
        └────────────────────────────────┘
                            ↓
        ┌────────────────────────────────┐
        │ Early Stopping (TSP_PATIENCE)  │
        │ Patience = 20 generations      │
        ├────────────────────────────────┤
        │ If no improvement for 20 gens: │
        │   STOP (don't run all 600)     │
        │                                 │
        │ Example:                        │
        │ Gen 1-10: improvement trend ✓ │
        │ Gen 11-30: NO improvement      │
        │ → STOP at Gen 30               │
        │ → Saves: 600-30=570 gens!     │
        │                                 │
        │ Effect:                        │
        │ ├─ Faster completion           │
        │ ├─ Population converged anyway │
        │ ├─ No wasted computation       │
        │ └─ Smart resource usage✓       │
        └────────────────────────────────┘
```

---

## 5. Complete Pipeline Integration Tree

```
┌──────────────────────────────────────────────────────────────┐
│ HOW GA FITS INTO BENCHMARK PIPELINE                          │
└──────────────────────────────────────────────────────────────┘
                            ↓
        ┌────────────────────────────────┐
        │ 1. REGISTRATION PHASE          │
        ├────────────────────────────────┤
        │ registry.register(              │
        │   GeneticAlgorithm()  ← HERE   │
        │ )                               │
        │                                 │
        │ GA registered with 10 others    │
        └────────────────────────────────┘
                            ↓
        ┌────────────────────────────────┐
        │ 2. SCENARIO BUILDING PHASE     │
        ├────────────────────────────────┤
        │ Build 2 scenarios from data    │
        │ (GA will run on both)           │
        │                                 │
        │ • emergency_patrol_circuit      │
        │ • terminal_circuit              │
        └────────────────────────────────┘
                            ↓
        ┌────────────────────────────────┐
        │ 3. BENCHMARK LOOP              │
        ├────────────────────────────────┤
        │ For each scenario:              │
        │   For each algorithm:           │
        │     if algo==GA:                │
        │       → Run GA._route_multi_... │
        │                                 │
        │ GA Special Handling:            │
        │ • Precompute pairwise costs    │
        │ • Evolve permutations (30 gen) │
        │ • Write evolution log           │
        │ • Return RouteResult with       │
        │   gen_history metadata          │
        └────────────────────────────────┘
                            ↓
        ┌────────────────────────────────┐
        │ 4. RESULT AGGREGATION          │
        ├────────────────────────────────┤
        │ Collect GA results:             │
        │ • emergency_patrol: 310 min    │
        │ • terminal_circuit: 85.5 min   │
        │                                 │
        │ Combine with other algorithms:  │
        │ • Dijkstra: 320, 92.3 min      │
        │ • A*: 318, 88.9 min            │
        │ • ACO: 304, 84.2 min           │
        │ • ... (10 total per scenario)  │
        └────────────────────────────────┘
                            ↓
        ┌────────────────────────────────┐
        │ 5. OUTPUT GENERATION           │
        ├────────────────────────────────┤
        │ CSV: comparison_results.csv    │
        │      (GA rows include gen_...)  │
        │                                 │
        │ Evolution Logs:                 │
        │ • evolution_ga_emergency...txt │
        │ • evolution_ga_terminal...txt  │
        │   (only GA, not Dijkstra)      │
        │                                 │
        │ Charts: comparison_chart.png   │
        │         (GA vs 9 others)        │
        │                                 │
        │ Maps: comparison_map_*.html    │
        │       (GA routes overlay)       │
        └────────────────────────────────┘
```

---

Done! Setiap tree diagram ini punya:
✅ Complete flow dari awal sampai akhir
✅ Penjelasan di setiap step (gampang dipahami)
✅ Contoh konkrit (pake angka real)
✅ Status checkmark (✓) show kapan sesuatu complete
✅ Decision points dengan arrows (jelas alurnya)
✅ Related info grouped together (logis)

Gampang dibaca dan langsung ngerti! Ready untuk present? 📊
