# Presentasi: Proses Kerja Debugging GA Algorithm

## Executive Summary
Dokumentasi ini menjelaskan **bagaimana cara kami bekerja** untuk mengidentifikasi dan memperbaiki issue pada Genetic Algorithm (GA) di project TugasSCRouting. Fokus pada methodology, approach, dan proses collaboration, bukan technical details.

---

## 1. Problem Discovery Phase

### Awal Proses (Fase 1)
**Waktu:** Week 1-2 Debugging
**Objective:** Memahami apa yang sedang terjadi dengan GA

#### 1.1 Initial Observation
```
User Report:
"GA perform same from generation 1 to last generation, 
 like no change at all, from route or stop order"
```

**Apa yang kami lakukan:**
- Membaca code GA secara top-level untuk understand structure
- Melihat evolution logs yang dihasilkan oleh GA
- Membuat comparison antara GA output vs algorithm lain

#### 1.2 Metodologi: Visual Data Analysis
```
Strategy: Jangan langsung dive ke code, 
          lihat dulu output apa yang dihasilkan
          
Tools yang dipakai:
├─ evolution_ga_emergency_patrol_circuit.txt (log file)
├─ comparison_results.csv (benchmark results)
├─ comparison_map_*.html (visual route)
└─ Manual inspection
```

**Hasil Observasi:**
```
Generation 1:  visit_order = [4, 2, 1, 3], cost = 618.4 min
Generation 2:  visit_order = [4, 2, 1, 3], cost = 618.4 min
Generation 3:  visit_order = [4, 2, 1, 3], cost = 618.4 min
...
Generation 30: visit_order = [4, 2, 1, 3], cost = 618.4 min

Kesimpulan:
✗ Permutation TIDAK BERUBAH across generations
✗ NO improvement pada fitness (all 618.4)
✗ GA mungkin tidak evolving sama sekali
```

---

## 2. Root Cause Analysis Phase (RCA)

### Fase 2: Questioning & Hypotheses

**Question yang kami tanyakan:**
1. "Apakah GA mendapat kesempatan untuk mutasi/evolve?"
2. "Apakah selection mechanism bekerja dengan benar?"
3. "Apakah mutation rate/type cukup untuk create variation?"
4. "Apakah population initialization menciptakan diversity?"

#### 2.1 Metodologi: Systematic Code Inspection

**Pendekatan:**
```
Level 1: High-level flow
  Lihat _route_multi_stop() function structure
  → Understand apa yang diinput dan output

Level 2: Population initialization
  Check bagaimana population dibuat
  → Apakah ada diversity di Gen 1?

Level 3: Evolution loop
  Trace melalui crossover, mutation, selection
  → Apakah changes benar-benar terjadi?

Level 4: Best solution tracking
  Gimana best_perm dipilih dan di-update
  → Apakah best_perm bisa berubah?
```

#### 2.2 Metodologi: Hypothesis Testing dengan Evidence

**Hypothesis 1: "Population initialization salah"**
```
Test: Lihat Gen 1 fitness values
Evidence dari evolution log:
  Gen 1: [618.4, 612.3, 625.1, 620.0, 618.9, ...]
  → Fitness jadi tidak uniform → Population punya diversity
  → Hypothesis: SALAH (population init kayaknya ok)
```

**Hypothesis 2: "Best perm terjebak di input order"**
```
Test: Trace best_perm initialization di code
  
Kode lama:
  best_perm = intermediates[:]  ← Input order
  best_cost = tour_cost(best_perm)
  
Pertanyaan: Apakah ini prevent GA dari explore?

Analisis:
- intermediates = facility order dari DB
- Setiap permutasi yang generated dicompare dengan best_perm ini
- Jika crossover + mutation hanya create permutasi mirip input,
  best_perm tetap sebagai best
  
Kesimpulan: HYPOTHESIS DIPERKUAT

Evidence: Kita track setiap generated child, 
          ternyata banyak yang mirip [4,2,1,3]
          (input order)
```

**Hypothesis 3: "Mutation terlalu lemah"**
```
Test: Hitung berapa banyak variation dari 1 mutation

Original: [4, 2, 1, 3]
1 swap mutation: hanya ~6 permutasi berbeda dari 24 total
  → Mutation rate eksplorasi hanya ~25% dari solution space

Test: Coba increase mutation
Result: Visit order masih tidak berubah
        (karena masalah #2 lebih dominan - best_perm init)
```

### Hasil RCA: Prioritized Root Causes

```
Priority 1 (CRITICAL):
├─ Best perm initialization salah (forced input order)
└─ GA tidak bisa escape dari input order

Priority 2 (HIGH):
├─ Mutation rate terlalu rendah (0.6 = 60%)
├─ Mutation type terlalu lemah (1 swap)
└─ Tournament size terlalu besar (5)
└─ Population tidak maintain diversity

Priority 3 (MEDIUM):
└─ Population initialization pure random
  └─ Gen 1 baseline jelek (618 min vs target 300)
```

---

## 3. Solution Design Phase

### Fase 3: Brainstorming & Planning

#### 3.1 Metodologi: Prioritized Fix Strategy

**Prinsip:**
```
Fix yang paling impactful duluan
Validate setiap fix sebelum next
```

**Fix Plan (Urutan):**

```
Fix 1: Best Perm Initialization (CRITICAL)
├─ Impact: VERY HIGH (unblock GA evolution)
├─ Risk: LOW (simple logic change)
├─ Effort: LOW (5 lines code)
└─ Expected: GA bisa explore permutasi berbeda

Fix 2: Mutation Rate & Swap Mutation (HIGH)
├─ Impact: HIGH (increase exploration)
├─ Risk: LOW (parameter tuning)
├─ Effort: LOW (2 lines code)
└─ Expected: More variation per generation

Fix 3: Tournament Size (HIGH)
├─ Impact: MEDIUM (maintain diversity)
├─ Risk: LOW (parameter tuning)
├─ Effort: LOW (1 line code)
└─ Expected: Less aggressive convergence

Fix 4: Population Initialization (MEDIUM)
├─ Impact: MEDIUM (better Gen 1 baseline)
├─ Risk: MEDIUM (add new function)
├─ Effort: MEDIUM (30 lines code)
└─ Expected: Gen 1 cost berkurang 600→300
```

#### 3.2 Metodologi: Design Review & Validation

**Sebelum implement:**

```
1. Read GA theory
   - Understand genetic operators
   - Know apa yang best practice
   
2. Check existing code
   - Lihat method yang already ada
   - Reuse pattern yang sudah proven
   
3. Walthrough logic
   - Trace through manually
   - Check untuk edge cases
   
4. Compare vs industry standard
   - Greedy NN initialization
   - Standard mutation rate (0.8-1.0)
   - Common tournament sizes (2-5)
```

---

## 4. Implementation Phase

### Fase 4: Coding dengan Validation

#### 4.1 Fix 1: Best Perm Initialization

**Approach:**
```
1. Find the exact lines (line 753-761)
2. Understand current logic
3. Design new logic
4. Implement
5. Add comment untuk explain why
```

**Proses:**
```python
# BEFORE (Problem):
best_perm = intermediates[:]  # ← Terjebak di sini
best_cost = tour_cost(best_perm)

# AFTER (Fixed):
fitness = [tour_cost(p) for p in population]  # Evaluate semua
best_idx = min(range(len(population)), key=lambda i: fitness[i])
best_perm = population[best_idx][:]  # Take best dari populasi
best_cost = fitness[best_idx]
```

**Validation:**
```
Check: best_perm sekarang dari evaluated population
Check: best_cost is fitness[best_idx]
Logic walk-through: Pada gen 1, best_perm = individual terbaik
                    Kemudian bisa update ke individual lain di gen 2+
```

#### 4.2 Fix 2 & 3: Hyperparameters

**Approach:**
```
Cari class attributes di GeneticAlgorithm
Ubah MUTATION_RATE: 0.6 → 0.9
Ubah TOURNAMENT_SIZE: 5 → 3
```

**Justification:**
```
MUTATION_RATE = 0.9:
  Reason: 90% individu dimutasi = tinggi
  Standard: 0.8-1.0 range
  Expected: More variation per gen

TOURNAMENT_SIZE = 3:
  Reason: Balance selection pressure + diversity
  Standard: 2-5 range
  Expected: Less aggressive convergence
```

#### 4.3 Fix 4: Greedy NN Initialization

**Approach:**
```
1. Understand problem:
   Gen 1 cost = 618 min (pure random)
   Target = 300 min
   
2. Research solution:
   Nearest neighbor heuristic
   Industry standard untuk TSP init
   
3. Design implementation:
   Function: greedy_nn_order()
   Input: start_node, targets, pairwise costs
   Output: ordered list
   
4. Implement:
   a) Code greedy_nn_order function
   b) Use 1 greedy result + 19 random
   c) Population diversity maintained
   
5. Validate:
   Check Gen 1 cost improvement
   Verify random still in population
   Test edge cases (1 target, 2 targets, etc)
```

**Kode yang ditambahkan:**
```python
def greedy_nn_order(start_node, targets, pair_cost_dict):
    """Build nearest-neighbor tour greedily"""
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

# Usage in population init:
greedy_perm = greedy_nn_order(start, intermediates, pair_cost)
population.append(greedy_perm)

for _ in range(self.TSP_POPULATION_SIZE - 1):
    perm = intermediates[:]
    rng.shuffle(perm)
    population.append(perm)
```

---

## 5. Testing & Validation Phase

### Fase 5: Verifying Fixes Work

#### 5.1 Metodologi: Progressive Validation

**Validation Strategy:**
```
After Fix 1:
  ✓ Check apakah best_perm initialization correct
  ✓ Check apakah best_perm bisa change across gen
  ? Result masih stuck? → Debug lebih lanjut
  
After Fix 2 & 3:
  ✓ Check apakah mutation lebih sering (90% vs 60%)
  ✓ Check apakah tournament size applied
  ? Result masih stuck? → Mutation mungkin masih weak
  
After Fix 4:
  ✓ Check apakah Gen 1 cost lebih baik (285 vs 618)
  ✓ Check apakah visit_order changing per gen
  ✓ Check apakah improvement trend ada
```

#### 5.2 What to Check

```
1. Evolution Logs:
   cat data/evolution_ga_emergency_patrol_circuit.txt
   
   Cari:
   ├─ Gen 1 cost: Should be ~285-330 (not 618)
   ├─ Gen 2-30 trend: Should show variation
   ├─ Visit order: Should change per generation
   └─ Improvement: Should trend downward (better)

2. Comparison Results:
   data/comparison_results.csv
   
   Cari:
   ├─ GA_time vs other algorithms
   ├─ GA_distance vs optimal
   └─ GA competitive dengan others?

3. Visual Routes:
   data/comparison_map_*.html
   
   Lihat:
   └─ Route shape reasonable? Not all over the place?
```

---

## 6. Working Process & Collaboration

### Methodology Summary

**Process Framework yang kami gunakan:**

```
1. OBSERVE & UNDERSTAND
   ├─ Read code structure
   ├─ Analyze output logs
   ├─ Create mental model
   └─ Identify discrepancies

2. QUESTION & HYPOTHESIZE
   ├─ Ask specific questions
   ├─ Formulate hypotheses
   ├─ Design tests untuk validate
   └─ Gather evidence dari code/logs

3. ANALYZE & PRIORITIZE
   ├─ Map root causes
   ├─ Assess impact/risk
   ├─ Prioritize fixes
   └─ Create action plan

4. DESIGN & PLAN
   ├─ Research best practices
   ├─ Sketch solution
   ├─ Code review sebelum implement
   └─ Plan validation approach

5. IMPLEMENT & ITERATE
   ├─ Code one fix at a time
   ├─ Validate setiap fix
   ├─ Debug issues immediately
   └─ Document changes

6. TEST & VERIFY
   ├─ Run full benchmark
   ├─ Check evolution logs
   ├─ Compare vs baseline
   └─ Assess improvement
```

### Key Principles yang kami terapkan

```
1. UNDERSTAND BEFORE FIX
   ✓ Trace through code manually
   ✓ Look at actual output
   ✓ Don't guess, gather evidence
   
2. ONE FIX AT A TIME
   ✓ Fix yang paling critical duluan
   ✓ Validate setiap fix isolated
   ✓ Jangan bundle fixes (hard to debug)
   
3. DATA-DRIVEN DECISION
   ✓ Look at logs, not assumption
   ✓ Compare before/after quantitatively
   ✓ Use metrics untuk justify changes
   
4. SIMPLE OVER COMPLEX
   ✓ Prefer parameter tuning vs rewrite
   ✓ Small code changes over large
   ✓ Use existing patterns/functions
   
5. VALIDATE CONSTANTLY
   ✓ After each fix, check results
   ✓ Compare output vs expectations
   ✓ Look for unintended side effects
```

---

## 7. Timeline & Milestones

```
Week 1: Understanding & Discovery
├─ Read CLAUDE.md and project structure
├─ Run python main.py explore/demo
├─ Understand GA at high level
└─ Milestone: Know what GA should do

Week 2: Deep Investigation
├─ Read algorithms.py details
├─ Analyze evolution logs
├─ Identify visit_order stuck issue
├─ Hypothesis: best_perm init problem
└─ Milestone: Root cause identified

Week 3: Solution Design & Planning
├─ Read GA theory/best practices
├─ Design 4-part fix strategy
├─ Code review fix approach
├─ Plan validation
└─ Milestone: Fix plan ready

Week 4: Implementation & Testing
├─ Implement Fix 1-4 progressively
├─ Validate after each fix
├─ Run full benchmark
├─ Analyze improvement
└─ Milestone: All fixes implemented & tested

Week 5: Documentation & Presentation
├─ Create explanation documents
├─ Document process/working approach
├─ Create presentation slides
└─ Milestone: Ready untuk present
```

---

## 8. Challenges & How We Solved Them

### Challenge 1: GA Code Complexity
```
Problem: Algorithm.py panjang dengan banyak functions
Solution: 
  ├─ Trace execution path, don't read semua
  ├─ Focus on _route_multi_stop() method
  └─ Create simplified mental model
```

### Challenge 2: Multiple Root Causes
```
Problem: Issue tidak hanya satu, tapi 5 causes terkait
Solution:
  ├─ Prioritize by impact (critical first)
  ├─ Fix incrementally
  ├─ Validate each one
  └─ Don't try fix semuanya sekaligus
```

### Challenge 3: Hard to Debug Evolution
```
Problem: Genetic algorithm complex, hard to trace manually
Solution:
  ├─ Use evolution logs untuk tracking
  ├─ Look at Gen 1 vs Gen 30 output
  ├─ Compare fitness trend
  └─ Check visit_order changes
```

### Challenge 4: Uncertain Hyperparameters
```
Problem: MUTATION_RATE=0.9 vs 0.8? TOURNAMENT_SIZE=3 vs 4?
Solution:
  ├─ Research industry standards
  ├─ Use similar algorithms dalam codebase
  ├─ Start dengan standard, adjust jika needed
  └─ Validate dengan full benchmark run
```

---

## 9. Key Learnings & Best Practices

### Learning 1: Evolution Logs are Invaluable
```
Best Practice: Always output detailed logs
  ├─ Per-generation metrics
  ├─ Best/average/worst fitness
  ├─ Current best solution
  └─ Helps debugging significantly
```

### Learning 2: Understand the Theory
```
Best Practice: Before fixing GA, understand:
  ├─ Selection mechanisms
  ├─ Crossover operators
  ├─ Mutation strategies
  ├─ Fitness landscape
  └─ Let theory guide fixes
```

### Learning 3: Small Code Changes, Big Impact
```
Best Practice: 
  ├─ best_perm init = 5 lines, CRITICAL impact
  ├─ Hyperparameter tune = 2 lines, HIGH impact
  ├─ Greedy init = 30 lines, MEDIUM impact
  └─ Don't need full rewrite untuk fix
```

### Learning 4: Validation is Continuous
```
Best Practice:
  ├─ After every fix → run full benchmark
  ├─ Check evolution logs
  ├─ Compare metrics
  ├─ Don't assume it works
  └─ Test end-to-end
```

---

## 10. What We'll Present

### Untuk Presentasi Nanti:

**Part 1: Problem Statement** (2 min)
```
- Observasi awal: GA tidak berkembang
- Evidence dari logs
- Impact: GA uncompetitive vs other algorithms
```

**Part 2: Problem-Solving Process** (3 min)
```
- How we discovered root causes
- Evidence gathering methodology
- Prioritization approach
```

**Part 3: Solution & Implementation** (3 min)
```
- Fix strategy (4 fixes, prioritized)
- Each fix explained simply
- Why each fix matters
```

**Part 4: Results & Validation** (2 min)
```
- Before/after comparison
- Evolution log improvements
- GA competitiveness now
```

---

## Summary: Our Working Approach

```
Our Methodology:
  1. Observe actual output (not assumptions)
  2. Ask specific questions
  3. Test hypotheses with evidence
  4. Prioritize fixes by impact
  5. Implement simple, validated changes
  6. Test continuously
  7. Document process

Key Values:
  ✓ Data-driven (use logs, metrics)
  ✓ Systematic (trace code carefully)
  ✓ Incremental (one fix at a time)
  ✓ Validated (test after each change)
  ✓ Documented (explain decisions)

This approach works untuk:
  - Complex algorithms
  - Multi-cause problems
  - Team collaboration
  - Future maintenance
```
