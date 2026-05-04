# Skill 4: OR Problem Classification and Strategy Selection

Before attempting to solve, first classify the problem type. Different problem types require different solution strategies. Misclassification leads to wrong approaches and wasted effort.

## Classification Decision Tree:

### Question 1: What is being optimized?
- Cost, profit, time, distance, quantity, number of items → Optimization problem
- "Is it possible to..." or "Does there exist..." → Feasibility problem (set objective to find any feasible solution)

### Question 2: Are all relationships linear?
- Objective and all constraints are linear in decision variables → **Linear Program (LP)**
- Any nonlinear terms (products of variables, powers, log, etc.) → Nonlinear program

### Question 3: Must variables be integers?
- Yes (counts of items, people, vehicles, binary decisions) → **Integer Program (IP)** or **Mixed-Integer Program (MIP)**
- No (amounts, weights, proportions) → Standard LP

### Question 4: Is there combinatorial structure?
- "Minimum number of facilities to cover all areas" → **Set Cover Problem**
- "Assign tasks to workers at minimum cost" → **Assignment Problem**
- "Find shortest route visiting all cities" → **Traveling Salesman Problem (TSP)**
- "Transport goods from sources to destinations" → **Transportation Problem**
- "Schedule jobs on machines" → **Scheduling Problem**
- "Select items within capacity to maximize value" → **Knapsack Problem**
- "Allocate budget across periods to maximize return" → **Dynamic Programming / Multi-period planning**

## Strategy by Problem Type:

| Problem Type | Recommended Approach |
|---|---|
| Small LP (2-3 variables) | Corner point enumeration |
| Larger LP (4+ variables) | Simplex-like reasoning: identify binding constraints |
| Small IP (variables with small bounds) | Enumerate feasible integer solutions near LP optimum |
| Set Cover (n <= 15) | Greedy + exhaustive check for small cover sizes |
| Knapsack | Greedy by value/weight ratio, then verify |
| Multi-period planning | Dynamic programming: define states and transitions by period |
| Transportation | Northwest corner + stepping stone, or direct LP |
| Network flow | Find augmenting paths or reduce to LP |

## Multi-Period Problem Recognition:
If the problem involves multiple time periods (years, months, quarters):
1. Define variables indexed by period: x_t for period t.
2. Write inventory/balance equations linking periods: I_t = I_{t-1} + supply_t - demand_t.
3. Be careful about what is available in each period (e.g., "pilots trained in year 1 are available starting year 2").
4. Track cumulative effects and carry-over between periods.
