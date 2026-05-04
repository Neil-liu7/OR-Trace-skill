# Skill 2: Systematic Solving (Step-by-Step Computation)

After formulating the mathematical model, solve it using the following structured approach. Show ALL intermediate computation steps — do NOT jump to the answer.

## For Linear Programming (LP) Problems:

### Method A: Corner Point Enumeration (for 2-3 variable problems)
1. Graph the feasible region mentally or algebraically.
2. Find ALL corner points by solving pairs of constraint equations simultaneously.
3. Evaluate the objective function at EACH corner point.
4. The optimal value is the best (min or max) among all corner points.
5. List results in a table:
   | Corner Point (x, y) | Objective Value |
   |---|---|
   | ... | ... |

### Method B: Substitution and Bound Tightening (for structured problems)
1. From constraints, derive upper and lower bounds for each variable.
2. Use the objective's direction to determine which bounds to push toward.
3. Substitute bounds systematically and verify feasibility.

## For Integer Programming (IP) Problems:
1. First solve the LP relaxation (ignore integer constraints).
2. If the LP solution is already integer, you are done.
3. If not, check nearby integer points. Round in BOTH directions and verify feasibility.
4. Never assume rounding one direction is optimal — always check both floor and ceiling.

## For Combinatorial Problems (Set Cover, Assignment, etc.):
1. List all elements that must be covered/assigned.
2. List all options and what each option covers.
3. Use a greedy approach: pick the option covering the most uncovered elements.
4. After greedy selection, verify completeness.
5. Try to improve: can any selected option be replaced by a better one?
6. For small instances (n <= 15), try exhaustive enumeration of small combinations (size 2, 3, ...).

## General Rules:
- Write out every algebraic step. Do not skip simplifications.
- When dividing, note whether you need floor or ceiling (e.g., "need at least 11.67 trucks → need 12 trucks").
- Always substitute your final answer back into ALL constraints to verify feasibility.
- Compute the final objective value explicitly.
