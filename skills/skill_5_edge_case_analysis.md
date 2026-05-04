# Skill 5: Edge Case and Boundary Analysis

OR problems often have tricky edge cases that cause errors. Before finalizing your answer, explicitly check these common pitfalls.

## 1. Strict vs Non-strict Inequalities
- "less than" (<) vs "at most" (<=) — In LP, strict inequalities cannot be directly modeled. Convert:
  - "x must be less than y" → x <= y - 1 (if integer), or x <= y - epsilon (if continuous, then take limit)
  - "x must not exceed y" → x <= y (non-strict, standard)
- When the problem says "more than" or "exceeds", check if boundary is included.

## 2. Boundary / Corner Solutions
- Optimal solutions to LPs always occur at corner points of the feasible region.
- For minimization with costs: the optimum is often at the minimum feasible values.
- For maximization with profits: the optimum is often at the maximum feasible values or where constraints are tight.
- Always check: what happens if a variable is at its lower bound (often 0)? At its upper bound?

## 3. Infeasibility Detection
- Before solving, check: can ALL constraints be satisfied simultaneously?
- If constraints conflict (e.g., x >= 100 and x <= 50), the problem is infeasible.
- If infeasible, state so explicitly rather than providing a wrong answer.

## 4. Unboundedness Detection
- If you can keep improving the objective without violating any constraint, the problem is unbounded.
- This usually signals a missing constraint — re-read the problem.

## 5. Multiple Optimal Solutions
- If the objective function is parallel to a constraint at the optimum, there may be infinitely many optimal solutions.
- Report one optimal solution and note the objective value.

## 6. Degenerate Cases
- Zero-valued variables: just because a variable CAN be zero doesn't mean it SHOULD be.
- Redundant constraints: some constraints may not be active at the optimum — that's fine, but verify they're still satisfied.

## 7. Unit and Scale Consistency
- Verify all terms in each constraint use the same units.
- If one constraint is in "per day" and another in "per month", convert before combining.
- Watch for percentage vs fraction: "40%" = 0.4, not 40.

## 8. Re-read the Question
- "What is the minimum cost?" → report the objective value (a dollar amount).
- "How many trucks are needed?" → report the variable value (an integer).
- "What is the maximum profit?" → report the objective value.
- Sometimes the question asks for one specific variable, not the full solution.
- Some problems ask for a computed quantity that is NOT directly a variable or the objective (e.g., "how many pilots are available" = derived from variables).
