# Skill 3: Verification and Self-Correction

Before outputting your final answer, you MUST perform the following verification checklist. This step prevents the most common errors.

## Verification Checklist:

### Check 1: Feasibility Verification
Substitute your proposed solution (x*, y*, ...) back into EVERY constraint:
- For each constraint, compute the left-hand side value.
- Verify the inequality/equation is satisfied.
- If ANY constraint is violated, your answer is WRONG — go back and fix it.

Format:
```
Constraint (1): 2*x + 3*y = 2*75 + 3*50 = 300 >= 300 ✓
Constraint (2): 3*y <= 2*x → 3*50=150 <= 2*75=150 ✓
...
```

### Check 2: Objective Value Computation
- Compute the objective function value at your solution explicitly.
- This is the number you report as the final answer (NOT the variable values).

### Check 3: Optimality Sanity Check
- Can you find a nearby feasible solution with a better objective value?
- For minimization: try slightly reducing the objective. Is it still feasible?
- For maximization: try slightly increasing the objective. Does it violate constraints?
- If you find a better feasible solution, your original answer was NOT optimal.

### Check 4: Integer Feasibility (if applicable)
- If variables must be integers, verify ALL variable values are integers.
- If you rounded from an LP relaxation, verify the rounded solution is still feasible.

### Check 5: Problem Re-reading
- Re-read the original question one more time.
- Verify: are you answering what was asked? (e.g., "minimum number of scooters" vs "minimum total vehicles" vs "minimum cost")
- Verify: did you use the correct units? (e.g., the answer might ask for cost in dollars, not number of items)

## Common Error Patterns to Watch For:
1. **Wrong inequality direction**: "at least" means >=, "at most" means <=, "no more than" means <=.
2. **Ratio constraint algebra errors**: When converting "at most 40% are type B" → always verify by plugging numbers back.
3. **Forgetting a constraint**: Count the constraints in your model vs. the conditions in the problem text.
4. **Off-by-one in integer rounding**: ceil(11.67) = 12, not 11. floor(11.67) = 11, not 12.
5. **Answering the wrong quantity**: The question asks for the objective value? Or the variable value? Or something else?

## Output Format:
After verification, output your final answer as:
#### [numerical answer]
