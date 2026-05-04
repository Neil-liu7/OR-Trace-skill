# Skill 1: Mathematical Modeling (OR Problem Formulation)

You are solving an Operations Research (OR) optimization problem. Before producing any answer, you MUST complete the following formulation steps explicitly in your response.

## Step 1: Identify Decision Variables

- Read the problem carefully. Ask: "What quantities am I choosing?"
- Define each variable with a clear symbol and its meaning (e.g., "Let x = number of trucks used per day").
- State the domain of each variable: continuous (x >= 0), integer (x in Z+), or binary (x in {0,1}).

## Step 2: Write the Objective Function

- Determine the optimization direction: minimize or maximize.
- Express the objective as a mathematical function of the decision variables.
- Write it explicitly: "Minimize Z = 5x + 10y" — do NOT skip this step.

## Step 3: List All Constraints

For EACH constraint in the problem:
1. Identify the resource, capacity, requirement, or logical condition described in natural language.
2. Translate it into a mathematical inequality or equation.
3. Label each constraint for clarity (e.g., "Budget constraint: 50x + 100y <= 1000").

Always include non-negativity constraints unless variables are explicitly unrestricted.

## Step 4: Handle Special Constraint Types

- **Percentage/ratio constraints**: Convert "at most 40% of vehicles are type B" into: B <= 0.4*(A + B), then simplify algebraically (e.g., 0.6B <= 0.4A => 3B <= 2A).
- **Conditional/logical constraints**: Use big-M formulation if needed (e.g., "if x > 0 then y >= 1" becomes y >= 1 - M*(1-b), x <= M*b, b in {0,1}).
- **"At least" vs "at most"**: Double-check inequality direction. "At least 300" means >= 300, "at most 100" means <= 100.

## Step 5: Write the Complete Model Summary

Before solving, write the complete model in standard form:

```
Minimize/Maximize: [objective function]
Subject to:
  (1) [constraint 1]
  (2) [constraint 2]
  ...
  (n) x, y, ... >= 0 (and integer if applicable)
```

Only proceed to solve AFTER you have written this complete summary.
