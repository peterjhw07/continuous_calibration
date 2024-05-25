from pulp import LpMinimize, LpProblem, LpVariable

# Create a LP problem
prob = LpProblem("Custom_MILP", LpMinimize)

# Define decision variables
x = LpVariable("x", lowBound=-5, cat='Integer')  # Continuous variable
y = LpVariable("y", lowBound=-5, cat='Integer')     # Integer variable


def func(x, y):
    ans = 2 * (x ** 2) + 3 * y + 4
    return ans


# Define the objective function
prob += func(x, y)

# Define constraints
prob += 3 * x + 4 * y <= 25
prob += 2 * x + y <= 10

# Solve the problem
prob.solve()

# Print the results
print("Status:", prob.status)
print("Objective value:", prob.objective.value())
for v in prob.variables():
    print(v.name, "=", v.varValue)
