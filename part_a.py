# 4a: Mutation only
import pandas as pd
import random
from ga_implementation import genetic_algorithm

mutation_rates = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.9]
results_a = []

count = 0
for rate in mutation_rates:
    random.seed(42)
    result = genetic_algorithm(
      pop_size=100,
      mutation_rate=rate,
      crossover_rate=0,
      crossover_technique="one-point",
      selection_type="tournament",
    )
    count += 1
    results_a.append([count, *result])
df_a = pd.DataFrame(results_a, columns=["No of Exp","Mutation rate", "Cross over rate", "Cross over technique", "Population Size", "Generations to solve", "Selection type"])

print(df_a.to_string(index=False))
