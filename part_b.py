# 4b: Crossover only
import pandas as pd
import random
from ga_implementation import genetic_algorithm

crossover_rates = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.9]
results_b = []

count = 0
for rate in crossover_rates:
    random.seed(42)
    result = genetic_algorithm(
      pop_size=100,
      mutation_rate=0,
      crossover_rate=rate,
      crossover_technique="one-point",
      selection_type="tournament",
    )
    count += 1
    results_b.append([count, *result])
df_b = pd.DataFrame(results_b, columns=["No of Exp","Mutation rate", "Cross over rate", "Cross over technique", "Population Size", "Generations to solve", "Selection type"])

print(df_b.to_string(index=False))