# 4c: Mutation + Crossover only
import pandas as pd
import random
from ga_implementation import genetic_algorithm

mutation_rates = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.9]
crossover_rates = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.9]
results_c = []

count = 0
for m_rate in mutation_rates:
    for c_rate in crossover_rates:
      random.seed(42)
      result = genetic_algorithm(
        pop_size=100,
        mutation_rate=m_rate,
        crossover_rate=c_rate,
        crossover_technique="one-point",
        selection_type="tournament",
      )
      count += 1
      results_c.append([count, *result])
df_c = pd.DataFrame(results_c, columns=["No of Exp","Mutation rate", "Cross over rate", "Cross over technique", "Population Size", "Generations to solve", "Selection type"])

print(df_c.to_string(index=False))