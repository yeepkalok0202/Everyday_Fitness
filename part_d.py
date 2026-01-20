# 4d: Different Population size
import pandas as pd
import random
from ga_implementation import genetic_algorithm

pupulation_sizes = [50, 100, 300, 500, 700, 900, 1000]

results_d = []

count = 0
for pop_size in pupulation_sizes:
    random.seed(42)
    result = genetic_algorithm(
      pop_size=pop_size,
      mutation_rate=0.05,
      crossover_rate=0.9,
      crossover_technique="one-point",
      selection_type="tournament",
    )
    count += 1
    results_d.append([count, *result])
df_d = pd.DataFrame(results_d, columns=["No of Exp","Mutation rate", "Cross over rate", "Cross over technique", "Population Size", "Generations to solve", "Selection type"])

print(df_d.to_string(index=False))