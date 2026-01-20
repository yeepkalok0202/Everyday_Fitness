# 4f: Different Selection types
import pandas as pd
import random
from ga_implementation import genetic_algorithm

selection_types = ["tournament", "rank", "roulette"]

results_f = []

count = 0
for type in selection_types:
    random.seed(42)
    result = genetic_algorithm(
      pop_size=1000,
      mutation_rate=0.05,
      crossover_rate=0.9,
      crossover_technique="two-point",
      selection_type=type,
    )
    count += 1
    results_f.append([count, *result])
df_f = pd.DataFrame(results_f, columns=["No of Exp","Mutation rate", "Cross over rate", "Cross over technique", "Population Size", "Generations to solve", "Selection type"])

print(df_f.to_string(index=False))