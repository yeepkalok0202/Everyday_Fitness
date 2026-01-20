# 4e: Different Crossover techniques
import pandas as pd
import random
from ga_implementation import genetic_algorithm

crossover_techniques = ["one-point", "two-point", "uniform"]

results_e = []

count = 0
for tech in crossover_techniques:
    random.seed(42)
    result = genetic_algorithm(
      pop_size=300,
      mutation_rate=0.2,
      crossover_rate=0.05,
      crossover_technique=tech,
      selection_type="tournament",
    )
    count += 1
    results_e.append([count, *result])
df_e = pd.DataFrame(results_e, columns=["No of Exp","Mutation rate", "Cross over rate", "Cross over technique", "Population Size", "Generations to solve", "Selection type"])

print(df_e.to_string(index=False))