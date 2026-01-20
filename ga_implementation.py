import numpy as np
import random

def fitness(chromosome):
    x = list(chromosome)
    
    w = [10, 50, 200, 5, 20, 15, 300, 25, 100, 3]

    base = sum(w[i] * x[i] for i in range(10))

    bonus = 0.5 * (x[3] * x[7])

    if x[1] > 0.8 and x[5] < 4.0:
        penalty = 500
    else:
        penalty = 0

    return base + bonus - penalty

def calculate_theoretical_max():

    x = [
        3,      # x1: Subscription Tier (Max 3)
        1.0,    # x2: Caffeine (Max 1.0)
        1.0,    # x3: Carry Index (Max 1.0)
        100,    # x4: Shape-Shifting (Max 100)
        5,      # x5: Big Tech (Max 5)
        10.0,   # x6: Deadline Warp (Max 10.0)
        1.0,    # x7: Bug Immunity (Max 1.0)
        10,     # x8: LLM Whisperer (Max 10)
        1.0,    # x9: Luck (Max 1.0)
        100     # x10: Visibility (Max 100)
    ]

    # 2. Define Weights from the description
    # x1(10), x2(50), x3(200), x4(5), x5(20), x6(15), x7(300), x8(25), x9(100), x10(3)
    weights = [10, 50, 200, 5, 20, 15, 300, 25, 100, 3]

    # 3. Calculate Base Score: Sum(Weight * Value)
    base_score = sum(w * val for w, val in zip(weights, x))

    # 4. Calculate Synergy Bonus
    # Formula: 0.5 * (x4 * x8) -> Shape-Shifting * LLM Whisperer
    synergy_bonus = 0.5 * (x[3] * x[7])

    # 5. Calculate Burnout Penalty
    # Logic: Penalty if Caffeine > 0.8 AND Deadline Resistance < 4.0
    if x[1] > 0.8 and x[5] < 4.0:
        penalty = 500
    else:
        penalty = 0

    # 6. Final Theoretical Max
    total_max = base_score + synergy_bonus - penalty
    
    print("--- Theoretical Max Breakdown ---")
    print(f"Base Score:    {base_score:.2f}")
    print(f"Synergy Bonus: +{synergy_bonus:.2f}")
    print(f"Burnout Penalty: -{penalty:.2f}")
    print(f"-----------------------------")
    print(f"MAX FITNESS:   {total_max:.2f}")
    
    return total_max

# Store this in the global variable for your GA to access
MAX_FITNESS = calculate_theoretical_max()

# 1. Traits Constraints
TRAITS_INFO = [
    {"name": "Subscription Tier", "type": "int", "min": 0, "max": 3},
    {"name": "Caffeine Mutation", "type": "float", "min": 0.0, "max": 1.0},
    {"name": "Carry Index", "type": "float", "min": 0.0, "max": 1.0},
    {"name": "Tech Stack Shape-Shifting", "type": "int", "min": 0, "max": 100},
    {"name": "Big-Tech Proximity Aura", "type": "int", "min": 0, "max": 5},
    {"name": "Deadline Warp Resistance", "type": "float", "min": 0.0, "max": 10.0},
    {"name": "Bug Immunity", "type": "float", "min": 0.0, "max": 1.0},
    {"name": "LLM Whisperer Instinct", "type": "int", "min": 0, "max": 10},
    {"name": "Demo-Day Luck Factor", "type": "float", "min": 0.0, "max": 1.0},
    {"name": "Recruiter Visibility Gene", "type": "int", "min": 0, "max": 100}
]

def generate_individual():
    chrom = []
    for t in TRAITS_INFO:
        if t['type'] == 'int':
            chrom.append(random.randint(t['min'], t['max']))
        else:
            chrom.append(random.uniform(t['min'], t['max']))
    return chrom

def generate_population(pop_size):
    return [generate_individual() for _ in range(pop_size)]

def crossover_one_point(p1, p2):
    point = random.randint(1, len(p1)-1)
    c1 = p1[:point] + p2[point:]
    c2 = p2[:point] + p1[point:]
    return c1, c2

def crossover_two_point(p1, p2):
    size = len(p1)

    point1 = random.randint(1, size - 2)
    point2 = random.randint(point1 + 1, size - 1)

    c1 = p1[:point1] + p2[point1:point2] + p1[point2:]
    c2 = p2[:point1] + p1[point1:point2] + p2[point2:]
    return c1, c2

def crossover_uniform(p1, p2):
    c1 = list(p1)
    c2 = list(p2)
    for i in range(len(p1)):
        if random.random() < 0.5:
            c1[i], c2[i] = c2[i], c1[i]
    return c1, c2

def selection_tournament(population, fitnesses, k=3):
    selected = []
    pop_len = len(population)
    for _ in range(pop_len):
        aspirants = random.sample(range(pop_len), k)
        best_idx = aspirants[0]
        for idx in aspirants[1:]:
            if fitnesses[idx] > fitnesses[best_idx]:
                best_idx = idx
        selected.append(population[best_idx])
    return selected

def selection_rank(population, fitnesses):
    sorted_indices = sorted(range(len(fitnesses)), key=lambda k: fitnesses[k])
    sorted_pop = [population[i] for i in sorted_indices]

    # Linear Ranking Weights
    N = len(population)
    ranks = range(1, N + 1)
    total_rank = sum(ranks)
    probs = [r / total_rank for r in ranks]

    # Select N parents based on rank probabilities
    indices = np.random.choice(N, size=N, p=probs)
    selected = [sorted_pop[i] for i in indices]
    return selected

def selection_roulette(population, fitnesses):
    # Handle negative fitness by shifting
    min_f = min(fitnesses)
    if min_f < 0:
        # shift so min is 0, plus a small epsilon to avoid 0 probability if all same
        adj_fitnesses = [f - min_f + 1e-6 for f in fitnesses]
    else:
        adj_fitnesses = [f + 1e-6 for f in fitnesses]

    total_fit = sum(adj_fitnesses)
    probs = [f / total_fit for f in adj_fitnesses]

    # Spin the wheel len(population) times
    indices = np.random.choice(len(population), size=len(population), p=probs)
    selected = [population[i] for i in indices]
    return selected

def mutate(individual, rate):
    new_ind = individual[:]
    for i in range(len(new_ind)):
        if random.random() < rate:
            t = TRAITS_INFO[i]
            if t['type'] == 'int':
                new_ind[i] = random.randint(t['min'], t['max'])
            else:
                new_ind[i] = random.uniform(t['min'], t['max'])
    return new_ind

import random
import numpy as np

MAX_FITNESS = 2480

def genetic_algorithm(pop_size, mutation_rate, crossover_rate, crossover_technique, selection_type, max_generation=15000):
    # 1. Map strings to functions
    crossover_map = {
        "one-point": crossover_one_point,
        "two-point": crossover_two_point,
        "uniform": crossover_uniform
    }
    
    selection_map = {
        "tournament": selection_tournament,
        "rank": selection_rank,
        "roulette": selection_roulette
    }
        
    crossover_func = crossover_map[crossover_technique]
    selection_func = selection_map[selection_type]

    # 2. Initialization
    population = generate_population(pop_size)
    
    global_best_ind = None
    global_best_fit = -float('inf')
    
    generation_count = 0
    
    # 4. The Loop
    while generation_count < max_generation:
        generation_count += 1
        
        # Calculate Fitness
        fitnesses = [fitness(ind) for ind in population]
        curr_best_fit = max(fitnesses)
        curr_best_ind = population[fitnesses.index(curr_best_fit)]
        
        # 5. Check for Improvement
        if curr_best_fit > global_best_fit:
            global_best_fit = curr_best_fit
            global_best_ind = curr_best_ind[:]
            
            # Success Check
            if round(global_best_fit, 1) >= MAX_FITNESS:
                break
            
        # 7. Selection & Breeding
        parents = selection_func(population, fitnesses)
        next_gen = []
        
        # Always keep the best
        next_gen.append(global_best_ind[:])
        
        while len(next_gen) < pop_size:
            p1 = random.choice(parents)
            p2 = random.choice(parents)
            
            # Crossover Check
            if random.random() < crossover_rate:
                c1, c2 = crossover_func(p1, p2)
            else:
                c1, c2 = p1, p2
            
            # Mutation Check
            c1 = mutate(c1, mutation_rate)
            if len(next_gen) < pop_size:
                next_gen.append(c1)
                
            c2 = mutate(c2, mutation_rate)
            if len(next_gen) < pop_size:
                next_gen.append(c2)
        
        population = next_gen
    
    if round(global_best_fit, 1) >= MAX_FITNESS:
        final_generations = generation_count
    else:
        final_generations = f"Failed (Best Fitness: {global_best_fit:.1f})"
    
    return [mutation_rate, crossover_rate, crossover_technique, pop_size, final_generations, selection_type]