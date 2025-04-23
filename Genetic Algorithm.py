# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 09:18:55 2025
@author: Vivi Rodriguez
"""

import random
import copy
import matplotlib.pyplot as plt  # <-- Añadido para visualización

# Node capacities
nodes = {
    1: {"cpu": 8, "mem": 16, "bw": 100},
    2: {"cpu": 4, "mem": 32, "bw": 50},
    3: {"cpu": 16, "mem": 8, "bw": 200},
    4: {"cpu": 12, "mem": 24, "bw": 150},
    5: {"cpu": 6, "mem": 12, "bw": 80}
}

# Resource usage constraints
CPU_LIMIT = 0.8
MEM_LIMIT = 0.9
BW_LIMIT = 0.75

# Example task requirements and execution times (per node)
tasks = [
    {"cpu": 2, "mem": 4, "bw": 30, "exec_times": [10, 15, 8, 12, 14]},
    {"cpu": 4, "mem": 8, "bw": 20, "exec_times": [12, 20, 10, 15, 18]},
    {"cpu": 1, "mem": 2, "bw": 10, "exec_times": [8, 10, 6, 9, 11]},
    {"cpu": 3, "mem": 6, "bw": 25, "exec_times": [14, 18, 12, 16, 17]},
    {"cpu": 2, "mem": 4, "bw": 15, "exec_times": [11, 13, 9, 12, 13]},
    {"cpu": 5, "mem": 10, "bw": 40, "exec_times": [20, 25, 16, 19, 22]},
    {"cpu": 2, "mem": 5, "bw": 18, "exec_times": [13, 17, 10, 14, 15]},
    {"cpu": 3, "mem": 7, "bw": 22, "exec_times": [15, 19, 13, 17, 18]},
    {"cpu": 1, "mem": 3, "bw": 12, "exec_times": [9, 11, 7, 10, 12]},
    {"cpu": 4, "mem": 9, "bw": 35, "exec_times": [18, 23, 15, 20, 21]}
]
NUM_TASKS = len(tasks)
NUM_NODES = len(nodes)

# Fitness function
def calculate_fitness(schedule):
    node_usage = {n: {"cpu": 0, "mem": 0, "bw": 0, "time": 0} for n in nodes}
    penalty = 0

    for t_idx, node in enumerate(schedule):
        task = tasks[t_idx]
        node_usage[node]["cpu"] += task["cpu"]
        node_usage[node]["mem"] += task["mem"]
        node_usage[node]["bw"] += task["bw"]
        node_usage[node]["time"] += task["exec_times"][node-1]

    for n, res in node_usage.items():
        if res["cpu"] > nodes[n]["cpu"] * CPU_LIMIT:
            penalty += 1000 * (res["cpu"] - nodes[n]["cpu"] * CPU_LIMIT)
        if res["mem"] > nodes[n]["mem"] * MEM_LIMIT:
            penalty += 1000 * (res["mem"] - nodes[n]["mem"] * MEM_LIMIT)
        if res["bw"] > nodes[n]["bw"] * BW_LIMIT:
            penalty += 1000 * (res["bw"] - nodes[n]["bw"] * BW_LIMIT)

    makespan = max(res["time"] for res in node_usage.values())
    utilization = sum(
        (res["cpu"] / (nodes[n]["cpu"] * CPU_LIMIT) +
         res["mem"] / (nodes[n]["mem"] * MEM_LIMIT) +
         res["bw"] / (nodes[n]["bw"] * BW_LIMIT)) / 3
        for n, res in node_usage.items()
    ) / NUM_NODES

    return makespan + penalty - 10 * utilization

# Generate a random valid schedule
def random_schedule():
    schedule = []
    for t in range(NUM_TASKS):
        valid_nodes = []
        for n in nodes:
            if (tasks[t]["cpu"] <= nodes[n]["cpu"] * CPU_LIMIT and
                tasks[t]["mem"] <= nodes[n]["mem"] * MEM_LIMIT and
                tasks[t]["bw"] <= nodes[n]["bw"] * BW_LIMIT):
                valid_nodes.append(n)
        schedule.append(random.choice(valid_nodes))
    return schedule

def initialize_population(pop_size):
    return [random_schedule() for _ in range(pop_size)]

# Tournament selection
def tournament_selection(pop, fitnesses, k=3):
    selected = random.sample(list(zip(pop, fitnesses)), k)
    selected.sort(key=lambda x: x[1])
    return copy.deepcopy(selected[0][0])

# Uniform crossover
def crossover(parent1, parent2):
    return [g1 if random.random() < 0.5 else g2 for g1, g2 in zip(parent1, parent2)]

# Mutation
def mutate(schedule, mutation_rate=0.1):
    new_schedule = schedule[:]
    for i in range(NUM_TASKS):
        if random.random() < mutation_rate:
            valid_nodes = []
            for n in nodes:
                if (tasks[i]["cpu"] <= nodes[n]["cpu"] * CPU_LIMIT and
                    tasks[i]["mem"] <= nodes[n]["mem"] * MEM_LIMIT and
                    tasks[i]["bw"] <= nodes[n]["bw"] * BW_LIMIT):
                    valid_nodes.append(n)
            new_schedule[i] = random.choice(valid_nodes)
    return new_schedule

## 
def get_resource_utilization(schedule):
    cpu_used = {n: 0 for n in nodes}
    mem_used = {n: 0 for n in nodes}

    for t_idx, node in enumerate(schedule):
        task = tasks[t_idx]
        cpu_used[node] += task["cpu"]
        mem_used[node] += task["mem"]

    total_cpu_ratio = 0
    total_mem_ratio = 0
    for n in nodes:
        cpu_ratio = cpu_used[n] / (nodes[n]["cpu"] * CPU_LIMIT)
        mem_ratio = mem_used[n] / (nodes[n]["mem"] * MEM_LIMIT)
        total_cpu_ratio += cpu_ratio
        total_mem_ratio += mem_ratio

    avg_cpu_util = (total_cpu_ratio / NUM_NODES) * 100
    avg_mem_util = (total_mem_ratio / NUM_NODES) * 100

    return avg_cpu_util, avg_mem_util


# Main genetic algorithm
def genetic_algorithm(pop_size=50, generations=100, crossover_rate=0.8, mutation_rate=0.1, elitism=2):
    population = initialize_population(pop_size)
    best_schedule = None
    best_fitness = float('inf')
    fitness_history = []

    for gen in range(generations):
        fitnesses = [calculate_fitness(ind) for ind in population]
        gen_best = min(fitnesses)
        fitness_history.append(gen_best)

        if gen_best < best_fitness:
            best_fitness = gen_best
            best_schedule = population[fitnesses.index(gen_best)]

        elite_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])[:elitism]
        new_population = [copy.deepcopy(population[i]) for i in elite_indices]

        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            child = crossover(parent1, parent2) if random.random() < crossover_rate else parent1[:]
            child = mutate(child, mutation_rate)
            new_population.append(child)
        population = new_population

    return best_schedule, best_fitness, fitness_history

# Run the algorithm
best_schedule, best_fitness, fitness_history = genetic_algorithm()

# Output
print("Best Schedule (Task → Node):")
for t_idx, node in enumerate(best_schedule):
    print(f"Task {t_idx+1} → Node {node}")
print(f"Best Fitness (lower is better): {best_fitness}")


# Visualización con matplotlib
plt.figure(figsize=(10, 5))
plt.plot(fitness_history, label="Fitness")
plt.title("Evolución del Fitness por Generación")
plt.xlabel("Generación")
plt.ylabel("Fitness")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# First-Fit Heuristic
def first_fit():
    node_usage = {n: {"cpu": 0, "mem": 0, "bw": 0, "time": 0} for n in nodes}
    schedule = []
    for t_idx, task in enumerate(tasks):
        for n in nodes:
            if (node_usage[n]["cpu"] + task["cpu"] <= nodes[n]["cpu"] * CPU_LIMIT and
                node_usage[n]["mem"] + task["mem"] <= nodes[n]["mem"] * MEM_LIMIT and
                node_usage[n]["bw"] + task["bw"] <= nodes[n]["bw"] * BW_LIMIT):
                schedule.append(n)
                node_usage[n]["cpu"] += task["cpu"]
                node_usage[n]["mem"] += task["mem"]
                node_usage[n]["bw"] += task["bw"]
                node_usage[n]["time"] += task["exec_times"][n-1]
                break
    fitness = calculate_fitness(schedule)
    return schedule, fitness

ff_schedule, ff_fitness = first_fit()
print("\nFirst-Fit Schedule (Task → Node):")
for t_idx, node in enumerate(ff_schedule):
    print(f"Task {t_idx+1} → Node {node}")
print(f"First-Fit Fitness: {ff_fitness}")


# AG
ga_cpu_util, ga_mem_util = get_resource_utilization(best_schedule)
print(f"GA: {ga_cpu_util:.1f}% CPU, {ga_mem_util:.1f}% Memory")

#  First-Fit
ff_cpu_util, ff_mem_util = get_resource_utilization(ff_schedule)
print(f"First-Fit: {ff_cpu_util:.1f}% CPU, {ff_mem_util:.1f}% Memory")
