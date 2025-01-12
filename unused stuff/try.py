import random

def tabu_search(solution, neighbors, cost_function, tenure, max_iterations):
    tabu_list = []
    best_solution = solution
    best_cost = cost_function(solution)

    for iteration in range(max_iterations):
        neighborhood = [neighbor for neighbor in neighbors(solution) if neighbor not in tabu_list]
        if not neighborhood:
            break

        neighborhood_costs = [(neighbor, cost_function(neighbor)) for neighbor in neighborhood]
        neighborhood_costs.sort(key=lambda x: x[1])
        best_candidate, best_candidate_cost = neighborhood_costs[0]

        if best_candidate_cost < best_cost:
            best_solution = best_candidate
            best_cost = best_candidate_cost

        tabu_list.append(solution)
        if len(tabu_list) > tenure:
            tabu_list.pop(0)

        solution = best_candidate

    return best_solution

# Example usage:
def sample_cost_function(solution):
    return sum(solution)

def sample_neighbors(solution):
    neighbors = []
    for i in range(len(solution)):
        neighbor = solution[:]
        neighbor[i] = random.randint(0, 100)
        neighbors.append(neighbor)
    return neighbors

initial_solution = [random.randint(0, 100) for _ in range(10)]
best_solution = tabu_search(initial_solution, sample_neighbors, sample_cost_function, tenure=5, max_iterations=100)

print("Best Solution:", best_solution)
print("Best Cost:", sample_cost_function(best_solution))
