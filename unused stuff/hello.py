def get_neighbors(solution):
    neighbors = []
    for i in range(len(solution)):
        neighbor = solution.copy()
        neighbor[i] = (neighbor[i] + 1) % 2
        neighbors.append(neighbor)
    return neighbors

def is_tabu(neighbor, tabu_list):
    neighbor_hash = hash_function(neighbor, tabu_list)
    return any(tabu_list[level][neighbor_hash] == 1 for level in range(len(tabu_list)))

def hash_function(solution, tabu_list):
    return sum(solution) % len(tabu_list[0])

def objective_function(solution):
    return sum(solution)

def tabu_search(solution, tabu_list, non_improvement_limit, tabu_levels, max_iterations):
    tabu_initialized = False
    best_solution = solution
    iteration = 1
    no_improvement_iterations = 0
    neighbor_change_iterations = 0
    
    if not tabu_initialized:
        for level in range(tabu_levels):
            for index in range(len(tabu_list[level])):
                tabu_list[level][index] = 0
        tabu_initialized = True
    
    while no_improvement_iterations <= non_improvement_limit:
        best_neighbor = None
        for neighbor in get_neighbors(solution):
            if not is_tabu(neighbor, tabu_list):
                if best_neighbor is None or objective_function(neighbor) < objective_function(best_neighbor):
                    best_neighbor = neighbor
        
        if best_neighbor is None:
            break
        
        if objective_function(best_neighbor) < objective_function(solution):
            solution = best_neighbor
            best_solution = best_neighbor
            no_improvement_iterations = 0
            neighbor_change_iterations = 0
        else:
            no_improvement_iterations += 1
            neighbor_change_iterations += 1
        
        if neighbor_change_iterations > non_improvement_limit:
            iteration += 1
            neighbor_change_iterations = 0
        
        if iteration > max_iterations:
            break
        
        for level in range(tabu_levels):
            tabu_list[level][hash_function(best_neighbor, tabu_list)] = 1
        
        iteration += 1
    
    return best_solution

def main():
    initial_solution = [0, 1, 1, 1, 1, 1, 0, 1]  # Initial solution (potential issue with binary assumption)
    tabu_levels = 8  # Number of tabu list levels
    max_iterations = 10  # Maximum number of iterations
    non_improvement_limit = 10  # Limit for non-improving iterations
    tabu_list = [[0] * 5 for _ in range(tabu_levels)]  # Initialize tabu list
    
    best_solution = tabu_search(initial_solution, tabu_list, non_improvement_limit, tabu_levels, max_iterations)
    
    print("Best solution found:", best_solution)

if __name__ == "__main__":
    main()
