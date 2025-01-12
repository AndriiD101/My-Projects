# Tabu Search implementation for Travelling Salesman Problem
# test: python tabu_search.py -f data.txt -i numberOfIterations -s sizeOfTabuSearch

import argparse
import copy
import random


def generate_neighbours(path):
    # create a dictionary for neighbors
    dict_of_neighbours = {}
    # read data from the file
    with open(path) as data_file:
        # read lines
        content = data_file.readlines()
        # split lines into start node, end node, and distance between them (starting from line 2 because the first line is the list of all nodes)
        for i in range(1, len(content)):
            start_node = content[i].split()[0]
            finish_node = content[i].split()[1]
            distance = content[i].split()[2]
            # check if nodes are already in the dictionary
            if start_node not in dict_of_neighbours:
                # add to the dictionary with the neighbor node and distance between them
                temp = [[finish_node, distance]]
                dict_of_neighbours[start_node] = temp
            # if node already in dictionary, add neighbor node and distance under the key (node in the dictionary)
            else:
                dict_of_neighbours[start_node].append([finish_node, distance])
            # check if end node is in dictionary
            if finish_node not in dict_of_neighbours:
                # if not, add end node with start node and distance parameters
                temp = [[start_node, distance]]
                dict_of_neighbours[finish_node] = temp
            # if node already in dictionary, add neighbor node and distance under the key
            else:
                dict_of_neighbours[finish_node].append([start_node, distance])
    return dict_of_neighbours

def generate_initial_solution(node, dict_of_neighbours): # add node destination node
    # variables
    # read the first line from the file and split it to choose a node
    # with open(path) as data_file:
    #     start_line = data_file.readline().split(' ')
    # choose a random start node
    # start_node = start_line[0]
    start_node = node
    # set the end node to the start node to form a cycle and return to the start node
    finish_node = start_node
    initial_solution = []
    current_node = start_node
    total_distance = 0

    # creation of initial solution
    while current_node not in initial_solution:
        min_distance = 10000
        for k in dict_of_neighbours[current_node]:
            current_distance = k[1]
            current_neighbour = k[0]
            # compare distance and check if neighbor is already in initial_solution
            if int(current_distance) < int(min_distance) and current_neighbour not in initial_solution:
                min_distance = current_distance
                nearest_node = current_neighbour
        # add node to initial solution and calculate distance
        initial_solution.append(current_node)
        total_distance += int(min_distance)
        current_node = nearest_node

    initial_solution.append(finish_node)

    # calculate distance from second to last element to last one
    # calculate distance between the penultimate and last nodes
    position = 0
    for k in dict_of_neighbours[initial_solution[-2]]:
        if k[0] == start_node:
            break
        position += 1
    total_distance = (total_distance + int(dict_of_neighbours[initial_solution[-2]][position][1]) - 10000)
    
    # return initial solution and its distance
    return initial_solution, total_distance


def find_neighborhood(current_solution, neighbors_dict):
    # The function implementation remains the same as previously defined.
    neighborhood = []
    # take from first to second last element (since it's a cycle, the first and last elements are the same)
    for node in current_solution[1:-1]:
        original_index = current_solution.index(node)
        # --||--
        for neighbor_node in current_solution[1:-1]:
            # check to ensure we're not swapping identical nodes
            if node == neighbor_node:
                continue
            # swap nodes to create a new solution
            temp_solution = copy.deepcopy(current_solution)
            temp_solution[original_index] = neighbor_node
            temp_solution[current_solution.index(neighbor_node)] = node

            total_distance = 0
            # this entire block is for evaluating the solution
            for i in range(len(temp_solution) - 1):
                current_node = temp_solution[i]
                next_node = temp_solution[i + 1]
                
                for neighbor in neighbors_dict[current_node]:
                    if neighbor[0] == next_node:
                        total_distance += int(neighbor[1])
            
            temp_solution.append(total_distance)
            # if solution is NOT already in neighborod, add it
            if temp_solution not in neighborhood:
                neighborhood.append(temp_solution)
    # sort by the last element
    neighborhood.sort(key=lambda x: x[-1])
    return neighborhood


def tabu_search(
    first_solution, distance_of_first_solution, dict_of_neighbours, iters, size
):
    # Iteration counter starts from 1, as the zero iteration is the initial solution
    count = 1
    initial_solution = first_solution
    tabu_list = []
    best_cost = distance_of_first_solution
    best_solution_ever = initial_solution

    # Perform iterations until the set number is reached
    while count <= iters:
        # Generate possible combinations
        neighborhood = find_neighborhood(initial_solution, dict_of_neighbours)
        # start from zero since the neighborhood list is sorted, and the "best" solution will be the first
        index_of_best_solution = 0
        best_solution = neighborhood[index_of_best_solution]
        # Index to access the total cost of the best solution
        best_cost_index = len(best_solution) - 1

        # Assume no better solution is found
        # Flag to find an unused solution in the tabu list
        found = False
        # While no solution that’s not in the tabu list is found
        while not found:
            i = 0
            # Find the first difference between current and best solution
            while i < len(best_solution):
                if best_solution[i] != initial_solution[i]:
                    first_exchange_node = best_solution[i]
                    second_exchange_node = initial_solution[i]
                    break
                i = i + 1

            # Check if the pair is not in the tabu list
            if [first_exchange_node, second_exchange_node] not in tabu_list and [
                second_exchange_node,
                first_exchange_node,
            ] not in tabu_list:
                # Add swap to the tabu list
                tabu_list.append([first_exchange_node, second_exchange_node])
                # Set flag to True since a solution has been added to the tabu list
                found = True
                # Update the current solution and remove distance
                initial_solution = best_solution[:-1]
                # Set the cost of the best neighbor solution
                cost = neighborhood[index_of_best_solution][best_cost_index]
                # If the new cost is lower than the current best, update the best solution
                if cost < best_cost:
                    best_cost = cost
                    best_solution_ever = initial_solution
            else:
                # Move to the next best solution in the neighborhood
                index_of_best_solution = index_of_best_solution + 1
                best_solution = neighborhood[index_of_best_solution]

        # If the tabu list exceeds the set size, remove the oldest element
        if len(tabu_list) >= size:
            tabu_list.pop(0)

        # Increment iteration counter
        count = count + 1

    # Return the best solution found and its total cost
    return best_solution_ever, best_cost



def main(args=None):
    dict_of_neighbours = generate_neighbours(args.File)

    first_solution, distance_of_first_solution = generate_initial_solution(
        args.Node, dict_of_neighbours
    )

    best_sol, best_cost = tabu_search(
        first_solution,
        distance_of_first_solution,
        dict_of_neighbours,
        args.Iterations,
        args.Size,
    )

    print(f"Best solution: {best_sol}, with total distance: {best_cost}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tabu Search")
    parser.add_argument(
        "-f",
        "--File",
        type=str,
        help="Path to the file containing the data",
        required=True,
    )
    parser.add_argument(
        "-i",
        "--Iterations",
        type=int,
        help="How many iterations the algorithm should perform",
        required=True,
    )
    parser.add_argument(
        "-s", "--Size", type=int, help="Size of the tabu list", required=True
    )
    parser.add_argument(
        "-n", "--Node", type=str, help="Starting node", required=True 
    )
    
    # Pass the arguments to main method
    main(parser.parse_args())
