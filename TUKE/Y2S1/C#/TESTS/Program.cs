using System;
using System.Collections.Generic;
using System.Linq;

class Program
{
    // Define the objective function
    static int ObjectiveFunction(List<int> solution)
    {
        // Added const qualifier
        // TODO: Implement your objective function here
        // The objective function should evaluate
        // the quality of a given solution and
        // return a numerical value representing
        // the solution's fitness
        // Example: return std::accumulate(solution.begin(),
        // solution.end(), 0);
        return solution.Sum();
    }

    // Define the neighborhood function
    static List<List<int>> GetNeighbors(List<int> solution)
    {
        // Added const qualifier
        List<List<int>> neighbors = new List<List<int>>();
        for (int i = 0; i < solution.Count; i++)
        {
            for (int j = i + 1; j < solution.Count; j++)
            {
                List<int> neighbor = new List<int>(solution);
                neighbor[i] = solution[j];
                neighbor[j] = solution[i];
                neighbors.Add(neighbor);
            }
        }
        return neighbors;
    }

    // Define the Tabu Search algorithm
    static List<int> TabuSearch(List<int> initial_solution,
                                int max_iterations,
                                int tabu_list_size)
    {

        // Added const qualifier
        List<int> best_solution = initial_solution;
        List<int> current_solution = initial_solution;
        List<List<int>> tabu_list = new List<List<int>>();
        for (int iter = 0; iter < max_iterations; iter++)
        {
            List<List<int>> neighbors = GetNeighbors(current_solution);
            List<int> best_neighbor = new List<int>();
            int best_neighbor_fitness = int.MaxValue;
            foreach (List<int> neighbor in neighbors)
            {
                if (!tabu_list.Contains(neighbor))
                {
                    int neighbor_fitness = ObjectiveFunction(neighbor);
                    if (neighbor_fitness < best_neighbor_fitness)
                    {
                        best_neighbor = neighbor;
                        best_neighbor_fitness = neighbor_fitness;
                    }
                }
            }
            if (best_neighbor.Count == 0)
            {
                // No non-tabu neighbors found,
                // terminate the search
                break;
            }
            current_solution = best_neighbor;
            tabu_list.Add(best_neighbor);
            if (tabu_list.Count > tabu_list_size)
            {
                // Remove the oldest entry from the
                // tabu list if it exceeds the size
                tabu_list.RemoveAt(0);
            }
            if (ObjectiveFunction(best_neighbor) < ObjectiveFunction(best_solution))
            {
                // Update the best solution if the
                // current neighbor is better
                best_solution = best_neighbor;
            }
        }
        return best_solution;
    }
    static void Main()
    {
        // Example usage
        // Provide an initial solution
        List<int> initial_solution = new List<int> { 59, 1, 0, 345, 999 };
        int max_iterations = 1;
        int tabu_list_size = 1;
        List<int> best_solution = TabuSearch(
            initial_solution, max_iterations, tabu_list_size);
        Console.Write("Best solution:");
        foreach (int val in best_solution)
        {
            Console.Write(" " + val);
        }
        Console.WriteLine();
        Console.WriteLine("Best solution fitness: " +
            ObjectiveFunction(best_solution));
        Console.ReadLine();
    }

}
