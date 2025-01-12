import random

import matplotlib.pyplot as plt



class Organism:
    def __init__(self, min_n2_conc, max_co2_conc, mutation_prob):
        self.min_n2 = min_n2_conc
        self.max_co2 = max_co2_conc
        self.mutation_prob = mutation_prob
    
    def get_min_n2(self):
        return self.min_n2
    
    def get_max_co2(self):
        return self.max_co2
    
    def get_mutation_prob(self):
        return self.mutation_prob

    def breed(self, current_n2, current_co2):
        n2 = current_n2 - self.get_min_n2()
        co2 = (self.get_max_co2() - current_co2)/self.get_max_co2()
        rep = n2*co2
        if rep > self.get_mutation_prob():
            offspring = Organism(self.min_n2-0.1, self.max_co2+0.1, self.mutation_prob)
            return offspring
        else:
            return None


class Population:
    def __init__(self, min_n2_conc, max_co2_conc, mutation_prob, size):
        self.population = list()
        
        self.generate_population()
    
    def get_population(self):
        return self.population 

    def generate_population(self, min_n2_conc, max_co2_conc, mutation_prob, size):
        pop = self.get_population()
        for i in range(size):
            indiv = Organism(min_n2_conc, max_co2_conc, mutation_prob)
            pop[i] = indiv
        self.population = pop

    def step(self, current_n2, current_co2):
        new_pop = list()
        for indiv in self.get_population():
            offspring = indiv.breed(current_co2, current_n2)
            if offspring:
                new_pop.append(offspring)
        self.population = new_pop


class Environment:
    def __init__(self, n2_conc, co2_conc):
        self.current_n2 = n2_conc
        self.current_co2 = co2_conc

        self.population = None

    def introduce_population(self, pop_size, mutation):
        pass

    def increase_n2(self, increase):
        pass

    def increase_co2(self, increase):
        pass

    def decrease_n2(self, decrease):
        pass

    def decrease_co2(self, decrease):
        pass

    def time_step(self):
        pass


def simulate_breeding(length, start_pop, mutation,
                      start_n2, start_co2, change_frequency):
    pop_size = []

    return pop_size


def main():
    # simulate_breeding(100, 10, 0.5, 0.1, 0.1, 5)
    pass


if __name__ == '__main__':
    main()
