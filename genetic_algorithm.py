from springed_body import Organism
import numpy as np
from evaluator import evaluate
from copy import deepcopy
from typing import List
import operator


class GeneticAlgorithm:
    def __init__(self, pop_size):
        self.population_size = pop_size
        self.population = []
        self.init_population()

    def init_population(self):
        self.population = [Organism() for _ in range(self.population_size)]
        for organism in self.population:
            node_cnt = 4
            for _ in range(node_cnt):
                organism.add_node()

        self.evaluate_population()
        self.population.sort(key=operator.attrgetter('score'), reverse=True)

    def evaluate_population(self):
        for organism in self.population:
            organism.score = evaluate(organism, False)

    def mutate_organisms(self, organisms: List[Organism]):
        for organism in organisms:
            mutation_type = np.random.choice([0, 1, 2, 3, 4], p=[0.0, 0.25, 0.25, 0.25, 0.25])
            if mutation_type == 0:
                organism.add_node()
            elif mutation_type == 1:
                organism.make_random_connection()
            elif mutation_type == 2:
                organism.mutate_mass()
            elif mutation_type == 3:
                organism.mutate_friction()
            else:
                organism.mutate_spring_damping()


    def make_offspring(self, count, population):
        best_in_tournament = np.random.randint(0,len(population),(count,8)).min(axis=1)
        offspring = []
        for i in range(count):
            new_organism = deepcopy(population[best_in_tournament[i]])
            offspring.append(new_organism)
        self.mutate_organisms(offspring)
        return offspring

    def make_new_generation(self):
        elite_cnt = int(self.population_size*0.2)
        remove_cnt = int(self.population_size*0.2)
        new_population = self.population[:elite_cnt]        # Uzmi "elite_cnt" najboljih i dodaj ih u new_population

        del self.population[-remove_cnt:]                   # Obrisi poslednjih "remove_cnt" iz self.population

        offspring = self.make_offspring(self.population_size - elite_cnt, self.population)      # Napravi novu decu
        new_population += offspring                         # Dodaj decu na new_population
        self.population = new_population

        self.evaluate_population()
        self.population.sort(key=operator.attrgetter('score'), reverse=True)
        return np.array([org.score for org in self.population])
