from springed_body import SpringedBody
import numpy as np
from evaluator import evaluate
from copy import deepcopy
from typing import List
import operator


class GeneticAlgorithm:
    def __init__(self, hparams):
        self.hparams = hparams
        self.population = []
        self.init_population()

    def init_population(self):
        self.population = [SpringedBody() for _ in range(self.hparams['population_size'])]
        for organism in self.population:
            node_cnt = np.random.randint(2, 5)
            for _ in range(node_cnt):
                organism.add_node()



        self.evaluate_population()
        self.population.sort(key=operator.attrgetter('score'), reverse=True)

    def evaluate_population(self):
        for organism in self.population:
            organism.score = evaluate(organism, False)

    def mutate_organisms(self, organisms: List[SpringedBody]):
        for organism in organisms:
            mutation_type = np.random.choice([0, 1, 2, 3, 4], p=[0.2, 0.2, 0.2, 0.2, 0.2])
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

    def step(self):
        elite_cnt = int(len(self.population)*0.2)
        remove_cnt = int(len(self.population)*0.2)
        new_population = self.population[:elite_cnt]

        del self.population[-remove_cnt:]

        offspring = self.make_offspring(self.hparams['population_size'] - elite_cnt, self.population)
        new_population += offspring
        self.population = new_population

        self.evaluate_population()
        self.population.sort(key=operator.attrgetter('score'), reverse=True)
        return np.array([org.score for org in self.population])
