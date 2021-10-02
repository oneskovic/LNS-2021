import matplotlib.pyplot as plt

from genetic_algorithm import GeneticAlgorithm
from evaluator import evaluate
import numpy as np

np.random.seed(42)

ga = GeneticAlgorithm(250)

for t in range(1,25):
    best_organism = ga.population[0]
    evaluate(best_organism, False, 400)
    print(f'Starting generation {t}')
    scores = ga.make_new_generation()
    max_score = scores.max()
    avg_score = scores.mean()

    print(f'Done max score: {scores.max()}, average score: {scores.mean()}')

plt.show()