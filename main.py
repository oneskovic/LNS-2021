from genetic_algorithm import GeneticAlgorithm
from evaluator import evaluate
from springed_body import SpringedBody

# test_organism = SpringedBody()
# test_organism.add_node()
# test_organism.add_node()
# test_organism.add_node()
# for _ in range(100):
#     test_organism.make_random_connection()
# evaluate(test_organism, True)

hparams = dict({
    'population_size':1000
})
ga = GeneticAlgorithm(hparams)

for t in range(1000):
    if (t+1)%5 == 0:
        best_organism = ga.population[0]
        evaluate(best_organism, True)
    print(f'Starting generation {t}')
    scores = ga.step()
    print(f'Done max score: {scores.max()}, average score: {scores.mean()}')