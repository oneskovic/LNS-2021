import os
import time
import pickle

from genetic_algorithm import GeneticAlgorithm
from evaluator import Evaluator
from springed_body import SpringedBody
from mpi4py import MPI
import subprocess
import sys
import numpy as np
from typing import List

import faulthandler
faulthandler.enable()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
seed = 854525411


def master():
    hparams = dict({
        'population_size': 1000,
        'random_seed': seed
    })
    ga = GeneticAlgorithm(hparams)

    for t in range(1000):
        if (t + 1) % 500 == 0:
            best_organism = ga.population[0]
            evaluator = Evaluator()
            evaluator.evaluate(best_organism, True)
        print(f'Starting generation {t}', flush=True)
        scores = ga.step()
        print(f'Done max score: {scores.max()}, average score: {scores.mean()}', flush=True)
        max_nodes = -1
        for organism in ga.population:
            max_nodes = max(max_nodes, len(organism.nodes))
        print(f'Max nodes: {max_nodes}', flush=True)


def slave():
    while True:
        organisms: List[SpringedBody] = comm.recv(source=0, tag=0)
        evaluator = Evaluator()
        scores = np.array([])
        for organism in organisms:
            pickle.dump(organism, open(f'current_organism{rank}.pk1', "wb"))
            scores = np.append(scores, evaluator.evaluate(organism, False))
        comm.Send([scores, MPI.FLOAT], dest=0, tag=1)


if os.getenv("IN_MPI") is None:
    env = os.environ.copy()
    thread_cnt = os.cpu_count()
    env.update(
        IN_MPI="1",
        THREAD_CNT=str(thread_cnt-2)
    )
    subprocess.call(["mpiexec", "-np", str(thread_cnt-2), sys.executable] + ['-Xfaulthandler'] + ['-u'] + sys.argv, env=env)
else:
    if rank == 0:
        print(f'Master at: {os.getpid()}', flush=True)
        master()
    else:
        slave()
