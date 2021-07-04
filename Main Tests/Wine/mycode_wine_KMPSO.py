import numpy as np
import pandas as pd
from time import time
from PSO import *
from utilities import *

t1 = time()
sum_scores = [0, 0, 0]
runs = 40

for seed in range(runs):
    np.random.seed(seed**2)
    data = pd.read_csv('wine.txt', sep=',', header=None)
    data = data.sample(frac=1, random_state=seed).reset_index(drop=True)

    y_data = data[0]
    y_data = y_data.values

    x_data = data.drop([0], axis=1)
    x_data = x_data.values
    x_data = normalize(x_data)

    clusters = 3

    pso = PSO(n_cluster=clusters,
              n_particles=10,
              data=x_data,
              hybrid=True,
              max_iter=200,
              random_state=seed,
              print_debug=0)
    pso.run()

    centroids = pso.gbest_centroids.copy()

    labels = []
    for y in y_data:
        if y == 1:
            labels.append(0)
        elif y == 2:
            labels.append(1)
        elif y == 3:
            labels.append(2)

    labels = np.array(labels)

    scores = metrics(x_data, centroids, labels)

    sum_scores[0] += scores[0]
    sum_scores[1] += scores[1]
    sum_scores[2] += scores[2]

print(f'Accuracy =\t {sum_scores[0] / runs}')
print(f'F-measure =\t {sum_scores[1] / runs}')
print(f'SSE =\t {sum_scores[2] / runs}')

t2 = time()
print(t2 - t1)
