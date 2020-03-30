import numpy as np
import pandas as pd
from time import time
from HybridCRO import *
from utilities import *

t1 = time()
sum_scores = [0, 0, 0]
runs = 30

for seed in range(runs):
    data = pd.read_csv('breastcancer.txt', sep=',', header=None)
    data = data.sample(frac=1, random_state=seed).reset_index(drop=True)

    y_data = data[0]
    y_data = y_data.values

    x_data = data.drop([0], axis=1)
    x_data = x_data.values
    x_data = normalize(x_data)

    clusters = 2

    kbox = KMeans(n_clusters=clusters, max_iter=60, random_state=seed)
    kbox.fit(x_data)
    centroids = np.array(kbox.cluster_centers_).copy()

    labels = []
    for y in y_data:
        if y == 'M':
            labels.append(0)
        elif y == 'B':
            labels.append(1)

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
