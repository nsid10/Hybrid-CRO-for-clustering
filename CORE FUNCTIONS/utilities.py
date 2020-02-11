import numpy as np
import pandas as pd
from copy import deepcopy
from math import factorial
from sklearn.metrics import accuracy_score, f1_score


def normalize(x):
    # Mean normalization. Scale to -1 to 1
    return (x - x.mean(axis=0)) / (x.max(axis=0) - x.min(axis=0))


def squared_sum_error(centroids, labels, data):
    # Squared sum error
    distances = 0
    for i, c in enumerate(centroids):
        idx = np.where(labels == i)
        distances += np.sum((data[idx] - c)**2)
    return distances


def metrics(data, centroids, labels):
    distances = []
    for c in centroids:
        dist = np.sum((data - c) * (data - c), axis=1)
        distances.append(dist)

    distances = np.array(distances)
    distances = np.transpose(distances)
    prediction = np.argmin(distances, axis=1)
    final = np.zeros_like(labels)

    clusters, classes = len(centroids), len(set(labels))

    for cl in range(clusters):
        block = np.nonzero(prediction == cl)
        count = [0 for x in range(classes)]
        for g in range(classes):
            box = np.transpose(np.nonzero(labels[block] == g))
            count[g] = len(box)
        final[block] = np.argmax(count)

    acc = accuracy_score(labels, final)
    f1 = f1_score(labels, final, average='macro')
    sse = squared_sum_error(centroids, final, data)
    return acc, f1, sse
