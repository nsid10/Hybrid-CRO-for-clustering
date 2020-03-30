import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def calc_sse(centroids: np.ndarray, labels: np.ndarray, data: np.ndarray):
    distances = 0
    for i, c in enumerate(centroids):
        idx = np.where(labels == i)
        dist = np.sum((data[idx] - c)**2)
        distances += dist
    return distances


class PSO:
    def __init__(self, n_cluster, n_particles, data, hybrid=True, max_iter=200, random_state=0, print_debug=10):
        self.n_cluster = n_cluster
        self.n_particles = n_particles
        self.data = data
        self.max_iter = max_iter
        self.particles = []
        self.hybrid = hybrid
        self.random_state = random_state

        self.print_debug = print_debug
        self.gbest_centroids = None
        self.gbest_sse = np.inf
        self._init_particles()

    def _init_particles(self):
        for i in range(self.n_particles):
            particle = None
            if i == 0 and self.hybrid:
                particle = Particle(self.n_cluster, self.data, self.random_state, use_kmeans=True)
            else:
                particle = Particle(self.n_cluster, self.data, self.random_state, use_kmeans=False)
            if particle.best_sse < self.gbest_sse:
                self.gbest_centroids = particle.centroids.copy()
                self.gbest_sse = particle.best_sse
            self.particles.append(particle)

    def run(self):
        # print('Initial global best score', self.gbest_score)
        history = []
        for i in range(self.max_iter):
            for particle in self.particles:
                particle.update(self.gbest_centroids, self.data)
                # print(i, particle.best_score, self.gbest_score)
            for particle in self.particles:
                if particle.best_sse < self.gbest_sse:
                    self.gbest_centroids = particle.centroids.copy()
                    self.gbest_sse = particle.best_sse
            history.append(self.gbest_sse)
            if self.print_debug != 0 and i % self.print_debug == 0:
                print('Iteration {:04d}/{:04d} current gbest score {:.18f}'.format(
                    i + 1, self.max_iter, self.gbest_sse))
        # print('Finish with gbest score {:.18f}'.format(self.gbest_score))
        return history


class Particle:
    def __init__(self, n_cluster, data, seed, use_kmeans=True, w=0.4, c1=1.49, c2=1.49):
        index = np.random.choice(list(range(len(data))), n_cluster)
        self.centroids = data[index].copy()
        if use_kmeans:
            kmeans = KMeans(n_clusters=n_cluster, max_iter=60, random_state=seed)
            kmeans.fit(data)
            self.centroids = kmeans.cluster_centers_.copy()
        self.best_position = self.centroids.copy()
        self.best_sse = calc_sse(self.centroids, self._predict(data), data)
        self.velocity = np.zeros_like(self.centroids)
        self._w = w
        self._c1 = c1
        self._c2 = c2

    def update(self, gbest_position: np.ndarray, data: np.ndarray):
        self._update_velocity(gbest_position)
        self._update_centroids(data)

    def _update_velocity(self, gbest_position: np.ndarray):
        """Update velocity based on old value, cognitive component, and social component
        """
        v_old = self._w * self.velocity
        cognitive_component = self._c1 * np.random.random() * (self.best_position - self.centroids)
        social_component = self._c2 * np.random.random() * (gbest_position - self.centroids)
        self.velocity = v_old + cognitive_component + social_component

    def _update_centroids(self, data: np.ndarray):
        self.centroids = self.centroids + self.velocity
        sse = calc_sse(self.centroids, self._predict(data), data)
        if sse < self.best_sse:
            self.best_sse = sse
            self.best_position = self.centroids.copy()

    def _predict(self, data: np.ndarray) -> np.ndarray:
        """Predict new data's cluster using minimum distance to centroid
        """
        distance = self._calc_distance(data)
        cluster = self._assign_cluster(distance)
        return cluster

    def _calc_distance(self, data: np.ndarray) -> np.ndarray:
        """Calculate distance between data and centroids
        """
        distances = []
        for c in self.centroids:
            distance = np.sum((data - c) * (data - c), axis=1)
            distances.append(distance)

        distances = np.array(distances)
        distances = np.transpose(distances)
        return distances

    def _assign_cluster(self, distance: np.ndarray) -> np.ndarray:
        """Assign cluster to data based on minimum distance to centroids
        """
        cluster = np.argmin(distance, axis=1)
        return cluster


if __name__ == "__main__":
    pass
