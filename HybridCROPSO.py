import numpy as np
from sklearn.cluster import KMeans
from copy import deepcopy


def calc_sse(centroids, labels, data):
    # Squared sum error
    distances = 0
    for i, c in enumerate(centroids):
        idx = np.where(labels == i)
        distances += np.sum((data[idx] - c) ** 2)
    return distances


def quantization_error(centroids, labels, data):
    # Quantization error
    error = 0
    for i, c in enumerate(centroids):
        idx = np.where(labels == i)
        dist = np.linalg.norm(data[idx] - c)
        dist /= len(idx)
        error += dist
    error /= len(centroids)
    return error


# SSE, Quantization error or another error funtion may be used as objective function


class Molecule:
    def __init__(self, n_cluster, data, KE, seed, use_kmeans=True):
        if use_kmeans:
            kmeans = KMeans(n_clusters=n_cluster, max_iter=60, random_state=seed)
            kmeans.fit(data)
            self.centroids = np.array(kmeans.cluster_centers_).copy()
        else:
            index = np.random.randint(0, len(data), n_cluster)
            self.centroids = np.array(data[index]).copy()

        self.n_cluster = n_cluster
        self.size = len(data[0])
        self.structure = self.centroids.flatten().copy()
        self.MinStruct = self.structure.copy()
        self.NHits = 0
        self.MHits = 0
        self.PE = self._potential_energy_function(data)
        self.KE = KE
        self.mPE = self.PE

        self.velocity = np.zeros_like(self.structure)

    def update(self, data, new=False):
        self.centroids = np.reshape(self.structure, (self.n_cluster, self.size)).copy()
        self.PE = self._potential_energy_function(data)
        if new:
            self.mPE = self.PE
            self.NHits = 0
            self.MHits = 0
            self.MinStruct = self.structure.copy()
        else:
            if self.PE < self.mPE:
                self.mPE = self.PE
                self.MHits = self.NHits
                self.MinStruct = self.structure.copy()

    def _potential_energy_function(self, data):
        # Objective function
        return calc_sse(self.centroids, self._predict(data), data)

    def _predict(self, data):
        # Predict data's cluster using minimum distance to centroid
        distances = []
        for c in self.centroids:
            dist = np.sum((data - c) * (data - c), axis=1)
            distances.append(dist)

        distances = np.array(distances)
        distances = np.transpose(distances)
        cluster = np.argmin(distances, axis=1)
        return cluster


class CRO:
    def __init__(
        self,
        n_clusters,
        n_molecules,
        data,
        max_generations,
        a,
        b,
        c1,
        c2,
        w,
        baseKE,
        MoleColl,
        KMeans=True,
        random_state=0,
        debug=0,
    ):
        self.n_clusters = n_clusters
        self.n_molecules = n_molecules
        self.data = data
        self.max_generations = max_generations
        self.KMeans = KMeans
        self.random_state = random_state

        self.a = a
        self.b = b
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.baseKE = baseKE
        self.MoleColl = MoleColl
        self.buffer = 0
        self.debug = debug
        self.molecules = []
        self.features = n_clusters * len(data[0])

        self.syn = 0
        self.iic = 0
        self.dec = 0
        self.oic = 0

        self.gbest_mPE = np.inf
        self.gbest_structure = None
        self._init_molecules()

    def _init_molecules(self):
        # population initialization
        for i in range(self.n_molecules):
            if i == 0 and self.KMeans:
                molecule = Molecule(self.n_clusters, self.data, self.baseKE, self.random_state, use_kmeans=True)
            else:
                molecule = Molecule(self.n_clusters, self.data, self.baseKE, self.random_state, use_kmeans=False)
            if molecule.mPE < self.gbest_mPE:
                self.gbest_mPE = molecule.mPE
                self.gbest_structure = deepcopy(molecule.MinStruct)
            self.molecules.append(molecule)

    def run(self):
        for i in range(self.max_generations):
            r = np.random.uniform()
            if r <= self.MoleColl and len(self.molecules) > 1:
                # Intermolecular reaction
                molecule_1, molecule_2 = np.random.choice(self.molecules, size=2, replace=False)
                if molecule_1.KE <= self.b and molecule_2.KE <= self.b and len(self.molecules) > self.n_molecules // 2:
                    self._synthesis(molecule_1, molecule_2)
                else:
                    self._intermolecular_ineffective_collision(molecule_1, molecule_2)
            else:
                # Unimolecular reaction
                molecule_1 = np.random.choice(self.molecules)
                if molecule_1.NHits - molecule_1.MHits > self.a:
                    self._decomposition(molecule_1)
                else:
                    self._onwall_ineffective_collision(molecule_1)
            self._check_mPE(i)
        if self.debug != 0:
            print("\nCollision counter:")
            print("Synthesis :", self.syn)
            print("Intermolecular ineffective collision :", self.iic)
            print("Decomposition :", self.dec)
            print("Onwall ineffective collision:", self.oic)
            print("Final population size:", len(self.molecules))
            print("\n")

    def _synthesis(self, molecule_1, molecule_2):
        # product obtains features at random from reactants
        self.syn += 1
        new_molecule = deepcopy(molecule_1)

        for i in range(self.features):
            if np.random.randint(0, 2) == 0:
                new_molecule.structure[i] = molecule_2.structure[i]

        new_molecule.update(self.data, new=True)
        new_molecule.KE = molecule_1.PE + molecule_1.KE + molecule_2.PE + molecule_2.KE - new_molecule.PE
        if new_molecule.KE > 0:
            self.molecules.remove(molecule_1)
            self.molecules.remove(molecule_2)
            self.molecules.append(new_molecule)
        else:
            molecule_1.NHits += 1
            molecule_2.NHits += 1

    def _intermolecular_ineffective_collision(self, molecule_1, molecule_2):
        # PSO particle search
        self.iic += 1
        molecule_1.NHits += 1
        molecule_2.NHits += 1

        new_molecule = [deepcopy(molecule_1), deepcopy(molecule_2)]

        for i in range(2):
            new_molecule[i].velocity = (
                self.c1 * np.random.random() * (new_molecule[i].MinStruct - new_molecule[i].structure)
                + self.c2 * np.random.random() * (self.gbest_structure - new_molecule[i].structure)
                + self.w * new_molecule[i].velocity
            )

            new_molecule[i].structure = new_molecule[i].structure + new_molecule[i].velocity
            new_molecule[i].update(self.data, new=False)

        total_energy_left = (
            molecule_1.PE + molecule_1.KE + molecule_2.PE + molecule_2.KE - new_molecule[0].PE - new_molecule[1].PE
        )

        if total_energy_left >= 0:
            delta = np.random.uniform()
            new_molecule[0].KE = total_energy_left * delta
            new_molecule[1].KE = total_energy_left * (1 - delta)
            self.molecules.remove(molecule_1)
            self.molecules.remove(molecule_2)
            self.molecules += new_molecule

    def _decomposition(self, molecule):
        # reactant gives features at random to each product. missing features are created randomly in the range [-1,1)
        self.dec += 1
        molecule.NHits += 1
        new_molecule = [deepcopy(molecule), deepcopy(molecule)]

        for i in range(self.features):
            new_molecule[np.random.randint(0, 2)].structure[i] = np.random.uniform(low=-1)

        new_molecule[0].update(self.data, new=True)
        new_molecule[1].update(self.data, new=True)
        total_energy_left = molecule.PE + molecule.KE - new_molecule[0].PE - new_molecule[1].PE

        if total_energy_left >= 0:
            delta = np.random.uniform()
            new_molecule[0].KE = total_energy_left * delta
            new_molecule[1].KE = total_energy_left * (1 - delta)
        elif total_energy_left + self.buffer >= 0:
            m1, m2, m3, m4 = np.random.uniform(size=4)
            new_molecule[0].KE = (total_energy_left + self.buffer) * (m1 * m2)
            new_molecule[1].KE = (total_energy_left + self.buffer - new_molecule[0].KE) * (m3 * m4)
            self.buffer += total_energy_left - new_molecule[0].KE - new_molecule[1].KE
        else:
            molecule.NHits += 1
            return None
        self.molecules.remove(molecule)
        self.molecules += new_molecule

    def _onwall_ineffective_collision(self, molecule):
        # PSO particle search
        self.oic += 1
        molecule.NHits += 1
        new_molecule = deepcopy(molecule)

        new_molecule.velocity = (
            self.c1 * np.random.random() * (new_molecule.MinStruct - new_molecule.structure)
            + self.c2 * np.random.random() * (self.gbest_structure - new_molecule.structure)
            + self.w * new_molecule.velocity
        )

        new_molecule.structure = new_molecule.structure + new_molecule.velocity

        new_molecule.update(self.data, new=False)
        total_energy_left = molecule.PE + molecule.KE - new_molecule.PE

        if total_energy_left >= 0:
            delta = np.random.uniform()
            new_molecule.KE = total_energy_left * delta
            self.buffer += total_energy_left - new_molecule.KE
            self.molecules.remove(molecule)
            self.molecules.append(new_molecule)

    def _check_mPE(self, generation):
        best_PE = np.inf
        best_structure = None
        for molecule in self.molecules:
            if molecule.mPE < best_PE:
                best_PE = molecule.mPE
                best_structure = deepcopy(molecule.structure)
        if best_PE < self.gbest_mPE:
            self.gbest_mPE = best_PE
            self.gbest_structure = deepcopy(best_structure)
            # print('Best PE =', self.gbest_mPE, 'Generation', generation)
        if generation % 100 == 0:
            # print("\tbest PE and structure : \t\t", best_PE, best_structure)
            # print("\t\tglobal best PE and structure: \t", self.gbest_mPE, self.gbest_structure)
            # print('Best PE =', best_PE, '\tGlobal best PE', self.gbest_mPE, '\tGeneration', generation + 1)
            pass
