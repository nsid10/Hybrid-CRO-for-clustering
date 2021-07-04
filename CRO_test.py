import numpy as np
from copy import deepcopy


class Molecule:
    def __init__(self, KE):
        self.structure = np.random.uniform(-10, 10, size=2)
        self.MinStruct = self.structure.copy()
        self.NHits = 0
        self.MHits = 0
        self.PE = self._potential_energy_function()
        self.KE = KE
        self.mPE = self.PE

    def update(self, new=False):
        self.PE = self._potential_energy_function()
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

    def _potential_energy_function(self):
        # Objective function
        x1, x2 = self.structure.copy()
        y1, y2 = (x1) * (x1 - 1) * (x1 - 3) * (x1 - 5), (x2 + 3) * (x2 + 2) * (x2 + 1) * (x2 - 1)
        return y1 + y2


class CRO:
    def __init__(self, n_molecules, max_generations, a, b, baseKE, MoleColl, debug=0):
        self.n_molecules = n_molecules
        self.max_generations = max_generations

        self.a = a
        self.b = b
        self.baseKE = baseKE
        self.MoleColl = MoleColl
        self.buffer = 0
        self.debug = debug
        self.molecules = []
        self.features = 2

        self.syn = 0
        self.iic = 0
        self.dec = 0
        self.oic = 0

        self.gbest_mPE = np.inf
        self.gbest_structure = None
        self._init_molecules()

    def _init_molecules(self):
        for i in range(self.n_molecules):
            molecule = Molecule(self.baseKE)
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
                if molecule_1.KE <= self.b and molecule_2.KE <= self.b:
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
        if self.debug == 1:
            print("\nCollision counter:")
            print("Synthesis :", self.syn)
            print("Intermolecular ineffective collision :", self.iic)
            print("Decomposition :", self.dec)
            print("Onwall ineffective collision:", self.oic)
            print("Final population size:", len(self.molecules))
            print("\n")

    def _synthesis(self, molecule_1, molecule_2):
        self.syn += 1
        new_molecule = deepcopy(molecule_1)

        for i in range(self.features):
            if np.random.randint(0, 2) == 0:
                new_molecule.structure[i] = molecule_2.structure[i]

        new_molecule.update(True)
        new_molecule.KE = molecule_1.PE + molecule_1.KE + molecule_2.PE + molecule_2.KE - new_molecule.PE
        if new_molecule.KE > 0:
            self.molecules.remove(molecule_1)
            self.molecules.remove(molecule_2)
            self.molecules.append(new_molecule)
        else:
            molecule_1.NHits += 1
            molecule_2.NHits += 1

    def _intermolecular_ineffective_collision(self, molecule_1, molecule_2):
        self.iic += 1
        molecule_1.NHits += 1
        molecule_2.NHits += 1

        new_molecule = [deepcopy(molecule_1), deepcopy(molecule_2)]

        for i in range(2):
            index1, index2 = np.random.randint(0, self.features, size=2)
            new_molecule[i].structure[index1], new_molecule[i].structure[index2] = (
                new_molecule[i].structure[index2],
                new_molecule[i].structure[index1],
            )

        new_molecule[0].update(False)
        new_molecule[1].update(False)
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
        self.dec += 1
        molecule.NHits += 1
        new_molecule = [deepcopy(molecule), deepcopy(molecule)]

        for i in range(self.features):
            new_molecule[np.random.randint(0, 2)].structure[i] = np.random.uniform(-10, 10)

        new_molecule[0].update(True)
        new_molecule[1].update(True)
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
        self.oic += 1
        molecule.NHits += 1
        new_molecule = deepcopy(molecule)

        index1, index2 = np.random.randint(0, self.features, size=2)
        new_molecule.structure[index1], new_molecule.structure[index2] = (
            new_molecule.structure[index2],
            new_molecule.structure[index1],
        )

        new_molecule.update(False)
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


box = CRO(n_molecules=25, max_generations=40000, a=80, b=15, baseKE=800, MoleColl=0.3, debug=0)
box.run()

box = deepcopy(box.gbest_structure)

print(box)
print("error:", abs(box[0] - 4.254), abs(box[1] - 0.326))
# solution is [4.254, 0.326]
