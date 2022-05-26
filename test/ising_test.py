import unittest

import numpy as np

from isingmc import IsingSite, IsingModel, IsingModel2D


class IsingModelTest(unittest.TestCase):
    def make_ising_chain(self, all_spin) -> IsingModel:
        sites = [IsingSite(spin) for spin in all_spin]
        model = IsingModel(sites)
        for i in range(len(sites)-1):
            model.add_edge(i, i+1)
        return model

    def make_random_ising(self, nsites: int, nedges: int) -> IsingModel:
        sites = [IsingSite(np.random.choice([-1, 1])) for _ in range(nsites)]
        model = IsingModel(sites)
        for _ in range(nedges):
            i, j = np.random.randint(nsites, size=2)
            try:
                model.add_edge(i, j)
            except ValueError:
                pass
        return model

    def test_energy_chain(self):
        all_spin = np.random.choice([-1, 1], size=10)
        chain = self.make_ising_chain(all_spin)
        energy = np.sum(-all_spin[:-1]*all_spin[1:])
        self.assertEqual(energy, chain.hamiltonian)

    def test_energy_random(self):
        model = self.make_random_ising(nsites=10, nedges=50)
        self.assertEqual(
            model.hamiltonian*2,
            sum([site.site_energy() for site in model.vertexes]))

    def test_flip(self):
        nsites = 10
        nedges = 50
        model = self.make_random_ising(nsites, nedges)
        for _ in range(2*nsites):
            model.flip(np.random.randint(nsites))
            self.assertEqual(model.hamiltonian, model.calculate_hamiltonian())

    def test_flip_2D(self):
        L = 6
        model = IsingModel2D(lattice_scale=L*L)
        for _ in range(int(1e4)):
            model.flip(np.random.randint(L*L))
            self.assertEqual(model.hamiltonian, model.calculate_hamiltonian())
    
    def _2D_ising_energy(self, state:np.ndarray):
        lattice_scale = state.shape[0]
        energy = 0
        for ii in range(lattice_scale):
            for jj in range(lattice_scale):
                site = state[ii, jj]
                # count number of  neighbors with periodic BC
                neighbors = state[(ii + 1) % lattice_scale, jj] + state[(ii - 1) % lattice_scale, jj] + \
                    state[ii, (jj + 1) % lattice_scale] + state[ii, (jj - 1) % lattice_scale]
                # account for double counting
                energy += -0.5 * site * neighbors
        return energy

    def test_2D_energy(self):
        L = 6
        model = IsingModel2D(lattice_scale=L)
        state_mat = np.array(
            [[model.vertexes[model.ij_to_i(i, j)].spin for j in range(L)] for i in range(L)])
        self.assertEqual(self._2D_ising_energy(state_mat), model.hamiltonian)
