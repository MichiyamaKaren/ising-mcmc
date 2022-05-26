import numpy as np

from ..graph import Vertex, Edge, Graph

from typing import List, Tuple, Optional, MutableSet


class IsingSite(Vertex):
    def __init__(self, spin: int,
                 edges: Optional[MutableSet['IsingInteraction']] = None) -> None:
        super().__init__(edges)
        self.edges: 'MutableSet[IsingInteraction]'

        self.spin = spin

    def flip(self) -> None:
        self.spin *= -1

    def site_energy(self) -> float:
        return sum([edge.interaction() for edge in self.edges])


class IsingInteraction(Edge):
    def __init__(self, headvex: IsingSite, tailvex: IsingSite) -> None:
        super().__init__(headvex, tailvex)
        self.headvex: IsingSite
        self.tailvex: IsingSite

    def interaction(self) -> float:
        return -self.headvex.spin*self.tailvex.spin


class IsingModel(Graph):
    def __init__(self, vertexes: List[IsingSite],
                 edges: Optional[List[IsingInteraction]] = None) -> None:
        super().__init__(vertexes, edges)
        self.vertexes: List[IsingSite]
        self.edges: List[IsingInteraction]

        self._hamiltonian = None

    def _create_edge(self, headvex: IsingSite, tailvex: IsingSite) -> Edge:
        return IsingInteraction(headvex, tailvex)

    def add_edge(self, headvex_i: int, tailvex_i: int) -> None:
        if headvex_i == tailvex_i:
            raise ValueError('Cannot add self interaction in Ising model')

        super().add_edge(headvex_i, tailvex_i)

        # after changed topology structure of sites, previously calculated energy is invalid now
        self._hamiltonian = None

    def calculate_hamiltonian(self) -> float:
        return sum([edge.interaction() for edge in self.edges])

    @property
    def hamiltonian(self):
        if self._hamiltonian is None:
            self._hamiltonian = self.calculate_hamiltonian()
        return self._hamiltonian

    def flip_energy_change(self, site_i: int) -> float:
        return -2 * self.vertexes[site_i].site_energy()  # flip cause E -> -E

    def flip(self, site_i: int) -> float:
        if self._hamiltonian is None:
            _ = self.hamiltonian
        self._hamiltonian += self.flip_energy_change(site_i)
        self.vertexes[site_i].flip()


class IsingModel2D(IsingModel):
    def __init__(self, lattice_scale: int, init_spin: Optional[int] = None) -> None:
        self.lattice_scale = lattice_scale

        if init_spin is None:
            def spin_generator(): return np.random.choice([-1, 1])
        else:
            def spin_generator(): return init_spin

        sites = [
            IsingSite(spin=spin_generator()) for _ in range(self.lattice_scale**2)]
        super().__init__(vertexes=sites)

        for i in range(lattice_scale):
            for j in range(lattice_scale):
                self.add_edge(i, j, i, (j+1)%lattice_scale)
                self.add_edge(i, j, (i+1)%lattice_scale, j)

    def i_to_ij(self, i: int) -> Tuple[int, int]:
        return i//self.lattice_scale, i % self.lattice_scale

    def ij_to_i(self, i: int, j: int) -> int:
        return i*self.lattice_scale + j

    def add_edge(self, headvex_i: int, headvex_j: int,
                 tailvex_i: int, tailvex_j: int) -> None:
        headi = self.ij_to_i(headvex_i, headvex_j)
        taili = self.ij_to_i(tailvex_i, tailvex_j)
        super().add_edge(headi, taili)
