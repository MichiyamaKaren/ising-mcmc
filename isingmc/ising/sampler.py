import numpy as np

from .model import IsingModel, IsingModel2D
from ..mcmc import MCMCSampler, MCMCState, StateTransfer

from typing import Optional


class FlipSite(StateTransfer):
    def __init__(self, flip_i: int) -> None:
        self.flip_i = flip_i

    def accept_prob(self, state: 'IsingMCMCState'):
        energy_change = state.model.flip_energy_change(self.flip_i)
        # not using min(1, exp): save a little time of calculating exp
        if energy_change <= 0:
            return 1
        else:
            return np.exp(-state.beta*energy_change)


class IsingMCMCState(MCMCState):
    def __init__(self, model: IsingModel, beta: float) -> None:
        self.model = model
        self.beta = beta

    def make_transfer(self) -> StateTransfer:
        flip_i = np.random.randint(len(self.model.vertexes))
        return FlipSite(flip_i)

    def apply_transfer(self, transfer: FlipSite) -> None:
        return self.model.flip(transfer.flip_i)

    def observable(self) -> float:
        return self.model.hamiltonian


def get_ising2D_sampler(beta: float, lattice_scale: int, init_spin: Optional[int] = None) -> MCMCSampler:
    model = IsingModel2D(lattice_scale, init_spin)
    state = IsingMCMCState(model, beta)
    sampler = MCMCSampler(state)
    return sampler
