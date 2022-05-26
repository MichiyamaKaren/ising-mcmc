import unittest

import numpy as np
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt

from isingmc import MCMCSampler, IsingMCMCState, IsingModel, IsingModel2D
from isingmc.ising.sampler import FlipSite


class IsingStateTransRecord(IsingMCMCState):
    def __init__(self, model: IsingModel, beta: float) -> None:
        super().__init__(model, beta)
        self.transfer_dE = []
        self.transfer_applied = []

    def make_transfer(self):
        transfer: FlipSite = super().make_transfer()
        dE = self.model.flip_energy_change(transfer.flip_i)
        self.transfer_dE.append(dE)
        self.transfer_applied.append(False)
        return transfer

    def apply_transfer(self, transfer: FlipSite) -> None:
        self.transfer_applied[-1] = True
        return super().apply_transfer(transfer)


class IsingMCMCTest(unittest.TestCase):
    def test_ising_transfer(self):
        beta = 0.3
        model = IsingModel2D(lattice_scale=6)
        state = IsingStateTransRecord(model, beta=beta)
        sampler = MCMCSampler(state)
        sampler.sample(nsamples=int(1e4))

        bin_result = binned_statistic(
            sampler.state.transfer_dE, sampler.state.transfer_applied,
            bins=np.arange(-9, 10, 2))
        bin_centers = (bin_result.bin_edges[1:]+bin_result.bin_edges[:-1])/2
        plt.bar(bin_centers, bin_result.statistic)
        plt.xticks(np.arange(-8, 9, 2))
        plot_dE = np.linspace(0, 8, 1000)
        plt.plot(plot_dE, np.exp(-beta*plot_dE), color='red')
        plt.show()
