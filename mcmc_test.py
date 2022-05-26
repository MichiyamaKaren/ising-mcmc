import unittest

import numpy as np
import matplotlib.pyplot as plt

from isingmc import MCMCSampler, MCMCState, StateTransfer


class ExpTransfer(StateTransfer):
    def __init__(self, x) -> None:
        self.x = x

    def _pi(self, x) -> float:
        return 0 if x < 0 else np.exp(-x)

    def accept_prob(self, state: 'ExpMCMCState'):
        return self._pi(self.x)/self._pi(state.x)


class ExpMCMCState(MCMCState):
    def __init__(self, x) -> None:
        self.x = x

    def make_transfer(self) -> ExpTransfer:
        new_x = np.random.normal(self.x, 1)
        return ExpTransfer(new_x)

    def apply_transfer(self, transfer: ExpTransfer) -> None:
        self.x = transfer.x

    def observable(self):
        return self.x


class MCMCTest(unittest.TestCase):
    def test_exp_sampling(self):
        state = ExpMCMCState(3)
        sampler = MCMCSampler(state)
        sample_x = sampler.sample(nsamples=int(1e3))

        plt.hist(sample_x, density=True)
        plot_x = np.linspace(0, np.max(sample_x), 1000)
        plt.plot(plot_x, np.exp(-plot_x))
        plt.show()
