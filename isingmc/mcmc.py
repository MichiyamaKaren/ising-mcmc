from abc import ABCMeta, abstractmethod

import numpy as np
from tqdm import trange

from typing import List, Optional


class StateTransfer(metaclass=ABCMeta):
    @abstractmethod
    def accept_prob(self, state: 'MCMCState'):
        pass


class MCMCState(metaclass=ABCMeta):
    @abstractmethod
    def make_transfer(self) -> StateTransfer:
        pass

    @abstractmethod
    def apply_transfer(self, transfer: StateTransfer) -> None:
        pass

    def transfer(self) -> None:
        transfer = self.make_transfer()
        if np.random.random() <= transfer.accept_prob(self):
            self.apply_transfer(transfer)

    @abstractmethod
    def observable(self):
        pass


class MCMCSampler:
    def __init__(self, state: MCMCState) -> None:
        self.state = state

    def sample(self, nsamples: int, sample_interval: int = 1,
               progress_bar_on: bool = False) -> List:
        observable = []

        nsteps_iter = trange if progress_bar_on else range
        for _ in nsteps_iter(nsamples):
            for __ in range(sample_interval):
                self.state.transfer()
            observable.append(self.state.observable())
        return observable

    def error_binning(self, sample: List[float], max_level: Optional[int] = None) -> np.ndarray:
        if max_level is None:
            max_level = len(sample)

        binned_sample = np.array(sample)
        error = []
        l = 0
        while binned_sample.shape[0] >= 2 and l < max_level:
            error.append(np.std(binned_sample))
            if binned_sample.shape[0] % 2:
                odd_samples = binned_sample[:-1:2]
            else:
                odd_samples = binned_sample[::2]
            binned_sample = (odd_samples + binned_sample[1::2]) / 2
            l += 1
        return error
