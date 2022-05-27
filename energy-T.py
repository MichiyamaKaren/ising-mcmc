import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from isingmc import get_ising2D_sampler


def ising(temperature: float, lattice_scale: int, nsamples: int):
    sampler = get_ising2D_sampler(beta=1/temperature, lattice_scale=lattice_scale)
    E = sampler.sample(nsamples=nsamples, sample_interval=lattice_scale**2)
    error = sampler.error_binning(E)
    return np.mean(E), error[-1]


Tc = 2 / np.log(np.sqrt(2)+1)
T = np.linspace(0.5, 2, 50) * Tc

energy = []
energy_error = []
for t_i in tqdm(T):
    E_T, sigmaE_T = ising(temperature=t_i, lattice_scale=6, nsamples=int(1e5))
    energy.append(E_T)
    energy_error.append(sigmaE_T)

np.savez('energy-T.npz', energy=energy, error=energy_error)

plt.errorbar(T/Tc, energy, yerr=energy_error*10)
plt.xlabel(r'Temperature $T/T_c$')
plt.ylabel('Energy')
plt.title('Ising model energy (error*10)')
plt.savefig('energy-T.png', bbox_inches='tight')