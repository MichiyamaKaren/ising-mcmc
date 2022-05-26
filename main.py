# %%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from isingmc import get_ising2D_sampler

# %%
def ising_energy(temperature: float, lattice_scale: int, nsamples: int = 1000):
    sampler = get_ising2D_sampler(beta=1/temperature, lattice_scale=lattice_scale)
    E = sampler.sample(nsamples=nsamples, sample_interval=lattice_scale**2)
    return np.mean(E)


Tc = 2 / np.log(np.sqrt(2)+1)
T = np.linspace(0.5, 2, 50)*Tc
energy = np.array([ising_energy(
    temperature=t_i, lattice_scale=6) for t_i in tqdm(T)])

plt.plot(T/Tc, energy)
plt.xlabel(r'Temperature $T/T_c$')
plt.ylabel('Energy')
plt.savefig('energy.png', bbox_inches='tight')
# %%
