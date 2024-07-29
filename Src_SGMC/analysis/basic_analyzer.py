import numpy as np 
import matplotlib.pyplot as plt
from typing import Tuple
from scipy.interpolate import CubicSpline
import os

kb = 8.617333262e-5
T = 1500.0 

def read_mu_file(file : str) -> Tuple[np.ndarray, np.ndarray, np.ndarray] :
    mu = []
    c = []
    sigma = []
    with open(file,'r') as r : 
        for l in r : 
            l_split = l.split('|')
            mu.append(float(l_split[0]))
            c.append(float(l_split[2]))
            sigma.append(float(l_split[3]))

    return np.array(mu), np.array(c), np.array(sigma)


def generate_color_map(nb_color : int, color_map : str = 'gnuplot') :
    cmap = plt.get_cmap(color_map)
    return [cmap(i) for i in np.linspace(0.0,1.0, num=nb_color)]

def estimate_uncertainty(c : np.ndarray, sigma_c : np.ndarray, tolerance_c : float) -> float : 
    index_c_star = np.argmin(np.abs(c - tolerance_c))
    other_index = index_c_star - 1 if c[index_c_star] - tolerance_c < 0.0 else index_c_star + 1 

    # linear estimation
    c1, c2 = c[index_c_star], c[other_index]
    s1, s2 = sigma_c[index_c_star], sigma_c[other_index]
    return (s2 - s1)*(tolerance_c - c1)/(c2 - c1) + s1

def plot_estimate_chemical_potential(file : str, color, target_concentration : float) -> Tuple[float,float] : 
    mu, c, sigma = read_mu_file(file)
    label = os.path.basename(file).split('.')[0]

    #pretty plot !
    plt.errorbar(mu, c, yerr=sigma, fmt='o', color=color, label=f'{label}')
    plt.fill_between(mu, c - sigma, c + sigma, color=color, alpha=0.2)

    #cubic spline time 
    # sorting
    combined = list(zip(mu, c))
    sorted_combined = sorted(combined, key=lambda x: x[0])
    mu, c = zip(*sorted_combined)

    inter_cs = CubicSpline(mu,c)
    fake_mu_array = np.linspace(np.amin(mu),np.amax(mu), num = 2000)
    plt.plot(fake_mu_array, inter_cs(fake_mu_array), color=color, alpha=0.5)

    uncertainty_c = estimate_uncertainty(np.array(c), sigma, target_concentration)
    return fake_mu_array[np.argmin(np.abs(inter_cs(fake_mu_array)-target_concentration))], uncertainty_c

######################
### INPUTS
######################
directory_data = 'data'
target_concetration = 0.2
color_map = 'viridis'

######################

if not os.path.exists(directory_data) : 
    raise TimeoutError('The data directory does not exist')

list_data_file = [f'{directory_data}/{f}' for f in os.listdir(directory_data)]
color_array = generate_color_map(len(list_data_file),color_map=color_map)
for id_file, file_data in enumerate(list_data_file) : 
    mu_estimation, uncertainty_c = plot_estimate_chemical_potential(file_data, color_array[id_file], target_concetration)
    print(f' {os.path.basename(file_data).split('.')[0]} : mu estimation for <c> = {target_concetration} pm {round(uncertainty_c,3)} is mu = {mu_estimation} eV')
plt.legend()

plt.xlabel(r'Chemical potential $\Delta \mu$ (eV)', size=16)
plt.ylabel(r'Average concentration $\langle c \rangle$', size=16)

xlim = plt.xlim()
plt.hlines(target_concetration, xlim[0], xlim[1], linestyles='dashed',colors='black')

plt.savefig('mu_data.pdf', dpi=300)
plt.show()
