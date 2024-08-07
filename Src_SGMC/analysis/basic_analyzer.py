import numpy as np 
import matplotlib.pyplot as plt

from matplotlib import ticker
from typing import Tuple, List
from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator
import os

kb = 8.617333262e-5
T = 1500.0 

def read_mu_file(file : os.PathLike[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray] :
    """Read the data file containing data about mu and avarage concentration
    
    Parameter
    ---------

    file : os.PathLike[str]
        File to read 
    
        
    Returns
    -------

    np.ndarray : 
        Chemical potential array extracted from file 

    np.ndarray : 
        Average concentration array extracted from file 

    np.ndarray : 
        Error estimation array on concentration from file
    """
    mu = []
    c = []
    sigma = []
    with open(file,'r') as r : 
        for l in r :
            try : 
                l_split = l.split('|')
                mu.append(float(l_split[0]))
                c.append(float(l_split[2]))
                sigma.append(float(l_split[3]))
            except : 
                continue

    return np.array(mu), np.array(c), np.array(sigma)


def generate_color_map(nb_color : int, color_map : str = 'gnuplot') -> List[tuple]:
    """Generate the adquate color map for pretty plots :) 
    
    Parameters
    ----------

    nb_color : int 
        Number of color to generate 

    color_map : str 
        Name of the colormap to used 

    Returns
    -------

    List[tuple]
        List of colors associated to the colormap
    """
    cmap = plt.get_cmap(color_map)
    return [cmap(i) for i in np.linspace(0.0,1.0, num=nb_color)]

def estimate_uncertainty(c : np.ndarray, sigma_c : np.ndarray, target_c : float, nb_sigma : float = 1.0) -> float : 
    """Estimation the uncertainty on concencration for a given centered concentration, 
    error estimation on linear interpolation from sigma_c error array
    
    Parameters
    ----------

    c : np.ndarray 
        Concentraction array

    sigma_c : np.ndarray
        Error array on concentration
        
    target_c : float
        Targeted concentration 

    Returns
    -------
    
    float 
        Uncertainty estimation for target concentration

    """
    index_c_star = np.argmin(np.abs(c - target_c))
    other_index = index_c_star - 1 if c[index_c_star] - target_c < 0.0 else index_c_star + 1 

    sigma_c *= nb_sigma
    # linear estimation
    c1, c2 = c[index_c_star], c[other_index]
    s1, s2 = sigma_c[index_c_star], sigma_c[other_index]
    return (s2 - s1)*(target_c - c1)/(c2 - c1) + s1

def plot_estimate_chemical_potential(file : os.PathLike[str], color : tuple, target_concentration : float, nb_sigma : float = 1.0, interpolation : str = 'cubic-spline') -> Tuple[float,float] : 
    """Plot <c> = f(mu) and estimate the chemical potential associated to the target_concetration
    
    Parameters
    ----------

    file : os.PathLike[str]
        Path to the file to analyze

    color : tuple 
        Color associated to the plot for this file 

    target_concentration : float 
        Target concentration associated to chemical potential to estimate


    Return
    ------

    float  
        Chemical potential associated to target concentration

    float 
        Uncertainty on target concentration
    """
    mu, c, sigma = read_mu_file(file)
    label = os.path.basename(file).split('.')[0]

    #pretty plot !
    plt.errorbar(mu, c, yerr=nb_sigma*sigma, fmt='o', color=color, label=f'{label}')
    plt.fill_between(mu, c - nb_sigma*sigma, c + nb_sigma*sigma, color=color, alpha=0.2)

    #cubic spline time 
    # sorting
    combined = list(zip(mu, c))
    sorted_combined = sorted(combined, key=lambda x: x[0])
    mu, c = zip(*sorted_combined)

    dic_inter = {'cubic-spline': lambda mu,c : CubicSpline(mu,c),
                 'pchip': lambda mu,c : PchipInterpolator(mu,c),
                 'akima' : lambda mu,c : Akima1DInterpolator(mu,c)}
    if interpolation not in dic_inter.keys() :
        raise NotImplementedError('This interpolation method is not implemented !')

    inter_cs =  dic_inter[interpolation](mu,c)
    fake_mu_array = np.linspace(np.amin(mu),np.amax(mu), num = 2000)
    plt.plot(fake_mu_array, inter_cs(fake_mu_array), color=color, alpha=0.5)

    uncertainty_c = estimate_uncertainty(np.array(c), sigma, target_concentration, nb_sigma=nb_sigma)
    return fake_mu_array[np.argmin(np.abs(inter_cs(fake_mu_array)-target_concentration))], uncertainty_c

def plot_estimate_chemical_potential_all(file : List[os.PathLike[str]], target_concentration : float, nb_sigma : float = 1.0) -> Tuple[float,float] : 
    """Plot <c> = f(mu) and estimate the chemical potential associated to the target_concetration
    
    Parameters
    ----------

    file : os.PathLike[str]
        Path to the file to analyze

    color : tuple 
        Color associated to the plot for this file 

    target_concentration : float 
        Target concentration associated to chemical potential to estimate


    Return
    ------

    float  
        Chemical potential associated to target concentration

    float 
        Uncertainty on target concentration
    """
    mu, c, var, compt = None, None, None, 0.0
    for f in file :
        compt += 1.0
        if mu is None : 
            mu, c, sigma = read_mu_file(f)
            var = sigma**2
        else : 
            mu_f, c_f, sigma_f = read_mu_file(f)
            mu += mu_f
            c += c_f
            var += sigma_f**2
            
    
    c *= 1.0/compt 
    mu *= 1.0/compt 
    var *= 1.0/(compt**2)
    
    sigma = np.sqrt(var)
    label = "Full"

    #pretty plot !
    plt.errorbar(mu, c, yerr=sigma, fmt='o', color="red", label=f'{label}')
    plt.fill_between(mu, c - sigma, c + sigma, color="red", alpha=0.2)

    #cubic spline time 
    # sorting
    combined = list(zip(mu, c))
    sorted_combined = sorted(combined, key=lambda x: x[0])
    mu, c = zip(*sorted_combined)

    inter_cs = CubicSpline(mu,c)
    fake_mu_array = np.linspace(np.amin(mu),np.amax(mu), num = 2000)
    plt.plot(fake_mu_array, inter_cs(fake_mu_array), color="red", alpha=0.5)

    uncertainty_c = estimate_uncertainty(np.array(c), sigma, target_concentration)
    return fake_mu_array[np.argmin(np.abs(inter_cs(fake_mu_array)-target_concentration))], uncertainty_c

######################
### INPUTS
######################
directory_data = 'data_ML'
target_concetration = 0.2
color_map = 'viridis'
nb_sigma = 2.0
interpolation_type = 'akima'
######################

if not os.path.exists(directory_data) : 
    raise TimeoutError('The data directory does not exist')

fig, ax = plt.subplots()



list_data_file = [f'{directory_data}/{f}' for f in os.listdir(directory_data)]
color_array = generate_color_map(len(list_data_file),color_map=color_map)
for id_file, file_data in enumerate(list_data_file) : 
    mu_estimation, uncertainty_c = plot_estimate_chemical_potential(file_data, color_array[id_file], target_concetration, nb_sigma=nb_sigma, interpolation=interpolation_type)
    print(f' {os.path.basename(file_data).split('.')[0]} : mu estimation for <c> = {target_concetration} pm {round(uncertainty_c,3)} is mu = {mu_estimation} eV')

#mu_estimation, uncertainty_c  = plot_estimate_chemical_potential_all(list_data_file, target_concetration)
#print(f' full : mu estimation for <c> = {target_concetration} pm {round(uncertainty_c,3)} is mu = {mu_estimation} eV')
plt.legend()

plt.xlabel(r'Chemical potential $\Delta \mu$ (eV)', size=16)
plt.ylabel(r'Average concentration $\langle c \rangle$', size=16)

xlim = plt.xlim()
plt.hlines(target_concetration, xlim[0], xlim[1], linestyles='dashed',colors='black')

#set pretty axes
ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:1.2f}"))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))

ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:1.2f}"))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

plt.tick_params(which='both', width=1)
plt.tick_params(which='major', length=7)
plt.tick_params(which='minor', length=4, color='black')

plt.savefig('mu_data.png', dpi=300)
plt.show()
