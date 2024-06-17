import os
import numpy as np
from lattice import SolidAse
from numpy.random import normal

from ase import Atoms 
from typing import Dict, List, TypedDict

from milady_writer import write_milady_poscar


class Dynamical(TypedDict) :
    dynamical_matrix : np.ndarray
    omega2 : np.ndarray
    xi_matrix : np.ndarray

class AtomsAssembly : 
    def __init__(self) : 
        self.assembly : Dict[str,List[Atoms]] = {}
    
    def update_assembly(self, struct : str, atoms : Atoms) -> None : 
        if struct in self.assembly : 
            self.assembly[struct].append(atoms)
        else : 
            self.assembly[struct] = [atoms]

    def extract_number_of_atoms(self, struct : str) -> int : 
        return len(self.assembly[struct][0])
    
    def extract_covariance_matrix_atom(self, struct : str) -> Dict[int,np.ndarray] :
        atoms_number = self.extract_number_of_atoms(struct)
        dic_cov = {id:None for id in range(atoms_number)}
        for config_k in self.assembly[struct] : 
            positions_k = config_k.positions
            for id in range(atoms_number) : 
                if dic_cov[id] is None : 
                    dic_cov[id] = [positions_k[id]]
                else : 
                    dic_cov[id].append(positions_k[id])

        for id in dic_cov.keys() : 
            design_matrix = np.reshape(dic_cov[id], (len(dic_cov[id]),3))
            dic_cov[id] = (design_matrix.T@design_matrix)/(design_matrix.shape[0]-1)

        return dic_cov    
    

class Fe_data :
    def __init__(self) : 
       self.dic_a0 = {'BCC':2.8553,
                  'FCC':3.5225,
                  'HCP':3.5225/np.sqrt(2),
                  'Diamond':2.8553,
                  'SimpleCubic':2.8553,
                  'A15':3.5225,
                  'C15':2.8553} 

    def _iron_based_a0(self, structure : str) -> None :
        try : 
            return self.dic_a0[structure]
        except KeyError : 
            raise NotImplementedError('{:s} structure is not implemented'.format(structure))

    def a0_T_Fe(self, temperature : float, a0 : float = 2.8553) -> float :
        """Estimate the lattice parameter expansion based on iron data

        Parameters:
        -----------

        temperature : float 
            Temperature of the estimation

        a0 : float 
            Lattice parameter at 0K

        Returns:
        --------

        float 
            a0(T) according the experimental thermal expansion

        """
        #interpolation data for lattice parameter...
        array_delta_a0 = np.array([-0.002,-0.00132, -0.00087, 0.0, 0.00089, 0.00204, 0.00371, 0.00439, 0.00611, 0.00698, 0.00795])
        array_temp = np.array([0.0, 97.0, 174.0 ,300.0, 397.0, 521.0, 614.0, 696.0, 821.0, 879.0, 922.0])
        poly_delta_a0 = np.polyfit(array_temp, array_delta_a0, deg=2)
        return a0*(1.0 + np.poly1d(poly_delta_a0)(temperature) - np.poly1d(poly_delta_a0)(0.0))

class ThermicSampling : 
    def __init__(self, dic_size : Dict[str,List[int]], path_npz : str,
                 temperature : float, 
                 scaling_factor=None,
                 nb_sample : int = 1000) -> None : 
        
        self.kB = 8.6173303e-5
        self.eV_per_Da_radHz2 = 2.656e-26
        self.eV_per_Da_radTHz2 = 2.656e-6
        self.Fe_data = Fe_data()

        self.dic_size = dic_size
        self.path_npz = path_npz
        self.temperature = temperature
        self.nb_sample = nb_sample


        if scaling_factor is not None : 
            if len(scaling_factor) != len(self.dic_size) : 
                raise ValueError('Number of scaling factor is not coherent with dictionnary')
            else : 
                self.scaling_factor = scaling_factor
        else : 
            self.scaling_factor = [1.0 for _ in range(len(self.dic_size))]

        print('... Reading dynamical matrices ...')
        self.dict_dynamical_matrix = self.read_npz_file()
        print('... Diagonalise dynamical matrices ...')
        self.diagonalise_dynamical_matrix()

    def read_npz_file(self) -> Dict[str, Dynamical] : 
        dict_dynamical_matrix : Dict[str, Dynamical] = {}
        list_npz_file = [file for file in self.path_npz if 'npz' in file]
        for npz in list_npz_file : 
            dict_dynamical_matrix[os.path.basename(npz).split('_')[0]]['dynamical_matrix'] = np.load(npz)

    def diagonalise_dynamical_matrix(self) : 
        for struct in self.dict_dynamical_matrix.keys() : 
            eigen_values, eigen_vectors = np.linalg.eigh(self.dict_dynamical_matrix[struct], UPLO='L')
            self.dict_dynamical_matrix[struct]['omega2'] = eigen_values*1.0e4
            self.dict_dynamical_matrix[struct]['xi_matrix'] = eigen_vectors

    def generate_harmonic_thermic_noise(self, temperature : float, Umatrix : np.ndarray, omega2_array : np.ndarray, atoms : Atoms, scaling_factor : float = 1.0) -> Atoms : 
        """Generate thermic noise on atoms for a given temperature
        Amplitude of displacement are based on equipartion theorem at debye frequency

        Parameters:
        -----------

        temperature : float 
            Tempertaure of the thermic noise

        Umatrix : np.ndarray 
            Eigenvectors matrix 

        omega_array : np.ndarray 
            eigen pulsations of the system

        atoms : Atoms 
            Atoms object to update 

        Returns:
        --------

        Atoms  
            Updated Atoms object with thermic noise

        """
        mean_mass = np.mean(atoms.get_masses())

        #compute the corresponding quadratic displacement based on equipartion in eigenvector basis 
        list_eigen_sigma_displacements = [ np.sqrt(scaling_factor*self.kB*temperature/(self.eV_per_Da_radTHz2*mean_mass*omega2)) for omega2 in omega2_array ]        
        noise_displacement_vector = np.array([ normal(0.0, eigen_sigma_displacements) for eigen_sigma_displacements in list_eigen_sigma_displacements ]) 

        #Temperature renormalisation...
        T_estimated = np.sum(self.eV_per_Da_radTHz2*mean_mass*omega2_array*np.power(noise_displacement_vector, 2))/(self.kB*len(omega2_array))
        print(' Estimated temperature for the configuration is {:3.2f} K'.format(T_estimated))

        #basis change to go back in cartesian !
        cartesian_displacement = Umatrix@noise_displacement_vector
        atoms.positions += cartesian_displacement.reshape( (len(atoms),3) )
        return atoms    
    
    def build_covariance_estimator(self, symbol : str = 'Fe') :
        for struct in self.dict_dynamical_matrix.keys() :
            solid_ase = SolidAse(self.dic_size[struct],symbol,self.Fe_data.dic_a0[struct])
            atoms_solid = solid_ase.structure(struct)
            atoms_assembly = AtomsAssembly()
            for _ in range(self.nb_sample) : 
                atoms_k = self.generate_harmonic_thermic_noise(self.temperature, 
                                                     self.dict_dynamical_matrix[struct]['xi_matrix'],
                                                     self.dict_dynamical_matrix[struct]['omega2'],
                                                     atoms_solid, 
                                                     self.scaling_factor[struct])
                atoms_assembly.update_assembly(struct,atoms_k.copy())

            atoms_assembly.extract_covariance_matrix_atom(struct)
            write_milady_poscar('{:s}.POSCAR'.format(struct),
                                atoms_solid)