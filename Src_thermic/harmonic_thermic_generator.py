import numpy as np
from lattice import SolidAse
from numpy.random import normal
import os, shutil

from ase import Atoms, Atom
from typing import List, Dict, Tuple
from milady_writer import write_milady_poscar

from phonons.phonons_diag import HarmonicVibration

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

class DBDictionnaryBuilder : 
    """Build the general dictionnary object to launch Milady calcuation, specific format of the dictionnay is
    detailed in DBManager doc object"""
    def __init__(self) : 
        self.dic : Dict[str,List[Atoms]] = {}

    def _builder(self, list_atoms : List[Atoms], list_sub_class : List[str] ) -> dict : 
        """Build the dictionnary from Atoms or List[Atoms] object"""
        for id_at, atoms in enumerate(list_atoms) :
            self._update(atoms, list_sub_class[id_at])
        
        return self._generate_dictionnary()

    def _update(self, atoms : Atoms, sub_class : str) -> None : 
        """Update entries of dictionnary"""
        if sub_class not in self.dic.keys() : 
            self.dic[sub_class] = [atoms]
        else : 
            self.dic[sub_class].append(atoms)     

    def _generate_dictionnary(self) -> dict :
        """Generate dictionnary object to launch Milady calcuation, specific format of the dictionnay is
        detailed in DBManager doc object"""
        dictionnary = {}
        for sub_class in self.dic.keys() : 
            for id, atoms in enumerate(self.dic[sub_class]) : 
                name_poscar = '{:}_{:}'.format(sub_class,str(1000000+id+1)[1:])
                dictionnary[name_poscar] = {'atoms':atoms,'energy':None,'energy':None,'forces':None,'stress':None}

        return dictionnary

class HarmonicThermicGenerator : 
    """Build fake thermic configuration for a given lattice"""
    def __init__(self, temperature = List[float], configs_per_temp : int = 100) -> None : 
        self.temperature = temperature
        self.configs_per_temp = configs_per_temp
        self.thermic_configs : Dict[str,Dict[str,List[Atoms]]]= {}

        # conversion time
        self.kB = 8.6173303e-5
        self.eV_per_Da_radHz2 = 2.656e-26
        self.eV_per_Da_radTHz2 = 2.656e-6
        
        # iron data object 
        self.Fe_data = Fe_data()
        self.harmonic_vibration = None 

    def init_harmonic_vibration_object(self, atoms : Atoms,
                                path_potential_lammps : str, 
                                kind_potential : str = 'eam/alloys',
                                working_path : str = '/harmonic_vib', 
                                name_lammps_file = 'in.lmp',
                                delta_xi : float = 1e-3,
                                relative_norm : float = 1e-2) -> None : 
        
        self.harmonic_vibration = HarmonicVibration(atoms, 
                                                    path_potential_lammps,
                                                    displacement_amplitude=delta_xi,
                                                    relative_symmetric_norm=relative_norm,
                                                    working_directory=working_path)
        
        species = ''.join(list(set(atoms.symbols)))
        self.harmonic_vibration.lammps_worker.SetInputsScripts(kind_potential=kind_potential,
                                                               potential='pot.fs',
                                                               species=species,
                                                               name_lammps_file=name_lammps_file)

        return 

    def compute_harmonic_spectra(self, name_file : str = 'in.lmp') -> None : 
        self.harmonic_vibration.InitSimulation(name_file=name_file)
        self.harmonic_vibration.VibrationDiagCalculation()

    def generate_thermic_noise(self, temperature : float, atoms : Atoms, spherical : bool = False, scaling_factor : float = 1.0) -> Atoms : 
        if spherical : 
            return self._generate_spherical_thermic_noise(temperature, self.harmonic_vibration.GetEinsteinOmega(), atoms, scaling_factor=scaling_factor) 
        else : 
            eigen_matrix = self.harmonic_vibration.GetUmatrix()
            eigen_values = self.harmonic_vibration.GetOmega2()
            return self._generate_harmonic_thermic_noise(temperature, eigen_matrix, eigen_values, atoms, scaling_factor=scaling_factor)


    def _generate_spherical_thermic_noise(self, temperature : float, omega_einstein : float, atoms : Atoms, scaling_factor : float = 1.0) -> Atoms : 
        """Generate thermic noise on atoms for a given temperature
        Amplitude of displacement are based on equipartion theorem at debye frequency

        Parameters:
        -----------

        temperature : float 
            Tempertaure of the thermic noise

        atoms : Atoms 
            Atoms object to update 

        Returns:
        --------

        Atoms  
            Updated Atoms object with thermic noise

        """
        mean_mass = np.mean(atoms.get_masses())

        #compute the corresponding quadratic displacement based on equipartion 
        sigma_displacement = np.sqrt(scaling_factor*self.kB*temperature/(self.eV_per_Da_radTHz2*mean_mass*omega_einstein**2))
        atoms.positions += normal(0.0, sigma_displacement, size=atoms.positions.shape)
        return atoms 

    def _generate_harmonic_thermic_noise(self, temperature : float, Umatrix : np.ndarray, omega2_array : np.ndarray, atoms : Atoms, scaling_factor : float = 1.0) -> Atoms : 
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

    def _check_interatomic_distances(self, atoms : Atoms, temperature : float) -> None :
        periodic_positions = self.dirty_periodic_system(atoms)
        list_rij = []
        for pos in atoms.positions :
            for periodic_pos in periodic_positions : 
                rij = np.linalg.norm(pos-periodic_pos) 
                if rij > 0.0 : 
                    list_rij.append(rij)
        
        print('Minimum distance in the system is {:2.4f} AA at {:3.1f} K'.format(np.amin(list_rij),temperature))
        
    def dirty_periodic_system(self, atoms : Atoms) -> np.ndarray : 
        #build periodic system...
        list_positions = [pos for pos in atoms.positions]
        cell = atoms.cell[:]

        for pos in atoms.positions : 
            for xi in [-1.0,0.0,1.0] : 
                for yi in [-1.0,0.0,1.0] : 
                    for zi in [-1.0,0.0,1.0] :
                        if xi == 0 and yi == 0 and zi == 0 : 
                            continue 
                        else : 
                            list_positions.append( pos +  np.array([xi,yi,zi])@cell)

        return np.reshape(list_positions, (len(list_positions),3))

    def GenerateThermicStructures(self, structure : str, 
                                  size : List[int], 
                                  lattice_param : float | List[float],
                                  path_pot_lammps : str, 
                                  name_lammps_file : str, 
                                  symbol : str,
                                  kind_potential : str = 'eam/alloys',
                                  working_dir : str = '/harmonic_vib',
                                  delta_xi : float = 1e-3,
                                  relative_norm : float = 1e-2,
                                  spherical_noise : bool = False,
                                  scaling_factor : float = 1.0) -> None : 
        """Generate thermic configurations for a given structure defined by the following parameters
        
        Parameters:
        -----------

        structure : str
            Type of cristalographic structure to generate
            Possible structres are : BCC, FCC, HCP, Diamond, SimpleCubic, A15, C15

        size : List[int]
            Size of the system to generate in term of unit cell per direction

        lattice_param : float | List[float]
            Lattice parameter(s) of the structure 

        symbol : str
            Chemical element of the structure

        adapted_frequency : bool 
            Adaptating debye frequency to the cristallographic structure 
        """
        
        solid_ase = SolidAse(size,symbol,lattice_param)
        atoms_solid = solid_ase.structure(structure)
        print('... You are generating {:s} systems ...'.format(structure))
        print('... Total number of atoms in the solid system is {:3d} ...'.format(len(atoms_solid)))
        
        self.init_harmonic_vibration_object(atoms_solid, 
                                            path_pot_lammps,
                                            kind_potential=kind_potential,
                                            working_path=working_dir,
                                            name_lammps_file=name_lammps_file,
                                            delta_xi=delta_xi,
                                            relative_norm=relative_norm)
        self.compute_harmonic_spectra(name_file=name_lammps_file)
        
        #writing npz file 
        dynamical_matrix_structure = self.harmonic_vibration.dynamical_matrix
        np.savez('{:s}/{:s}_dynamical'.format(os.path.dirname(path_pot_lammps),structure),dynamical_matrix_structure)

        for temp in self.temperature : 
            temp_list_atoms = []
            for _ in range(self.configs_per_temp) : 
                atoms_solid_temp = self.generate_thermic_noise(temp,atoms_solid, 
                                                               spherical=spherical_noise,
                                                               scaling_factor=scaling_factor)
                self._check_interatomic_distances(atoms_solid_temp, temp)
                temp_list_atoms.append(atoms_solid_temp.copy())

            if structure in self.thermic_configs.keys() : 
                if str(temp) in self.thermic_configs[structure].keys() : 
                     self.thermic_configs[structure][str(temp)] += temp_list_atoms
                else : 
                    self.thermic_configs[structure][str(temp)] = temp_list_atoms

            else : 
                self.thermic_configs[structure] = {str(temp): temp_list_atoms}
        print('... Generation is done ...')
        print()    
        return 
    
    def GenerateDBDictionnary(self) -> Tuple[dict] : 
        db_dictionnary = DBDictionnaryBuilder()
        dic_equiv = {}
        for id_struc, struc in enumerate(self.thermic_configs.keys()) : 
            sub_class = '{:s}_000'.format(str(int(100+id_struc))[1:])
            dic_equiv[sub_class] = struc
            list_full_config_structure = sum([self.thermic_configs[struc][temp] for temp in self.thermic_configs[struc].keys()], [])
            list_sub_class = [sub_class for _ in range(len(list_full_config_structure))]
            db_dictionnary._builder(list_full_config_structure, list_sub_class)
        
        return dic_equiv, db_dictionnary._generate_dictionnary()

    def writer(self, db_dic : dict , path_writing : str) -> None : 
        for name_poscar in db_dic.keys() : 
            write_milady_poscar('{:s}/{:s}.POSCAR'.format(path_writing,name_poscar),
                                db_dic[name_poscar]['atoms'],
                                energy=None,
                                forces=None,
                                stress=None)

################################
### INPUTS
################################
configs_per_T = 10
list_temperature = [50.0,300.0,500.0]
writing_path = '{:s}/test_write'.format(os.getcwd())
structure_to_do = ['BCC','FCC','HCP','A15','C15']
size_to_do = [[4,4,4], [3,3,3], [4,4,4], [3,3,3], [2,2,2]]

################################
### LAMMPS INPUTS 
################################
lammps_dict = {'kind_pot':'eam/alloy',
               'pot_lammps':'/home/lapointe/WorkML/GenerateThermalConfig/pot_lammps/AM05.fs'}

if not os.path.exists(writing_path) : 
    os.mkdir(writing_path)
else : 
    shutil.rmtree(writing_path)
    os.mkdir(writing_path)


thermal_obj = HarmonicThermicGenerator(list_temperature, configs_per_T)

for struct, size in zip(structure_to_do, size_to_do) : 
    thermal_obj.GenerateThermicStructures(struct, 
                                          size, 
                                          2.8553, 
                                          lammps_dict['pot_lammps'],
                                          'in.lmp',
                                          'Fe',
                                          kind_potential=lammps_dict['kind_pot'],
                                          working_dir='./harmonic_vib',
                                          delta_xi=1e-3,
                                          relative_norm=1e-2,
                                          spherical_noise=False,
                                          scaling_factor=0.333)
dic_equiv, db_dic = thermal_obj.GenerateDBDictionnary()

print(dic_equiv)
thermal_obj.writer(db_dic, writing_path)