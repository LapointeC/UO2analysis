import os,shutil
import numpy as np
from lattice import SolidAse
from numpy.random import normal

import pickle
import warnings
from ase import Atoms 
from typing import Dict, List, TypedDict, Tuple

from milady import DBManager, Optimiser, Regressor, Descriptor, Milady
from milady_writer import write_milady_poscar
from tools import nearest_mode
from sklearn.linear_model import BayesianRidge

class Dynamical(TypedDict) :
    dynamical_matrix : np.ndarray
    omega2 : np.ndarray
    xi_matrix : np.ndarray

class MLdata(TypedDict) : 
    covariance : np.ndarray
    atoms : Atoms
    name_poscar : str

class FitData(TypedDict) : 
    array_desc : np.ndarray 
    array_flat_cov : np.ndarray

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

class AtomsAssembly : 
    """Class needed to build local displacement covariance matrix for each atom of a given system"""
    def __init__(self) -> None : 
        self.assembly : Dict[str,List[Atoms]] = {}
        self.ml_data : Dict[str,MLdata] = {}
    
    def update_assembly(self, struct : str, atoms : Atoms) -> None : 
        """Update assembly object with structures
        
        Parameters:
        -----------

        struct : str 
            name of the structure to fill 

        atoms : Atoms 
            Atoms object associated to the structure
        """

        if struct in self.assembly : 
            self.assembly[struct].append(atoms)
        else : 
            self.assembly[struct] = [atoms]

    def extract_number_of_atoms(self, struct : str) -> int : 
        """Give the total number of atom for a given structure
        
        Parameters:
        -----------

        struct : str 
            structure to do 

        Return:
        -------

        int 
            Number of atoms in the structure
        """
        return len(self.assembly[struct][0])
    
    def extract_covariance_matrix_atom(self, struct : str) -> Dict[int,np.ndarray] :
        """Build the local thermic covariance matrix for each atom of a structure
        
        Parameters:
        -----------
        struct : str
            Structure to do 

        Return:
        -------

        Dict[int,np.ndarray]
            Dictionnary where key are the index of atoms in the structure and value associated to the key is the thermic covariance matrix
        """
        
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
    
    def extract_covariance_matrix_atom_array(self, struct : str) -> np.ndarray :
        """Build the local thermic covariance matrix for each atom of a structure
        
        Parameters:
        -----------
        struct : str
            Structure to do 

        Return:
        -------

        Dict[int,np.ndarray]
            Dictionnary where key are the index of atoms in the structure and value associated to the key is the thermic covariance matrix
        """
        
        atoms_number = self.extract_number_of_atoms(struct)
        dic_cov = {id:None for id in range(atoms_number)}
        for config_k in self.assembly[struct] : 
            positions_k = config_k.positions
            for id in range(atoms_number) : 
                if dic_cov[id] is None : 
                    dic_cov[id] = [positions_k[id]]
                else : 
                    dic_cov[id].append(positions_k[id])

        array_cov = np.zeros((atoms_number,6))
        for id in dic_cov.keys() : 
            design_matrix = np.reshape(dic_cov[id], (len(dic_cov[id]),3))
            tmp_cov = (design_matrix.T@design_matrix)/(design_matrix.shape[0]-1)
            
            compt = 0
            for i in range(tmp_cov.shape[0]) : 
                for j in range(tmp_cov.shape[1]) :
                    if i >= j :
                        array_cov[id,compt] = tmp_cov[i,j]
                        compt += 1
                        
        return array_cov    

    def fill_MLdata(self, struct : str, atoms : Atoms, covariance : Dict[int,np.ndarray], name_poscar : str = None) -> None :
        """Fill the ml_data dictionnary with atomic configurations, covariance matrix
        
        Parameters:
        -----------

        struct : str 
            Structure to fill 

        atoms : Atoms 
            Associated Atoms object 

        covariance : Dict[int,np.ndarray] 
            Associated thermic covariance dictionnary (see extract_covariance_matrix_atom method)

        name_poscar : str 
            name of the associated milady poscar
        """
        self.ml_data[struct] = {'atoms':atoms, 
                                'covariance':covariance,
                                'name_poscar':name_poscar} 

class Fe_data :
    """Litte class to estimate lattice data for iron structures"""
    def __init__(self) : 
       self.dic_a0 = {'BCC':2.8553,
                  'FCC':3.5225,
                  'HCP':3.5225/np.sqrt(2),
                  'Diamond':2.8553,
                  'SimpleCubic':2.8553,
                  'A15':3.5225,
                  'C15':2.8553} 

    def _iron_based_a0(self, structure : str) -> None :
        """Give the estimation of a0 at 0 K
        
        Parameters:
        -----------

        structure : str 
            Cristallographic structure to estimate

        """
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
    """"Thermic sampler object based on previous dynamical matrices calculations"""
    def __init__(self, dic_size : Dict[str,List[int]], path_data : str,
                 temperature : float, 
                 scaling_factor : Dict[str, float] = None,
                 nb_sample : int = 1000,
                 type_data : str = 'npz',
                 save_diag : bool = False) -> None : 
        
        self.kB = 8.6173303e-5
        self.eV_per_Da_radHz2 = 2.656e-26
        self.eV_per_Da_radTHz2 = 2.656e-6
        self.Fe_data = Fe_data()

        self.dic_size = dic_size
        self.path_data = path_data
        self.temperature = temperature
        self.nb_sample = nb_sample

        self.type_data = type_data
        self.atoms_assembly : AtomsAssembly = None
        self.ml_dic : DBDictionnaryBuilder = None

        if scaling_factor is not None : 
            if len(scaling_factor) != len(self.dic_size) : 
                raise ValueError('Number of scaling factor is not coherent with dictionnary')
            else : 
                self.scaling_factor = scaling_factor
        else : 
            self.scaling_factor = None
            #self.scaling_factor = {key : 0.333 for key in self.dic_size.keys()}

        if self.type_data == 'npz' :
            self.dict_dynamical_matrix = self.read_npz_file()
        elif self.type_data == 'hdf5' :
            self.dict_dynamical_matrix = self.read_hdf5_file()
        else :
            raise NotImplementedError('This type of data is not implemented')

        print('... Diagonalise dynamical matrices ...')
        self.diagonalise_dynamical_matrix(save=save_diag)

    def read_npz_file(self) -> Dict[str, Dynamical] :
        dict_dynamical_matrix : Dict[str, Dynamical] = {}
        all_file = ['{:}/{:}'.format(self.path_data,f) for f in os.listdir(self.path_data)]
        list_npz_file = [file for file in all_file if 'npz' in file]
        for npz in list_npz_file :
            dict_dynamical_matrix[os.path.basename(npz).split('_')[0]] = {'dynamical_matrix':np.load(npz)['arr_0'],
                                                                          'omega2':None,
                                                                          'xi_matrix':None,
                                                                          'atoms':None}
        return dict_dynamical_matrix

    def read_hdf5_file(self, sym : str = 'Fe') -> Dict[str, Dynamical] :
        dict_dynamical_matrix : Dict[str, Dynamical] = {}
        with h5py.File(self.path_data,'r') as r :
            dynamical_group = r['dynamical']
            for key, val in dynamical_group.items() :
                cell = val['cell'][:,:]
                positions = val['positions'][:,:]
                symbol = [sym for _ in range(positions.shape[0] )]
                dict_dynamical_matrix[key] = {'dynamical_matrix':val['dynamical_matrix'][:,:],
                                              'omega2':None,
                                              'xi_matrix':None,
                                              'atoms':Atoms(symbols=symbol,positions=positions,cell=cell)}

                break

        return dict_dynamical_matrix

    def CheckFrequencies(self, eigenvalues : np.ndarray, eigenvector : np.ndarray, struc : str) -> Tuple[np.ndarray,bool] : 
        imaginary_modes = [el for el in eigenvalues if el < 1e-2]
        bool_imaginary = False
        if len(imaginary_modes) > 3 : 
            warnings.warn('Instabilities in dynamical matrix for {:} : {:2d} modes are imaginary'.format(struc,len(imaginary_modes)))
            bool_imaginary = True        
        idx_to_keep = [id for id in range(len(eigenvalues)) if eigenvalues[id] > 1e-2]
        eigenvector = eigenvector[:,idx_to_keep]
        eigenvalues = eigenvalues[idx_to_keep]
        return eigenvalues, eigenvector, bool_imaginary

    def diagonalise_dynamical_matrix(self) :
        key2del = [] 
        for struct in self.dict_dynamical_matrix.keys() : 
            eigen_values, eigen_vectors = np.linalg.eigh(self.dict_dynamical_matrix[struct]['dynamical_matrix'], UPLO='L')
            stable_eigen_values, stable_eigen_vectors, bool_im = self.CheckFrequencies(eigen_values*1.0e4, eigen_vectors, struct)
            if bool_im : 
                warnings.warn('Dynamical matrix for {:} presents a problem and will be skiped ...')
                key2del.append(struct)
                #self.dict_dynamical_matrix[struct]['omega2'] = stable_eigen_values
                #self.dict_dynamical_matrix[struct]['xi_matrix'] = stable_eigen_vectors
            else :
                self.dict_dynamical_matrix[struct]['omega2'] = stable_eigen_values
                self.dict_dynamical_matrix[struct]['xi_matrix'] = stable_eigen_vectors
            
        [self.dict_dynamical_matrix.pop(key) for key in key2del] 

    def generate_harmonic_thermic_noise(self, temperature : float, Umatrix : np.ndarray, omega2_array : np.ndarray, atoms : Atoms, scaling_factor : float = 1.0, sigma_number : float = 2.0) -> Atoms :
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
        mass_array = np.zeros(3*len(atoms))
        for id_mass, mass in enumerate(atoms.get_masses()) :
            mass_array[3*id_mass] = mass
            mass_array[3*id_mass+1] = mass
            mass_array[3*id_mass+2] = mass

        mass_vector = np.zeros(len(omega2_array))
        #mass_nu = sum_ia |xi_ia(v)|^2 m_ia
        for id in range(len(omega2_array)) :
            mass_vector[id] = np.sum([Umatrix.T[id,k]**2*mass_array[k] for k in range(len(mass_array))])

        #compute the corresponding quadratic displacement based on equipartion in eigenvector basis 
        omega2_array = np.abs(omega2_array)
        list_eigen_sigma_displacements = [ np.sqrt(scaling_factor*self.kB*temperature/(self.eV_per_Da_radTHz2*mass_eff*omega2)) for mass_eff,omega2 in zip(mass_vector,omega2_array) ]
        noise_displacement_vector = np.array([ normal(0.0, eigen_sigma_displacements) for eigen_sigma_displacements in list_eigen_sigma_displacements ])

        # condition on value of noise...
        condition_noise = np.array([ abs(cart_disp) < sigma_number*eig_disp for cart_disp, eig_disp in zip(noise_displacement_vector,list_eigen_sigma_displacements) ])
        noise_displacement_vector = np.where( condition_noise, noise_displacement_vector, np.sign(noise_displacement_vector)*sigma_number*list_eigen_sigma_displacements )


        #Temperature renormalisation...
        T_estimated = np.sum(self.eV_per_Da_radTHz2*mass_vector*omega2_array*np.power(noise_displacement_vector, 2))/(self.kB*len(omega2_array))
        #print(' Estimated temperature for the configuration is {:3.2f} K'.format(T_estimated))

        #basis change to go back in cartesian !
        cartesian_displacement = Umatrix@noise_displacement_vector
        
        # here we only keep the displacements ...
        atoms.positions = cartesian_displacement.reshape( (len(atoms),3) )
        return atoms    

    def GenerateDBDictionnary(self, atoms_assembly : AtomsAssembly) -> Tuple[dict] : 
        db_dictionnary = DBDictionnaryBuilder()
        dic_equiv = {}
        for id_struc, struc in enumerate(self.dict_dynamical_matrix.keys()) : 
            sub_class = '{:s}_000'.format(str(int(100+id_struc))[1:])
            dic_equiv[sub_class] = struc
            config_struct = atoms_assembly.ml_data[struc]['atoms']
            atoms_assembly.ml_data[struc]['name_poscar'] = '{:}_{:}'.format(sub_class,str(1000000+1)[1:])
            db_dictionnary._builder([config_struct], [sub_class])
        
        return dic_equiv, db_dictionnary._generate_dictionnary()

    def writer(self, db_dic : dict , path_writing : str) -> None : 
        for name_poscar in db_dic.keys() : 
            write_milady_poscar('{:s}/{:s}.POSCAR'.format(path_writing,name_poscar),
                                db_dic[name_poscar]['atoms'],
                                energy=None,
                                forces=None,
                                stress=None)

    def fill_dictionnaries(self, atoms_assembly : AtomsAssembly, ml_dic : dict) -> None :
        self.atoms_assembly = atoms_assembly
        self.ml_dic = ml_dic
        return

    def build_covariance_estimator_basic(self, path_writing : str = './ml_poscar', symbol : str = 'Fe') :
        atoms_assembly = AtomsAssembly()
        for struct in self.dict_dynamical_matrix.keys() :
            print('... Starting convariance estimation for {:}'.format(struct))
            solid_ase = SolidAse(self.dic_size[struct],symbol,self.Fe_data.dic_a0[struct])
            atoms_solid = solid_ase.structure(struct)
            atoms_solid2keep = atoms_solid.copy()
            for _ in range(self.nb_sample) : 
                atoms_k = self.generate_harmonic_thermic_noise(self.temperature, 
                                                     self.dict_dynamical_matrix[struct]['xi_matrix'],
                                                     self.dict_dynamical_matrix[struct]['omega2'],
                                                     atoms_solid, 
                                                     self.scaling_factor[struct])
                atoms_assembly.update_assembly(struct,atoms_k.copy())
            displacement_covariance = atoms_assembly.extract_covariance_matrix_atom(struct)
            atoms_assembly.fill_MLdata(struct,atoms_solid2keep,displacement_covariance)

        print('... Dictionnary object is generated ...')
        dic_equiv, ml_dic = self.GenerateDBDictionnary(atoms_assembly)
        self.fill_dictionnaries(atoms_assembly, ml_dic)
        print(dic_equiv)
        print('... Writing POSCAR files for Milady ...')
        if not os.path.exists(path_writing) : 
            os.mkdir(path_writing)
        else : 
            shutil.rmtree(path_writing)
            os.mkdir(path_writing)
        self.writer(ml_dic, path_writing)
        return 

    def build_covariance_estimator(self, path_writing : str = './ml_poscar', nb_sigma : float = 1.5) :
        atoms_assembly = AtomsAssembly()
        for struct, obj in self.dict_dynamical_matrix.items() :
            print('... Starting convariance estimation for {:}'.format(struct))
            atoms_solid = obj['atoms']
            atoms_solid2keep = atoms_solid.copy()
            for id_sample in range(self.nb_sample) :
                
                if id_sample%(int(0.1*self.nb_sample)) == 0 : 
                    print(f'    {id_sample} / {self.nb_sample} iterations ...')
                atoms_k = self.generate_harmonic_thermic_noise(self.temperature,
                                                     obj['xi_matrix'],
                                                     obj['omega2'],
                                                     atoms_solid,
                                                     scaling_factor= 0.333 if self.scaling_factor is None else self.scaling_factor[struct],
                                                     sigma_number=nb_sigma)
                atoms_assembly.update_assembly(struct,atoms_k.copy())
            displacement_covariance = atoms_assembly.extract_covariance_matrix_atom(struct)
            atoms_assembly.fill_MLdata(struct,atoms_solid2keep,displacement_covariance)

        print('... Dictionnary object is generated ...')
        dic_equiv, ml_dic = self.GenerateDBDictionnary(atoms_assembly)
        self.fill_dictionnaries(atoms_assembly, ml_dic)
        print(dic_equiv)
        print('... Writing POSCAR files for Milady ...')
        if not os.path.exists(path_writing) :
            os.mkdir(path_writing)
        else :
            shutil.rmtree(path_writing)
            os.mkdir(path_writing)
        self.writer(ml_dic, path_writing)
        return

    def build_pickle(self, path_pickles : str = './thermic_sampling.pickle') -> None : 
        pickle.dump(self, open(path_pickles,'wb'))
        return

class ThermicFiting : 
    def __init__(self) -> None :
        self.covariance_models : Dict[str, Dict[int, np.ndarray]] = {}
        self.fit_data : Dict[str,FitData] = {}
        self.dic_equiv_index = {i+j:[i,j] for i in range(3) for j in range(3)}
    
    def flatten_dic_covariance(self, dic_cov : Dict[int,np.ndarray]) -> np.ndarray : 
        array_cov = np.zeros((len(dic_cov),6))
        for id in dic_cov.keys() : 
            tmp_cov = dic_cov[id]
            compt = 0 
            for i in range(tmp_cov.shape[0]) : 
                for j in range(tmp_cov.shape[1]) :
                    if i >= j :
                        array_cov[id,compt] = tmp_cov[i,j]
                        compt += 1

        return array_cov

    def update_fit_data(self, key : str, array_desc : np.ndarray, dic_cov : Dict[int, np.ndarray]) -> None : 
        if key in self.fit_data.keys() : 
            self.fit_data[key]['array_desc'] = np.concatenate((self.fit_data[key]['array_desc'], array_desc), axis=0)
            self.fit_data[key]['array_flat_cov'] = np.concatenate((self.fit_data[key]['array_flat_cov'], self.flatten_dic_covariance(dic_cov)), axis=0)
        else : 
            self.fit_data[key] = {'array_desc':array_desc, 'array_flat_cov':self.flatten_dic_covariance(dic_cov)}
        
        return 

    def pseudo_inverse_regression(self, y : np.ndarray, X : np.ndarray, lamb : float = 1e-4) -> np.ndarray : 
        pseudo_inv = np.linalg.pinv( X.T@X + lamb*np.eye(X.shape[1]) )
        return y.T@X@pseudo_inv

    def build_regression_models(self, key : str) -> None : 
        local_models = {i:None for i in range(6)}
        X_key = self.fit_data[key]['array_desc']
        y_key = self.fit_data[key]['array_flat_cov']
        for id in local_models.keys() : 
            local_models[id] = self.pseudo_inverse_regression(y_key[:,id], X_key)
        
        self.covariance_models[key] = local_models
        return 

######################################
## INPUTS
######################################
dict_size = {'BCC':[4,4,4], 
             'FCC':[3,3,3], 
             'HCP':[4,4,4], 
             'A15':[3,3,3], 
             'C15':[2,2,2]}
path_npz = '/home/lapointe/WorkML/GenerateThermalConfig/pot_lammps'
scaling_factor = {'BCC':0.333, 
             'FCC':0.333, 
             'HCP':0.333, 
             'A15':0.333, 
             'C15':0.333}
mode = 'thermic_fiting'
#######################################


list_mode = ['precalculation','thermic_fiting']
if mode not in list_mode :
    print('There is something wrong with mode')
    possible_mode = nearest_mode(list_mode,mode,3)
    strg = ''
    for string in possible_mode :
        strg += '%s, '%(string)
    print('Maybe you mean : %s...'%(strg[:-2]))
    exit(0)

if mode == 'precalculation' :
    thermic_sampler = ThermicSampling(dict_size,
                                      path_npz,
                                      300.0,
                                      scaling_factor=scaling_factor,
                                      nb_sample=10000)
    thermic_sampler.build_covariance_estimator(symbol='Fe')
    print('... Pickle file for thermic covariance will be written ...')
    thermic_sampler.build_pickle()
    # milady ! 
    dbmodel = DBManager(model_ini_dict=thermic_sampler.ml_dic)
    print('... All configurations have been embeded in Atoms objects ...')
    optimiser = Optimiser.Milady(fix_no_of_elements=1,
                                 chemical_elements=['Fe'])
    regressor = Regressor.ComputeDescriptors(write_design_matrix=False)
    descriptor = Descriptor.BSO4(r_cut=4.7,j_max=4.0,lbso4_diag=False)
    # command setup for milady
    os.environ['MILADY_COMMAND'] = '/home/lapointe/Git/mld_build_intel/bin/milady_main.exe'
    os.environ['MPI_COMMAND'] = 'mpirun -np'
    # launch milady for descriptor computation
    print('... Starting Milady ...')
    mld_calc = Milady(optimiser,
                          regressor,
                          descriptor,
                          dbmodel=dbmodel,
                          directory='mld',
                          ncpu=2)
    mld_calc.calculate(properties=['milady-descriptors'])
    print('... Writing pickle object from ML ...')
    pickle.dump(mld_calc.dbmodel, open('mld.pickle','wb'))
    print('... Pickle object is written :) ...')

if mode == 'thermic_fiting' : 
    dbmodel : DBManager = pickle.load(open('mld.pickle','rb'))
    thermic_obj : ThermicSampling = pickle.load(open('thermic_sampling.pickle','rb'))

    fit_obj = ThermicFiting()

    print('... Reading structures ...')
    for struc in dbmodel.model_init_dic.keys() : 
        descriptor_struct = dbmodel.model_init_dic[struc]['atoms'].get_array('milady-descriptors')
        print(struc)
        print([thermic_obj.atoms_assembly.ml_data[key]['name_poscar'] for key in thermic_obj.atoms_assembly.ml_data.keys()])
        key_cov = [key for key in thermic_obj.atoms_assembly.ml_data.keys() if struc in thermic_obj.atoms_assembly.ml_data[key]['name_poscar']][0]
        print(' ... {:} structure ...'.format(key_cov))
        covariance_struct = thermic_obj.atoms_assembly.ml_data[key_cov]['covariance']
        print(covariance_struct)
        fit_obj.update_fit_data(key_cov, descriptor_struct, covariance_struct)
        print(' ... {:} structure is filled ...'.format(key_cov))

    fit_obj.build_regression_models('BCC')
    print(fit_obj.covariance_models)
