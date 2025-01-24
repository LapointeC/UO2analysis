import numpy as np
import pickle
import time

from typing import Tuple, Dict, List, TypedDict
from ase import Atoms

from fitting_objects import WeightMapper, RegressorFreeEnergy
from sklearn.covariance import MinCovDet

class Data(TypedDict) : 
    array_temperature : np.ndarray
    array_ref_FE : np.ndarray 
    array_anah_FE : np.ndarray
    array_full_FE : np.ndarray
    array_sigma_FE : np.ndarray
    atoms : Atoms
    volume : float
    stress : np.ndarray
    Ff : np.ndarray

class CUR :
    def __init__(self, c_cur : int, eps : float) -> None :
        self.c_cur = c_cur
        self.eps = eps
        self.c = c_cur * np.log(c_cur) / eps**2 #expectation number of sampled columns
        self.C, self.U, self.R = None, None, None #matrices of decomposition
        self.pi_col, self.pi_row = None, None #leverage scores of corresponding columns/rows
        self.col_indices = None
        self.row_indices = None

    def column_select(self, A : np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = A.shape[1]
        A = np.array(A.copy())
        _, _, vh = np.linalg.svd(A, full_matrices=False)
        v_k = vh[0:self.c_cur, :]

        pi = 1 / self.c_cur * np.sum(v_k**2, axis=0)
        c_index = [np.random.choice(2,
                        p=[1 - min(1, self.c * pi[i]), min(1, self.c * pi[i])]) for i in range(n)
                  ]
        c_index = np.nonzero(c_index)[0]

        C = A[:, c_index]
        return C, c_index, pi

    def run_CUR(self, A : np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        A = np.array(A.copy())
        self.C, self.col_indices, self.pi_col = self.column_select(A)
        self.R, self.row_indices, self.pi_row = self.column_select(A.T)
        self.U = np.linalg.pinv(self.C) @ A @ np.linalg.pinv(self.R.T)
        return self.C, self.U, self.R.T

class DescriptorFreeEnergy :
    def __init__(self, nb_column_selected : int,
                 error_cur : float = 1e-3,
                 descriptor_type : str = 'quad',
                 dim_2b : int = None,
                 dim_mb : int = None) -> None : 
    
        self.nb_column_selected = nb_column_selected
        self.descriptor_type = descriptor_type
        print(f'... You are using desc_type : {self.descriptor_type} ...')
        self.cur = CUR(nb_column_selected, error_cur)
        self.selected_c_index = None
        self.dim_2b = dim_2b
        self.dim_mb = dim_mb

    def FindIndexCUR(self, design_matrix : np.ndarray) -> Tuple[np.ndarray, np.ndarray] :
        reduced_matrix_c, index_columns, _ = self.cur.column_select(design_matrix)
        self.selected_c_index = index_columns
        return reduced_matrix_c, index_columns

    def flat_matrix(self, matrix : np.ndarray) -> np.ndarray:
        # Get the indices for the upper triangular part of the matrix
        triu_indices = np.triu_indices(matrix.shape[0])
        # Use these indices to extract the upper triangular elements and flatten them
        flat_matrix = matrix[triu_indices]

        return flat_matrix

    def LinDesc(self, array_desc : np.ndarray) -> np.ndarray : 
        return np.sum(array_desc, axis=0)

    def QuadDesc(self, array_desc : np.ndarray) -> np.ndarray : 
        linear_desc = np.sum(array_desc, axis=0) #np.sum(array_desc[:,self.selected_c_index], axis=0)
        quad_desc = self.flat_matrix(array_desc[:,self.selected_c_index].T @ array_desc[:,self.selected_c_index])
        return np.concatenate((linear_desc, quad_desc))     

    def CubicCUR(self, array_desc : np.ndarray) -> np.ndarray : 
        linear_desc = np.sum(array_desc, axis=0) #np.sum(array_desc[:,self.selected_c_index], axis=0)
        quad_desc = self.flat_matrix(array_desc[:,self.selected_c_index].T @ array_desc[:,self.selected_c_index])
        cubic_desc = np.einsum('mi,mj,mk->')      
        pass

    def QuadDescSemi(self, array_desc : np.ndarray) -> np.ndarray : 
        # linear part
        linear_desc = np.sum(array_desc, axis=0)
        
        # quad part 
        eigen_values_cov, eigvect = np.linalg.eigh(array_desc.T@array_desc, UPLO='L')
        nb_to_remove = int((len(eigen_values_cov) - self.nb_column_selected))
        reduced_eigvalue = eigen_values_cov[nb_to_remove:]
        reduced_eigvect = eigvect[nb_to_remove:,nb_to_remove:]
        quad_desc = self.flat_matrix(reduced_eigvect@np.diag(reduced_eigvalue)@reduced_eigvect.T)
        return np.concatenate( (linear_desc,quad_desc) )

    def CubicDescSemi(self, array_desc : np.ndarray) -> np.ndarray : 
        # linear part
        linear_desc = np.sum(array_desc, axis=0)
        
        # quad part 
        cov = np.cov(array_desc.T)
        #eigen_values_cov, eigvect = np.linalg.eigh(array_desc.T@array_desc, UPLO='L')
        eigen_values_cov, eigvect = np.linalg.eigh(cov, UPLO='L')
        nb_to_remove = int((len(eigen_values_cov) - self.nb_column_selected))
        reduced_eigvalue = eigen_values_cov[nb_to_remove:]
        reduced_eigvect = eigvect[nb_to_remove:,nb_to_remove:]

        reduced_matrix = reduced_eigvect@np.diag(reduced_eigvalue)@reduced_eigvect.T
        quad_desc = self.flat_matrix(reduced_matrix)

        # concatenation
        tmp_desc = np.concatenate( (linear_desc,quad_desc) )
        cubic_desc = np.power(reduced_eigvalue,1.5)
        return np.concatenate( (tmp_desc, cubic_desc))

    def CubicDescSemiG(self, array_desc : np.ndarray) -> np.ndarray : 
        # linear part
        linear_desc = np.sum(array_desc, axis=0)
        
        # quad part 
        cov = np.cov(array_desc.T[:self.dim_mb,:])
        #eigen_values_cov, eigvect = np.linalg.eigh(array_desc.T@array_desc, UPLO='L')
        eigen_values_cov, eigvect = np.linalg.eigh(cov, UPLO='L')
        nb_to_remove = int((len(eigen_values_cov) - self.nb_column_selected))
        reduced_eigvalue = eigen_values_cov[nb_to_remove:]
        reduced_eigvect = eigvect[nb_to_remove:,nb_to_remove:]

        reduced_matrix = reduced_eigvect@np.diag(reduced_eigvalue)@reduced_eigvect.T
        quad_desc = self.flat_matrix(reduced_matrix)

        # concatenation
        tmp_desc = np.concatenate( (linear_desc,quad_desc) )
        cubic_desc = np.power(reduced_eigvalue,1.5)
        return np.concatenate( (tmp_desc, cubic_desc))

    def CubicDescSemiGG(self, array_desc : np.ndarray) -> np.ndarray : 
        # linear part
        linear_desc = np.sum(array_desc, axis=0)
        
        # quad part 
        cov = np.cov(array_desc.T[:self.dim_mb,:])

        #np.fill_diagonal(cov, 0.0)
        #eigen_values_cov, eigvect = np.linalg.eigh(array_desc.T@array_desc, UPLO='L')
        eigen_values_cov, eigvect = np.linalg.eigh(cov, UPLO='L')
        nb_to_remove = int((len(eigen_values_cov) - self.nb_column_selected))
        reduced_eigvalue = np.abs(eigen_values_cov[nb_to_remove:])
        reduced_eigvect = eigvect[nb_to_remove:,nb_to_remove:]

        reduced_matrix = reduced_eigvect@np.diag(reduced_eigvalue)@reduced_eigvect.T
        quad_desc = self.flat_matrix(reduced_matrix)

        # concatenation
        tmp_desc = np.concatenate( (linear_desc,quad_desc) )

        cov_2b = np.cov(array_desc.T[self.dim_mb:,:])
        #np.fill_diagonal(cov_2b, 0.0)

        eigen_values_cov, eigvect = np.linalg.eigh(cov_2b, UPLO='L')
        nb_to_remove = int((len(eigen_values_cov) - self.nb_column_selected))
        reduced_eigvalue = np.abs(eigen_values_cov[nb_to_remove:])
        reduced_eigvect = eigvect[nb_to_remove:,nb_to_remove:]

        reduced_matrix = reduced_eigvect@np.diag(reduced_eigvalue)@reduced_eigvect.T
        quad_desc_2b = self.flat_matrix(reduced_matrix)

        tmp_desc = np.concatenate(( tmp_desc, quad_desc_2b ))

        cubic_desc = np.power(reduced_eigvalue,1.5)
        return np.concatenate( (tmp_desc, cubic_desc))

    def PowerDesc(self, array_desc : np.ndarray, power : int = 2) -> np.ndarray :
        linear_desc = np.sum(array_desc, axis=0)
        desc = linear_desc.copy()
        for k in range(2,power+1) : 
            desc = np.concatenate((desc, np.power(linear_desc,k)/len(linear_desc)**k ))
        return desc

    def ComputeDescriptorFreeEnergy(self, atoms : Atoms) -> np.ndarray : 
        method_desc = {'lin':lambda x : self.LinDesc(x),
                       'quad':lambda x : self.QuadDesc(x),
                       'power':lambda x : self.PowerDesc(x),
                       'semi-quad':lambda x : self.QuadDescSemi(x),
                       'semi-cubic':lambda x : self.CubicDescSemi(x),
                       'semi-cubicG':lambda x : self.CubicDescSemiG(x),
                       'semi-cubicGG':lambda x : self.CubicDescSemiGG(x)}
        desc_mld = atoms.get_array('milady-descriptors')
        return method_desc[self.descriptor_type](desc_mld)

    def FillDescriptorFreeEnergy(self, collection_atoms : Dict[str,Data]) -> Dict[str,Data]: 
        method_desc = {'lin':lambda x : self.LinDesc(x),
                       'quad':lambda x : self.QuadDesc(x),
                       'power':lambda x : self.PowerDesc(x),
                       'semi-quad':lambda x : self.QuadDescSemi(x),
                       'semi-cubic':lambda x : self.CubicDescSemi(x),
                       'semi-cubicG':lambda x : self.CubicDescSemiG(x)}
        
        for _, data in collection_atoms.items() : 
            desc_mld = data['atoms'].get_array('milady-descriptors')
            data['atoms'].set_array('free-energy-desc',
                            method_desc[self.descriptor_type](desc_mld),
                            dtype=float)
            
        return collection_atoms
    
class FreeEnergyManager : 
    def __init__(self, collection_atoms : Dict[str,Data],
                 nb_columns_selected : int,
                 list_temperature : List[float],
                 desc_type : str = 'quad',
                 error_cur : float = 1e-3,
                 mcd_analysis : bool = False,
                 contamination : float = 0.05,
                 size_k2b : int = None,
                 size_bso4 : int = None) -> None :

        self.list_temperature = list_temperature
        self.collection_data = collection_atoms
        self.size_k2b = size_k2b
        self.size_bso4 = size_bso4

        self.descriptor_object = DescriptorFreeEnergy(nb_columns_selected,
                                                 error_cur,
                                                 desc_type,
                                                 dim_2b=self.size_k2b,
                                                 dim_mb=self.size_bso4)
        print('... Preprossing data with CUR ...')
        self.preprocessing_cur()
        
        self.contamination = contamination
        self.mask_mcd = None
        if mcd_analysis : 
            print('... Start MCD analysis ...')
            self.mask_mcd = self.MCDAnalysis()

        self.design_matrix = None
        self.dictionnary_target = { temp:np.zeros(len(collection_atoms)) for temp in self.list_temperature }
        self.dictionnary_errorF = { temp:np.zeros(len(collection_atoms)) for temp in self.list_temperature }
        self.dictionnary_sigmaF = { temp:np.zeros(len(collection_atoms)) for temp in self.list_temperature }

    def MCDAnalysis(self) -> np.ndarray :
        array_design = None
        for idx, name in enumerate(self.collection_data.keys()) : 
            if idx == 0 : 
                desc = self.collection_data[name]['atoms'].get_array('milady-descriptors')
                array_design = np.zeros(( len(self.collection_data), desc.shape[1] ))
                array_design[idx,:] = np.sum(desc, axis=0)

            else : 
                desc = self.collection_data[name]['atoms'].get_array('milady-descriptors')
                array_design[idx,:] = np.sum(desc, axis=0)

        mcd = MinCovDet(support_fraction=1.0-self.contamination)
        mcd.fit(array_design)

        # predicting distances ! 
        mcd_distance = np.sqrt(mcd.mahalanobis(array_design))
        percentile_values = np.percentile(mcd_distance, (1.0-self.contamination)*100 )
        mask_mcd = mcd_distance < percentile_values

        selected = len([el for el in mask_mcd if el])
        print(f'... MCD selection : {selected}/{array_design.shape[0]}...')

        return mask_mcd 

    def preprocessing_cur(self) -> None : 
        array_design = None
        for idx, name in enumerate(self.collection_data.keys()) : 
            if idx == 0 : 
                desc = self.collection_data[name]['atoms'].get_array('milady-descriptors')
                array_design = np.zeros(( len(self.collection_data), desc.shape[1] ))
                array_design[idx,:] = np.sum(desc, axis=0)

            else : 
                desc = self.collection_data[name]['atoms'].get_array('milady-descriptors')
                array_design[idx,:] = np.sum(desc, axis=0)

        print('... Full design matrix is filled ...')
        print(f'... Shape of design matrix : {array_design.shape}')

        start = time.process_time()
        _, idx_col = self.descriptor_object.FindIndexCUR(array_design)
        end = time.process_time()
        
        print(f'... CUR selection is done : {len(idx_col)} columns are selected ({end - start}s) ...')
        return 


    def TestDescriptor(self, array_design : np.ndarray,
                       new_desc : np.ndarray,
                       idx : int,
                       tolerance : float =1e-13,
                       pression_bool : bool = False) -> bool : 
        
        array_design = np.delete(array_design, idx, 0)
        
        if pression_bool : 
            norm_matrix = np.linalg.norm((array_design[:,6:self.size_bso4+self.size_k2b+6] - new_desc[:self.size_bso4+self.size_k2b]), axis = 1)
        else : 
            norm_matrix = np.linalg.norm((array_design[:,:self.size_bso4+self.size_k2b] - new_desc[:self.size_bso4+self.size_k2b]), axis = 1)
        
        if (norm_matrix/np.linalg.norm(new_desc) < tolerance).any() : 
            return False
        else :
            return True

    def PrepareRegression(self, pression : bool = False,
                          data_name : str = 'array_anah_FE',
                          desc_bool : bool = False) -> Tuple[np.ndarray, Dict[float, np.ndarray], Dict[float, np.ndarray], Dict[float, np.ndarray], np.ndarray] :
        array_idx_config = np.zeros(len(self.collection_data))
        for id_config, config in enumerate(self.collection_data.keys()) : 
            desc_free_energy = self.descriptor_object.ComputeDescriptorFreeEnergy(self.collection_data[config]['atoms'])
            if id_config == 0 :
                print(f'... New descriptor dimension is {len(desc_free_energy)}')
                if not pression : 
                    self.design_matrix = np.zeros(( len(self.collection_data), len(desc_free_energy) ))
                else : 
                    self.design_matrix = np.zeros(( len(self.collection_data), len(desc_free_energy)+6))

            if not pression : 
                self.design_matrix[id_config,:] = desc_free_energy
            else : 
                pression_data = self.collection_data[config]['stress']*self.collection_data[config]['volume']
                self.design_matrix[id_config,:] = np.concatenate((pression_data*1e-6,desc_free_energy))
            
            try : 
                array_idx_config[id_config] = 100000 + int(config)
            except : 
                array_idx_config[id_config] = 100000 
            
            for id, temp in enumerate(self.dictionnary_target.keys()) : 
                if np.isnan(pression_data).any() : 
                    self.dictionnary_target[temp][id_config] = np.nan
                    self.dictionnary_errorF[temp][id_config] = np.nan
                    self.dictionnary_sigmaF[temp][id_config] = np.nan
                
                else :
                    self.dictionnary_target[temp][id_config] = self.collection_data[config][data_name][id]
                    self.dictionnary_errorF[temp][id_config] = np.abs(self.collection_data[config]['array_delta_FE'][id]/self.collection_data[config]['array_anah_FE'][id])
                    self.dictionnary_sigmaF[temp][id_config] = self.collection_data[config]['array_sigma_FE'][id]

                if desc_bool : 
                    if not self.TestDescriptor(self.design_matrix, desc_free_energy, id_config, pression_bool=pression) :
                        self.dictionnary_target[temp][id_config] = np.nan
                        self.dictionnary_errorF[temp][id_config] = np.nan 
                        self.dictionnary_sigmaF[temp][id_config] = np.nan             

        print('... Data are filled for regression ...')
        print(f'... Shape of the new design matrix is {self.design_matrix.shape}')

        return self.design_matrix, self.dictionnary_target, self.dictionnary_errorF, self.dictionnary_sigmaF ,array_idx_config

####################
## INPUTS
####################
path_pickle = '/home/lapointe/WorkML/FreeEnergySurrogate/full_data/it2/mab_desc_k2bj5_r6.pickle'
#path_pickle = '/home/lapointe/WorkML/FreeEnergySurrogate/data/svd_desc_mab_k2b50_j4.pickle'
nb_columns = 40
list_temperature = [300., 600., 900., 1200.]
error_cur = 1.0
desc_type = 'semi-cubicGG' #'lin'
pression_bool = True
data_name = 'array_full_FE'
#size_k2b, size_bso4 = 49, 55
size_k2b, size_bso4 = 50, 91

normalised_design_matrix = False
####################

collection_data : Dict[str, Data] = pickle.load(open(path_pickle,'rb'))
free_manager = FreeEnergyManager(collection_data,
                                nb_columns,
                                list_temperature,
                                desc_type=desc_type,
                                error_cur=error_cur,
                                mcd_analysis=False,
                                contamination=0.02  ,
                                size_k2b=size_k2b,
                                size_bso4=size_bso4)
design_matrix, target_dictionnary, error_dictionnary, sigma_dictionnary, array_idx_config = free_manager.PrepareRegression(pression=pression_bool, 
                                                                                                        data_name=data_name,
                                                                                                        desc_bool=True)

print()
for temp in error_dictionnary.keys() : 
    mask_nan = np.isnan(error_dictionnary[temp])
    print(f'... TEMPERATURE : {temp} K ...')
    print(f'mean error for delta_F : {np.mean(error_dictionnary[temp][~mask_nan])} U.A')
    print(f'max error for delta_F : {np.amax(error_dictionnary[temp][~mask_nan])} U.A')
    print()

mask_mcd = free_manager.mask_mcd
regressor_obj = RegressorFreeEnergy(design_matrix,
                                    target_dictionnary,
                                    error_dictionnary,
                                    sigma_dictionnary,
                                    array_idx_config,
                                    mask_mcd=mask_mcd,
                                    percentile=1.0,
                                    test_size=0.4,
                                    normalised_design_matrix=normalised_design_matrix)

regressor_obj.BuildAllModels(dim_lin=size_bso4+size_k2b+6)
#regressor_obj.BuildAllModels(dim_lin=None)