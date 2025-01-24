import numpy as np
import pickle
import os

from typing import Dict, TypedDict
from ase import Atoms
from mld_src import DBManager, DBDictionnaryBuilder, \
                  Optimiser, Regressor, Descriptor, Milady, DescriptorsHybridation

##########################
## INPUTS
##########################
pickle_file = '/home/lapointe/WorkML/FreeEnergySurrogate/full_data/it2/desc_mab_Ff.pickle'
new_pickle_file = '/home/lapointe/WorkML/FreeEnergySurrogate/full_data/it2/mab_desc_k2bj5_r6.pickle'
##########################

class Data(TypedDict) : 
    array_temperature : np.ndarray
    array_ref_FE : np.ndarray 
    array_anah_FE : np.ndarray
    array_full_FE : np.ndarray
    array_sigma_FE : np.ndarray
    atoms : Atoms

data_object : Dict[str,Data] = pickle.load(open(pickle_file, 'rb'))
Db_dic_builder = DBDictionnaryBuilder()

print('... Loading {:4d} configurations file for descriptors calculation ...'.format(len(data_object)))
dic_equivalence = {}
id = 0
for config, data in data_object.items() :
    corresponding_sub_class = '00_000'
    Db_dic_builder._update(data['atoms'],corresponding_sub_class)
    dic_equivalence['{:}_{:}'.format(corresponding_sub_class,str(1000000+id+1)[1:])] = config
    id += 1

# Full setting for milady
dbmodel = DBManager(model_ini_dict=Db_dic_builder._generate_dictionnary())

print('... All configurations have been embeded in Atoms objects ...')
optimiser = Optimiser.Milady(fix_no_of_elements=1,
                             chemical_elements=['Fe'],
                             desc_forces=False)
regressor = Regressor.ComputeDescriptors(write_design_matrix=False)
descriptor_bso4 = Descriptor.BSO4(r_cut=6.0,j_max=5.0,lbso4_diag=False)
descriptor_k2b = Descriptor.Kernel2Body(r_cut=6.0,
                                       sigma_2b=0.3,
                                       delta_2b=1.0,
                                       np_radial_2b=50)

#descriptor = DescriptorsHybridation(descriptor_bso4, descriptor_k2b) 
#descriptor_g2 = Descriptor.G2(r_cut=5.3,n_g2_eta=2,n_g2_rs=50,eta_max_g2=0.8)

descriptor = DescriptorsHybridation(descriptor_bso4, descriptor_k2b)
#descriptor_bso4 

# command setup for milady
os.environ['MILADY_COMMAND'] = '/home/lapointe/Git/mld_build_intel/bin/milady_main.exe'
os.environ['MPI_COMMAND'] = 'mpirun -np'

# launch milady for descriptor computation
print('... Starting Milady ...')
mld_calc = Milady(optimiser,
                      regressor,
                      descriptor,
                      dbmodel=dbmodel,
                      directory='mld_j5_r6',
                      ncpu=1)

mld_calc.calculate(properties=['milady-descriptors'])
print('... Milady calculation is done ...')

new_data_object = data_object.copy()
for key, data in mld_calc.dbmodel.model_init_dic.items() : 
    new_data_object[dic_equivalence[key]]['atoms'] = data['atoms']

if os.path.exists(new_pickle_file) : 
    os.remove(new_pickle_file)
pickle.dump(new_data_object, open(new_pickle_file,'wb'))
print('... Pickle file with descriptors is filled ...')