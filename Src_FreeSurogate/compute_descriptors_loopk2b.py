import numpy as np
import pickle
import os, shutil

from typing import Dict, TypedDict
from ase import Atoms
from mld_src import DBManager, DBDictionnaryBuilder, \
                  Optimiser, Regressor, Descriptor, Milady, DescriptorsHybridation

##########################
## INPUTS
##########################
pickle_file = '/home/lapointe/WorkML/FreeEnergySurrogate/data/desc_mab_Ff.pickle'
dir_pickle = '/home/lapointe/WorkML/FreeEnergySurrogate/data/full_k2b'
grid_k2b_sigma = np.linspace(0.2,1.2,num=10)
grid_k2b_rcut = np.linspace(4.5,6.5,num=5)
np_radial = 50
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


if os.path.exists(dir_pickle) : 
    shutil.rmtree(dir_pickle)
os.mkdir(dir_pickle)


print('... All configurations have been embeded in Atoms objects ...')
optimiser = Optimiser.Milady(fix_no_of_elements=1,
                             chemical_elements=['Fe'],
                             desc_forces=False)


# loop on grid
# command setup for milady
os.environ['MILADY_COMMAND'] = '/home/lapointe/Git/mld_build_intel/bin/milady_main.exe'
os.environ['MPI_COMMAND'] = 'mpirun -np'

# launch milady for descriptor computation
print('... Starting Milady ...')
print(f'... Total number of descriptor calculation is {len(grid_k2b_rcut)*len(grid_k2b_sigma)}')
compt = 0
for rcut in grid_k2b_rcut : 
    for sigma in grid_k2b_sigma :
        compt += 1 
        print(f' ... Couple of parameter : rcut = {rcut}, sigma = {sigma}')

        # Full setting for milady
        dbmodel = DBManager(model_ini_dict=Db_dic_builder._generate_dictionnary())
        print('... All configurations have been embeded in Atoms objects ...')
        optimiser = Optimiser.Milady(fix_no_of_elements=1,
                                     chemical_elements=['Fe'],
                                     desc_forces=False)
        
        regressor = Regressor.ComputeDescriptors(write_design_matrix=False)
        descriptor_bso4 = Descriptor.BSO4(r_cut=rcut,j_max=0.5,lbso4_diag=False)
        descriptor_k2b = Descriptor.Kernel2Body(r_cut=rcut,
                                               sigma_2b=sigma,
                                               delta_2b=1.0,
                                               np_radial_2b=np_radial)

        descriptor = DescriptorsHybridation(descriptor_bso4, descriptor_k2b) 
        mld_calc = Milady(optimiser,
                              regressor,
                              descriptor,
                              dbmodel=dbmodel,
                              directory='mld',
                              ncpu=1)

        mld_calc.calculate(properties=['milady-descriptors'])

        new_data_object = data_object.copy()
        for key, data in mld_calc.dbmodel.model_init_dic.items() : 
            new_data_object[dic_equivalence[key]]['atoms'] = data['atoms']
        pickle.dump(new_data_object, open(f'{dir_pickle}/mab_{compt}.pickle','wb'))

print('... Milady calculation is done ...')
print('... Pickle file with descriptors is filled ...')