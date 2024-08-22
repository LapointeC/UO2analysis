import glob
import pickle

import os, sys 
import pickle

sys.path.append(os.getcwd())
from matplotlib import colors

sys.path.insert(0,'../')
from ..Src import DBManager, DBDictionnaryBuilder, \
                  Milady, Descriptor, Regressor, Optimiser, \
                  DfctAnalysisObject, \
                  my_cfg_reader

########################################################
### INPUTS
########################################################
path_dfct = '/home/lapointe/WorkML/UO2Analysis/data/I2UO2/stab_test'
dic_sub_class = {'stab_test':'00_000'}
pickle_data_file = 'data_I2_min.pickle'
pickle_model_file = 'MCD.pickle'

Db_dic_builder = DBDictionnaryBuilder()
md_list = glob.glob('{:}/*.cfg'.format(path_dfct))

print('... Loading {:4d} configurations file for descriptors calculation ...'.format(len(md_list)))
for md_file in md_list : 
    md_atoms = my_cfg_reader(md_file,extended_properties=['displacement','local-energy'])
    corresponding_sub_class = dic_sub_class[md_file.split('/')[-2]]
    Db_dic_builder._update(md_atoms,corresponding_sub_class)
# Full setting for milady
dbmodel = DBManager(model_ini_dict=Db_dic_builder._generate_dictionnary())
print('... All configurations have been embeded in Atoms object ...')
optimiser = Optimiser.Milady(weighted=True,
                             fix_no_of_elements=2,
                             chemical_elements=['U','O'],
                             weight_per_element=[0.9,0.8],
                             desc_forces=False)
regressor = Regressor.ComputeDescriptors(write_design_matrix=False)
descriptor = Descriptor.BSO4(r_cut=6.0,j_max=4.0,lbso4_diag=False)

# command setup for milady
os.environ['MILADY_COMMAND'] = '/home/lapointe/Git/mld_build_intel/bin/milady_main.exe'
os.environ['MPI_COMMAND'] = 'mpirun -np'

# launch milady for descriptor computation
print('... Starting Milady ...')
mld_calc = Milady(optimiser,
                      regressor,
                      descriptor,
                      dbmodel=dbmodel,
                      directory='mld_I2_min',
                      ncpu=2)
mld_calc.calculate(properties=['milady-descriptors'])

print('... Milady calculation is done ...')
print('... Fill other properties for logistic regression ...')
dfct_object = DfctAnalysisObject(mld_calc.dbmodel,extended_properties=['coordination','atomic-volume'])

if os.path.exists(pickle_data_file) : 
    os.remove(pickle_data_file)
print('... Writing pickle object ...')
pickle.dump(mld_calc.dbmodel, open(pickle_data_file,'wb'))
print('... Pickle object is written :) ...')