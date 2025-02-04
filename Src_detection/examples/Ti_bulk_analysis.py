import os, sys
import glob
import pickle

from ase import Atoms
from ase.io import read

sys.path.insert(0,'../')
from Src import DBManager, DBDictionnaryBuilder, \
                  Optimiser, Regressor, Descriptor, Milady, \
                  MCDAnalysisObject

import matplotlib.pyplot as plt

from typing import List
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

#######################################################

def change_species(atoms : Atoms, species : List[str]) -> Atoms : 
    """Just give the right type for atoms after lammps file reading ..."""
    for id_at, at in enumerate(atoms) : 
        at.symbol = species[id_at]
    return atoms 

########################################################
### INPUTS
########################################################
path_bulk = '/home/lapointe/WorkML/TiAnalysis/bulk_data'
dic_sub_class = {'600K':'00_000','900K':'01_000'}
milady_compute = True
pickle_data_file = 'dataTi_bulk.pickle'
pickle_model_file = 'MCD.pickle'
#########################################################

if milady_compute : 
    Db_dic_builder = DBDictionnaryBuilder()

    md_list = glob.glob('{:}/**/*.dump'.format(path_bulk),recursive=True)
    print('... Loading {:4d} configurations file for descriptors calculation ...'.format(len(md_list)))
    for md_file in md_list : 
        md_atoms = read(md_file,format='lammps-dump-text')
        md_atoms = change_species(md_atoms,['Ti' for _ in range(len(md_atoms))])
        corresponding_sub_class = dic_sub_class[md_file.split('/')[-2]]
        Db_dic_builder._update(md_atoms,corresponding_sub_class)

    # Full setting for milady
    dbmodel = DBManager(model_ini_dict=Db_dic_builder._generate_dictionnary())
    print('... All configurations have been embeded in Atoms objects ...')
    optimiser = Optimiser.Milady(fix_no_of_elements=1,
                                 chemical_elements=['Ti'],
                                 desc_forces=False)
    regressor = Regressor.ComputeDescriptors(write_design_matrix=False)
    descriptor = Descriptor.BSO4(r_cut=5.0,j_max=4.0,lbso4_diag=False)

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
    print('... Milady calculation is done ...')

    if os.path.exists(pickle_data_file) : 
        os.remove(pickle_data_file)
    print('... Writing pickle object ...')
    pickle.dump(mld_calc.dbmodel, open(pickle_data_file,'wb'))
    print('... Pickle object is written :) ...')

else : 
    print('... Starting from the previous pickle file ...')
    previous_dbmodel = pickle.load(open(pickle_data_file,'rb'))
    analysis_mcd = MCDAnalysisObject(previous_dbmodel)
    analysis_mcd.fit_mcd_envelop('Ti',contamination=0.05, nb_selected=10000)
    print()
    print('... Writing pickle object ...')
    pickle.dump(analysis_mcd, open(pickle_model_file,'wb'))
    print('... Pickle object is written :) ...')
    plt.show()
