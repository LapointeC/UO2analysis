import os, sys
import glob
import pickle

sys.path.append(os.getcwd())

from ase import Atoms, Atom
from ase.io import read

sys.path.insert(0,'../../')
from Src import DBDictionnaryBuilder, DBManager, \
                  Milady, Optimiser, Regressor, Descriptor, \
                  DfctAnalysisObject, \
                  FrameOvito, MCDModifier

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

from ovito.pipeline import Pipeline, PythonSource


from typing import List
from PySide6.QtCore import QEventLoop
from PySide6.QtWidgets import QApplication

from typing import List, Dict
plt.rcParams['text.usetex'] = True


if not QApplication.instance():
    app = QApplication(sys.argv)

#######################################################

def change_species(atoms : Atoms, species : List[str]) -> Atoms : 
    """Just give the right type for atoms after lammps file reading ..."""
    for id_at, at in enumerate(atoms) : 
        at.symbol = species[id_at]
    return atoms 

########################################################
### INPUTS
########################################################
#path_dfct = '/home/lapointe/ToMike/data/shortlab-2021-Ti-Wigner-04b16ab/3_SIM/1_Annealing/Files/A92-10-8000-570.xyz'
path_dfct = '/home/lapointe/WorkML/TiAnalysis/pka_data'
dic_sub_class = {'pka_data':'00_000'}
milady_compute = False
pickle_data_file = 'data_pka.pickle'
pickle_model_file = 'MCD.pickle'


if milady_compute : 
    Db_dic_builder = DBDictionnaryBuilder()

    md_list = glob.glob('{:s}/*.xyz'.format(path_dfct))
    print('... Loading {:4d} configurations file for descriptors calculation ...'.format(len(md_list)))
    for md_file in md_list : 
        corresponding_sub_class = dic_sub_class[md_file.split('/')[-2]]
        try : 
            md_atoms = read(md_file,format='lammps-dump-text')
        except : 
            md_atoms = read(md_file,format='lammps-data',style='atomic')
        
        md_atoms = change_species(md_atoms,['Ti' for _ in range(len(md_atoms))])
        Db_dic_builder._update(md_atoms,corresponding_sub_class)

    # Full setting for milady
    dbmodel = DBManager(model_ini_dict=Db_dic_builder._generate_dictionnary())
    print('... All configurations have been embeded in Atoms object ...')
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
                          directory='mld_pka',
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
    previous_dbmodel : DBManager = pickle.load(open(pickle_data_file,'rb'))
    print('... Loading previous MCD model ...')
    dfct_analysis = DfctAnalysisObject(previous_dbmodel,extended_properties=['atomic-volume'])
    dfct_analysis.setting_mcd_model(pickle.load(open(pickle_model_file,'rb')))
    print('... MCD model is set ...')
    print()
    print('... Starting analysis ...')
    for id, key_conf in enumerate(previous_dbmodel.model_init_dic.keys()) : 
        print(' ... Analysis for : {:s} ...'.format(key_conf))
        atom_conf = dfct_analysis.one_the_fly_mcd_analysis(previous_dbmodel.model_init_dic[key_conf]['atoms'])
        dfct_analysis.VoronoiDistribution(atom_conf,'Ti',nb_bin=50)
        print('... Starting vacancies analysis ...')
        dfct_analysis.VacancyAnalysis(atom_conf,0.3, elliptic = 'iso')
        print('... Starting DXA analysis ...')
        dfct_analysis.DXA_analysis(atom_conf,'hcp')
        print(' ... MCD distances are filled for {:s} ...'.format(key_conf))

        if id == 0 : 
            print('... MCD distribution analysis ...')
            dfct_analysis.MCDDistribution(atom_conf,'Ti',nb_bin=100,threshold=0.05)
            atoms_config : List[Atoms] = [atom_conf]
            break

    frame_object = FrameOvito(atoms_config)
    logistic_modifier = MCDModifier(init_transparency=1.0,
                                    threshold_mcd=0.05,
                                    color_map='viridis')

    pipeline_config = Pipeline(source = PythonSource(delegate=frame_object))
    pipeline_config.modifiers.append(logistic_modifier.PerformMCDVolumeSelection)
    for frame in range(pipeline_config.source.num_frames) :
        data = pipeline_config.compute(frame)

    pipeline_config.add_to_scene()