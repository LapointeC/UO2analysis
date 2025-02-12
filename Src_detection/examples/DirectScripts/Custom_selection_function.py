import os, sys
import glob
import pickle
import numpy as np

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

from PySide6.QtCore import QEventLoop
from PySide6.QtWidgets import QApplication

from typing import List, Dict, Any
plt.rcParams['text.usetex'] = True


if not QApplication.instance():
    app = QApplication(sys.argv)

#######################################################

def change_species(atoms : Atoms, species : List[str]) -> Atoms : 
    """Just give the right type for atoms after lammps file reading ..."""
    for id_at, at in enumerate(atoms) : 
        at.symbol = species[id_at]
    return atoms 


def VacancySelectionFunction(atoms : Atoms,
                             dic_properties : Dict[str, Any]) -> List[int] : 
    """Custom function for vacancy selection"""
    threshold_mean_volume = dic_properties['atomic-volume']['threshold-std']
    voronoi = atoms.get_array('atomic-volume').flatten()
    mean_volume = np.mean(voronoi)

    threshold_mcd = dic_properties['mcd-distance']['threshold-std']
    mcd_distance = atoms.get_array('mcd-distance').flatten()
    max_mcd = np.amax(mcd_distance)

    # mask time !
    mask_volume = voronoi > (1.0 + threshold_mean_volume)*mean_volume
    mask_mcd = mcd_distance/max_mcd > threshold_mcd

    full_mask = mask_volume & mask_mcd
    return np.where(full_mask)[0]

def InterstitialSelectionFunction(atoms : Atoms,
                             dic_properties : Dict[str, Any]) -> List[int] : 
    """Custom function for interstitial selection"""
    threshold_mean_volume = dic_properties['atomic-volume']['threshold-std']
    voronoi = atoms.get_array('atomic-volume').flatten()
    mean_volume = np.mean(voronoi)

    threshold_mcd = dic_properties['mcd-distance']['threshold-std']
    mcd_distance = atoms.get_array('mcd-distance').flatten()
    max_mcd = np.amax(mcd_distance)

    # mask time !
    mask_volume = voronoi < (1.0 + threshold_mean_volume)*mean_volume
    mask_mcd = mcd_distance/max_mcd > threshold_mcd

    full_mask = mask_volume & mask_mcd
    return np.where(full_mask)[0]

#######################################################

########################################################
### INPUTS
########################################################
path_dfct = '/home/lapointe/WorkML/TiAnalysis/pka_data'
dic_sub_class = {'pka_data':'00_000'}
milady_compute = False
pickle_data_file = '/home/lapointe/WorkML/TiAnalysis/Src/data_pka.pickle'
pickle_model_file = '/home/lapointe/WorkML/TiAnalysis/Src/MCD.pickle'


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
        # Compute needed properies for vacancy analysis
        atom_conf = dfct_analysis.one_the_fly_mcd_analysis(previous_dbmodel.model_init_dic[key_conf]['atoms'])
        dfct_analysis.VoronoiDistribution(atom_conf,'Ti',nb_bin=50)

        print('... Starting vacancy analysis ...')
        dfct_analysis.PointDefectAnalysisFunction(atom_conf,
                                                  VacancySelectionFunction,
                                                  {'atomic-volume':{'threshold-std':0.1},
                                                   'mcd-distance':{'threshold-std':0.1}},
                                                  kind='vacancy',
                                                  elliptic='iso')
        print('... Starting interstitial analysis ...')
        dfct_analysis.PointDefectAnalysisFunction(atom_conf,
                                                  InterstitialSelectionFunction,
                                                  {'atomic-volume':{'threshold-std':0.1},
                                                   'mcd-distance':{'threshold-std':0.1}},
                                                  kind='vacancy',
                                                  elliptic='iso')

        dfct_analysis.GetAllPointDefectData('.point_dfct.data')
        print('... Analysis is done :) ...')
