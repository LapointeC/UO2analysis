import os, sys
import glob
import pickle

sys.path.insert(0,'../')
from Src import DBManager, DBDictionnaryBuilder, \
                  Milady, Regressor, Optimiser, Descriptor, \
                  my_cfg_reader, \
                  MCDAnalysisObject

import matplotlib.pyplot as plt


plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

########################################################
### INPUTS
########################################################
path_bulk = '/home/lapointe/WorkML/UO2Analysis/data/thermique'
dic_sub_class = {'600K':'01_000','300K':'00_000'}
milady_compute = False
pickle_data_file = '../data/dataUO2.pickle'
pickle_model_file = 'UO2.pickle'
metric_type = 'mcd'
#########################################################

implemented_type = ['gmm', 'mcd']
if metric_type not in implemented_type : 
    raise NotImplementedError(f'{metric_type} is not implemented !')

if milady_compute : 
    Db_dic_builder = DBDictionnaryBuilder()

    md_list = glob.glob('{:}/**/*.cfg'.format(path_bulk),recursive=True)
    print('... Loading {:4d} configurations file for descriptors calculation ...'.format(len(md_list)))
    for md_file in md_list : 
        md_atoms = my_cfg_reader(md_file,extended_properties=None)
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

    if metric_type == 'mcd' : 
        analysis_mcd.fit_mcd_envelop('U',contamination=0.05)
        analysis_mcd.fit_mcd_envelop('O',contamination=0.05)
    
    elif metric_type == 'gmm' :
        analysis_mcd.fit_gmm_envelop('U',nb_bin_histo=100,nb_selected=2000,dict_gaussian={'n_components':3,
                                                                'covariance_type':'full',
                                                                'init_params':'k-means++', 
                                                                'max_iter':100,
                                                                'weight_concentration_prior_type':'dirichlet_process',
                                                                'weight_concentration_prior':0.8})
        analysis_mcd.fit_gmm_envelop('O',nb_bin_histo=100,nb_selected=2000,dict_gaussian={'n_components':3,
                                                                'covariance_type':'full',
                                                                'init_params':'k-means++', 
                                                                'max_iter':100,
                                                                'weight_concentration_prior_type':'dirichlet_process',
                                                                'weight_concentration_prior':0.8})
    print()
    print('... Writing pickle object ...')
    analysis_mcd.__dict__[f'{metric_type}_model']._write_pkl(f'{metric_type}_{pickle_model_file}')
    print('... Pickle object is written :) ...')
    plt.show()
