import glob
import pickle

import os, sys , shutil
import pickle

sys.path.append(os.getcwd())

sys.path.insert(0,'../')
from Src import DBManager, DBDictionnaryBuilder, \
                  Milady, Descriptor, Regressor, Optimiser, \
                  DfctAnalysisObject, \
                  my_cfg_reader

########################################################
### INPUTS
########################################################
path_dfct = '/home/lapointe/WorkML/UO2Analysis/data/I1UO2/cols_test'
dic_sub_class = {'cols_test':'00_000'}
milady_compute = False
metric_type = 'mcd'
pickle_data_file = '../data/UO2data_dfct_col.pickle'
pickle_model_file = '../data/mcd_UO2.pickle'
pickle_logistic = 'logistic.pickle'
########################################################

implemented_type = ['gmm', 'mcd']
if metric_type not in implemented_type : 
    raise NotImplementedError(f'{metric_type} is not implemented !')

if milady_compute : 
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
                          directory='mld_dfct_col',
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
    print('... DBmanager data is stored ...')
    print()
    print('... Starting defect analysis ...')
    analysis_dfct = DfctAnalysisObject(previous_dbmodel,
                                         extended_properties=['local-energy','atomic-volume','coordination','label-dfct'],
                                         dic_nb_dfct={'U':1,'O':2})
    if metric_type == 'mcd' :
        analysis_dfct.setting_mcd_model(pickle_model_file)
        print('... MCD models are stored ...')
        print()
        print('... Analysis for U ...')    
        analysis_dfct.fit_logistic_regressor('U',inputs_properties=['mcd-distance','atomic-volume','coordination'])
        print('... Analysis for O ....')    
        analysis_dfct.fit_logistic_regressor('O',inputs_properties=['mcd-distance','atomic-volume','coordination'])
        print('... Writing pkl file ...')
        analysis_dfct.mcd_model._write_pkl(f'{metric_type}_{pickle_logistic}')

    elif metric_type == 'gmm' :
        analysis_dfct.setting_gmm_model(pickle_model_file)
        print('... MCD models are stored ...')
        print()
        print('... Analysis for U ...')    
        analysis_dfct.fit_logistic_regressor('U',inputs_properties=['gmm-distance','atomic-volume','coordination'])
        print('... Analysis for O ....')    
        analysis_dfct.fit_logistic_regressor('O',inputs_properties=['gmm-distance','atomic-volume','coordination'])
        print('... Writing pkl file ...')
        analysis_dfct.gmm_model._write_pkl(f'{metric_type}_{pickle_logistic}')    
