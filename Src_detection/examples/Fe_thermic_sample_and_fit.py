import os, sys
import pickle

sys.path.insert(0,'../')
from Src import   DBManager, \
                    Milady, Regressor, Optimiser, Descriptor, \
                    ThermicSampling, ThermicFiting, \
                    write_milady_poscar, nearest_mode, \
                    FastEquivariantDescriptor

import time

######################################
## INPUTS
######################################
dict_size = {'BCC':[4,4,4], 
             'FCC':[3,3,3], 
             'HCP':[4,4,4], 
             'A15':[3,3,3], 
             'C15':[2,2,2]}
path_data = '../data/dynamical.h5'
scaling_factor =  None 
"""{'BCC':0.333, 
 'FCC':0.333, 
 'HCP':0.333, 
 'A15':0.333, 
 'C15':0.333}"""
mode = 'thermic_fiting'
equivariant = True
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
                                      path_data,
                                      300.0,
                                      scaling_factor=scaling_factor,
                                      nb_sample=1000,
                                      type_data='hdf5')
    thermic_sampler.build_covariance_estimator(path_writing='./big_mat_ml_poscar',nb_sigma=1.5)
    print('... Pickle file for thermic covariance will be written ...')
    thermic_sampler.build_pickle(path_pickles='./big_mat_sampling.pkl')
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
    thermic_obj : ThermicSampling = pickle.load(open('big_mat_sampling.pkl','rb'))

    fit_obj = ThermicFiting()

    print('... Reading structures ...')
    if equivariant : 
        equivariant_obj = FastEquivariantDescriptor(dbmodel, rcut=6.0)
        
        start = time.process_time()
        equivariant_obj.BuildEquivariantDescriptors()
        end =  time.process_time()
        print(f'Descriptor time for {len(equivariant_obj.configurations)*1028} local env is {end-start} s')
        configurations = equivariant_obj.GetConfigurations()
        for struc, config in configurations.items() : 
            descriptor_struct = config['equiv_descriptor']
            print(struc)
            key_cov = [key for key in thermic_obj.atoms_assembly.ml_data.keys() if struc in thermic_obj.atoms_assembly.ml_data[key]['name_poscar']][0]
            print(' ... {:} structure ...'.format(key_cov))
            covariance_struct = thermic_obj.atoms_assembly.ml_data[key_cov]['covariance']
            #print(covariance_struct)
            fit_obj.update_fit_data(key_cov, descriptor_struct, covariance_struct)
            print(' ... {:} structure is filled ...'.format(key_cov))

    else :
        for struc in dbmodel.model_init_dic.keys() : 
            descriptor_struct = dbmodel.model_init_dic[struc]['atoms'].get_array('milady-descriptors')
            print(struc)
            #print([thermic_obj.atoms_assembly.ml_data[key]['name_poscar'] for key in thermic_obj.atoms_assembly.ml_data.keys()])
            key_cov = [key for key in thermic_obj.atoms_assembly.ml_data.keys() if struc in thermic_obj.atoms_assembly.ml_data[key]['name_poscar']][0]
            print(' ... {:} structure ...'.format(key_cov))
            covariance_struct = thermic_obj.atoms_assembly.ml_data[key_cov]['covariance']
            #print(covariance_struct)
            fit_obj.update_fit_data(key_cov, descriptor_struct, covariance_struct)
            print(' ... {:} structure is filled ...'.format(key_cov))

    fit_obj.build_regression_models('BCC')
    print(fit_obj.covariance_models)
