import os, sys
import pickle

sys.path.insert(0,'../../')
sys.path.insert(0,'/home/marinica/GitHub/UO2analysis.git/Src_detection')

from Src import   DBManager, \
                    Milady, Regressor, Optimiser, Descriptor, \
                    ThermicSampling, ThermicFiting
import time

######################################
## INPUTS
######################################
#path_data = '/home/lapointe/WorkML/GenerateThermalConfig/data/dynamical.h5'
path_data = '/home/marinica/Work_ML/AlexandreThese/UnseenToken/Fe/put_noise/sia_111/dynamical.h5'
scaling_factor, temperature =  0.5, 300.0
path_writing = './thermic_config'
symbol = 'W250H2'
symbol= 'Fe769'
#######################################


thermic_sampler = ThermicSampling(path_data,
                                  temperature,
                                  scaling_factor=scaling_factor,
                                  nb_sample=10,
                                  type_data='hdf5',
                                  symbols=symbol)
thermic_sampler.generate_harmonic_noise_configurations(path_writing)

