import os, sys
import pickle

sys.path.insert(0,'../../')
from Src import   DBManager, \
                    Milady, Regressor, Optimiser, Descriptor, \
                    ThermicSampling, ThermicFiting
import time

######################################
## INPUTS
######################################
#path_data = '/home/lapointe/WorkML/GenerateThermalConfig/data/dynamical.h5'
path_data = '/home/lapointe/ToThomas/dynamical_thomas.h5'
scaling_factor, temperature =  0.5, 800.0
path_writing = './thermic_config'
symbol = 'W250H2'
#######################################


thermic_sampler = ThermicSampling(path_data,
                                  temperature,
                                  scaling_factor=scaling_factor,
                                  nb_sample=10,
                                  type_data='hdf5',
                                  symbols=symbol)
thermic_sampler.generate_harmonic_noise_configurations(path_writing)