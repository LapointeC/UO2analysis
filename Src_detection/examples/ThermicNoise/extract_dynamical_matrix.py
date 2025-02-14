import sys
import h5py

sys.path.insert(0,'../../')
from Src import DataPhondy 

###########################
## INPUTS
###########################
root_dir_ph = ''
njob = 1
h5_file_path = './dynamical.h5'
############################

file = h5py.File(h5_file_path, 'w')
# Create a group for potentials
dynamical_group = file.create_group('dynamical')

#Reader object
data_ph = DataPhondy(root_dir_ph)
print('... Starting storage of dynamical matrices ...')
data_ph.GenerateDataParallel(dynamical_group,
                             njob=njob)
print('... Storage is Done ...')