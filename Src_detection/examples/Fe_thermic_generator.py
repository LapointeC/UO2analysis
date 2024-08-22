import os, sys, shutil

sys.path.insert(0,'../')
from ..Src import HarmonicThermicGenerator

################################
### INPUTS
################################
configs_per_T = 10
list_temperature = [50.0,300.0,500.0]
writing_path = '{:s}/test_write'.format(os.getcwd())
structure_to_do = ['BCC','FCC','HCP','A15','C15']
size_to_do = [[4,4,4], [3,3,3], [4,4,4], [3,3,3], [2,2,2]]

################################
### LAMMPS INPUTS 
################################
lammps_dict = {'kind_pot':'eam/alloy',
               'pot_lammps':'/home/lapointe/WorkML/GenerateThermalConfig/pot_lammps/AM05.fs'}

if not os.path.exists(writing_path) : 
    os.mkdir(writing_path)
else : 
    shutil.rmtree(writing_path)
    os.mkdir(writing_path)


thermal_obj = HarmonicThermicGenerator(list_temperature, configs_per_T)

for struct, size in zip(structure_to_do, size_to_do) : 
    thermal_obj.GenerateThermicStructures(struct, 
                                          size, 
                                          2.8553, 
                                          lammps_dict['pot_lammps'],
                                          'in.lmp',
                                          'Fe',
                                          kind_potential=lammps_dict['kind_pot'],
                                          working_dir='./harmonic_vib',
                                          delta_xi=1e-3,
                                          relative_norm=1e-2,
                                          spherical_noise=False,
                                          scaling_factor=0.333)
dic_equiv, db_dic = thermal_obj.GenerateDBDictionnary()

print(dic_equiv)
thermal_obj.writer(db_dic, writing_path)