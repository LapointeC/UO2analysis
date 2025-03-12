import pickle
import os
import glob

from typing import Dict, TypedDict
from ase import Atoms
from ase.io import read
from typing import List

from SrcMld import DBManager, DBDictionnaryBuilder, \
                  Optimiser, Regressor, Descriptor, Milady, DescriptorsHybridation

#####################################
## INPUTS
#####################################
# path
path_config = os.path.expandvars('./vasp_poscars')
pickle_file = os.path.expandvars('./feFPA.pkl')
#pickle_file = 'path/to/pkl'

path_config = os.path.abspath(path_config)
pickel_file = os.path.abspath(pickle_file)


# format
format = 'poscar'

#ml 
species = 'Fe'
class_mld ='00_000'

#descriptor
hybridation = False

# command setup for milady
os.environ['MILADY_COMMAND'] = 'path/to/milady.exe'
os.environ['MPI_COMMAND'] = 'ccc_mprun'
######################################


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]

def change_species(atoms: Atoms, species: List[str]) -> Atoms:
    """
    Update the Atoms object so that each atom gets the symbol provided in the species list.
    
    Parameters
    ----------
    atoms : Atoms
        An ASE Atoms object.
    species :  str
        A list of symbols, one for each atom.
    
    Returns
    -------
    Atoms
        The updated Atoms object.
    """
    print(f"{species}{len(atoms)}")
    atoms.set_chemical_symbols(f"{species}{len(atoms)}")
    return atoms

print("Initializing DB Dictionary Builder ...")
db_dic_builder = DBDictionnaryBuilder()

#old # Build the file pattern using the given directory and file extension.
#old pattern = os.path.join(path_config, '**', f'*.{format}')
#old md_list = glob.glob(pattern, recursive=True)

# Build the file pattern using the given directory and file extension.
pattern = os.path.join(path_config, '**', f'*.{format}')

# Get the list of files
md_list = glob.glob(pattern, recursive=True)

# Sort the files naturally
files = sorted(md_list, key=natural_sort_key)
md_list = files

print(md_list)
print(f"... Loading {len(md_list):4d} configuration files for descriptors calculation ...")
 
# Define the allowed file extensions and a mapping to ASE read formats.
allowed_formats = {'cfg', 'poscar', 'data', 'xyz', 'dump'}
format_mapping = {
    'cfg':    'cfg',  # same as before
    'dump':   'lammps-dump-text',
    'poscar': 'vasp',
    'data':   'lammps-data',
    'xyz':    'xyz',}
for md_file in md_list:
    # Get the file extension (without the dot) in lowercase.
    ext = os.path.splitext(md_file)[1].lstrip('.').lower()
    if ext not in allowed_formats:
        print(f"File {md_file} has extension '{ext}' which is not in the allowed list, skipping...")
        continue

    # Get the appropriate format string for ASE.read, defaulting to 'lammps-dump-text' if not mapped.
    read_format = format_mapping.get(ext, 'lammps-dump-text')
    try:
        md_atoms = read(md_file, format=read_format)
    except Exception as e:
        print(f"Error reading file {md_file} with format '{read_format}': {e}")
        continue

    # Update the atoms with the correct species.
    md_atoms = change_species(md_atoms, species)
    
    # Add the configuration to the DB dictionary with a default label.
    db_dic_builder._update(md_atoms, class_mld)
     
print("... All configurations have been embedded in Atoms objects ...")

# Full setting for milady
dbmodel = DBManager(model_ini_dict=db_dic_builder._generate_dictionnary())

print('... All configurations have been embeded in Atoms objects ...')
optimiser = Optimiser.Milady(fix_no_of_elements=1,
                             chemical_elements=['Fe'],
                             desc_forces=False)
regressor = Regressor.ComputeDescriptors(write_design_matrix=False)
descriptor_bso4 = Descriptor.BSO4(r_cut=5.0,j_max=4.0,lbso4_diag=False)

if hybridation :
    descriptor_k2b = Descriptor.Kernel2Body(r_cut=5.0,
                                           sigma_2b=0.5,
                                           delta_2b=1.0,
                                           np_radial_2b=50)

    descriptor = DescriptorsHybridation(descriptor_bso4, descriptor_k2b)

else : 
    descriptor = descriptor_bso4

# launch milady for descriptor computation
print('... Starting Milady ...')
mld_calc = Milady(optimiser,
                      regressor,
                      descriptor,
                      dbmodel=dbmodel,
                      directory='./mld_calculation',
                      ncpu=None)

mld_calc.calculate(properties=['milady-descriptors'])
print('... Milady calculation is done ...')

if os.path.exists(pickle_file) : 
    os.remove(pickle_file)
pickle.dump(mld_calc.dbmodel, open(pickle_file,'wb'))
print('... Pickle file with descriptors is filled ...')
