import os
import numpy as np
import glob
import pickle
import shutil

from ase import Atoms
from milady import * 
from create_inputs import *
from milady_writer import *
from my_cfg_reader import my_cfg_reader

from sklearn.covariance import MinCovDet
from scipy.stats import gaussian_kde

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from library_mcd import DBDictionnaryBuilder, MCD_analysis_object

from ovito.io.ase import ase_to_ovito
from ovito.modifiers import VoronoiAnalysisModifier
from ovito.pipeline import StaticSource, Pipeline

from typing import List, Dict
plt.rcParams['text.usetex'] = True


def custom_writer(atoms : Atoms, path : str) : 
    dic = {el:k+1 for k, el in enumerate(atoms.symbols.species())}
    cell = atoms.cell[:]
    with open(path,'w') as w : 
        w.write('Custom writing ...\n')
        w.write('\n')
        w.write('{:5d} atoms \n'.format(atoms.get_global_number_of_atoms()))
        w.write('{:2d} atom types \n'.format(len(dic)))
        w.write('{:1.9f} {:3.9f} xlo xhi \n'.format(0,cell[0,0]))
        w.write('{:1.9f} {:3.9f} ylo zhi \n'.format(0,cell[1,1]))
        w.write('{:1.9f} {:3.9f} zlo zhi \n'.format(0,cell[2,2]))
        w.write('\n')
        w.write('\n')
        w.write('Atoms \n')
        w.write('\n')
        mcd_distance = atoms.get_array('mcd-distance')
        for id, pos in enumerate(atoms.get_positions()) : 
            w.write('{:5d} {:2d} {:3.9f} {:3.9f} {:3.9f} {:3.9f} \n'.format(id+1,dic[atoms[id].symbol],mcd_distance[id],pos[0],pos[1],pos[2]))

#######################################################
## Dfct analysis object
#######################################################
class Dfct_analysis_object : 

    def __init__(self, dbmodel : DBManager, extended_properties : List[str] = None) -> None : 
        self.dic_class : Dict[str,Dict[str,List[Atoms]]] = {}
        self.mcd_model : Dict[str,MinCovDet] = {}

        # sanity check !
        implemented_properies = {'local-energy':(1,),'atomic-volume':(1,),'coordination':(1,)}
        for prop in extended_properties : 
            if prop not in implemented_properies :
                raise NotImplementedError('{:s} : this property is not yet implemented'.format(prop))

        for key in dbmodel.model_init_dic.keys() : 
            if key[0:6] in self.dic_class.keys() : 
                #voronoi analysis
                if 'atomic-volume' or 'coordination' in extended_properties : 
                    dbmodel.model_init_dic[key]['atoms'] = self.compute_Voronoi(dbmodel.model_init_dic[key]['atoms'])

                descriptors = dbmodel.model_init_dic[key]['atoms'].get_array('milady-descriptors')
                for id_at, at in enumerate(dbmodel.model_init_dic[key]['atoms']) :
                    atoms = Atoms()
                    atoms.append(at)          
                    atoms.set_array('milady-descriptors',descriptors[id_at,:].reshape(1,descriptors.shape[1]), dtype=float)
                    if extended_properties is not None : 
                        for property in extended_properties : 
                            property_value = dbmodel.model_init_dic[key]['atoms'].get_array(property)[id_at]
                            atoms.set_array(property,
                                            property_value.reshape(1,),#len(property_value)),
                                            dtype=float)
                    self.dic_class[key[0:6]][at.symbol].append(atoms)

            else : 
                # voronoi analysis
                if 'atomic-volume' or 'coordination' in extended_properties : 
                    dbmodel.model_init_dic[key]['atoms'] = self.compute_Voronoi(dbmodel.model_init_dic[key]['atoms'])
                
                species = dbmodel.model_init_dic[key]['atoms'].symbols.species()
                dic_species : Dict[str,List[Atoms]] = {sp:[] for sp in species}
                descriptors = dbmodel.model_init_dic[key]['atoms'].get_array('milady-descriptors')
                for id_at, at in enumerate(dbmodel.model_init_dic[key]['atoms']) :
                    atoms = Atoms()
                    atoms.append(at)          
                    atoms.set_array('milady-descriptors',descriptors[id_at,:].reshape(1,descriptors.shape[1]), dtype=float)
                    if extended_properties is not None : 
                        for property in extended_properties :
                            property_value = dbmodel.model_init_dic[key]['atoms'].get_array(property)[id_at]
                            atoms.set_array(property,
                                            property_value.reshape(1,),
                                            dtype=float)
                    dic_species[at.symbol].append(atoms)

                self.dic_class[key[0:6]] = dic_species

    def compute_Voronoi(self, atoms : Atoms) -> Atoms : 
        """Compute atomic volume and coordination based on Ovito Voronoi analysis
        
        Parameters:
        -----------

        atoms : Atoms
            Atoms object corresponding to a given configuration

        Returns:
        --------

        Atoms 
            Updated Atoms object with new arrays : atomic-volume, coordination
        """
        ovito_config = ase_to_ovito(atoms)
        pipeline = Pipeline(source = StaticSource(data = ovito_config))
        voro = VoronoiAnalysisModifier(
            compute_indices = True,
            use_radii = False,
            edge_threshold = 0.1)
        pipeline.modifiers.append(voro)
        data = pipeline.compute()

        atoms.set_array('atomic-volume',
                        data.particles['Atomic Volume'][:],
                        dtype=float)
        atoms.set_array('coordination',
                        data.particles['Coordination'][:],
                        dtype=int)
        
        return atoms


    def setting_mcd_model(self, MCD_object : MCD_analysis_object) -> None : 
        """Loading MCD models from a previous bulk analysis
        
        Parameters:
        -----------

        MCD_object : MCD_analysis_object 
            Filled MCD_analysis_object from a bulk analysis

        """
        for species in MCD_object.mcd_model.keys() : 
            self.mcd_model[species] = MCD_object.mcd_model[species] 

    def _get_all_atoms_species(self, species : str) -> List[Atoms] : 
        """Create the full list of Atoms for a given species
        
        Parameters:
        -----------

        species : str
            Species to select 

        Returns:
        --------

        List[Atoms]
            List of selected Atoms objects based on species
        """
        list_atoms_species = []
        for sub_class in self.dic_class.keys() : 
            list_atoms_species += self.dic_class[sub_class][species]
        return list_atoms_species

    def _get_mcd_distance(self, list_atoms : List[Atoms], species : str) -> List[Atoms] :
        """Compute mcd distances based for a given species and return updated Atoms objected with new array : mcd-distance
        
        Parameters:
        -----------

        list_atoms : List[Atoms]
            List of Atoms objects where mcd distance will be computed

        species : str
            Species associated to list_atoms

            
        Returns:
        --------

        List[Atoms]
            Updated List of Atoms with the new array "mcd-distance"
        """
        for atoms in list_atoms : 
            mcd_distance = self.mcd_model[species].mahalanobis(atoms.get_array('milady-descriptors'))  
            atoms.set_array('mcd-distance',np.sqrt(mcd_distance), dtype=float)

        return list_atoms

    def one_the_fly_mcd_analysis(self, atoms : Atoms) -> Atoms :
        """Build one the fly mcd distances
        
        Parameters:
        -----------

        atoms : Atoms 
            Atoms object containing a given configuration
        """

        list_mcd = []
        descriptor = atoms.get_array('milady-descriptors')
        for id_at, at in enumerate(atoms) : 
            list_mcd.append( np.sqrt(self.mcd_model[at.symbol].mahalanobis(descriptor[id_at,:].reshape(1,descriptor.shape[1]))  ) )
        
        atoms.set_array('mcd-distance',
                        np.array(list_mcd).reshape(len(list_mcd),),
                        dtype=float)

        return atoms



    def mcd_energy_correlation(self, species : str) -> None : 
        """Plot the correlation between local energy and mcd distance
        
        Parameters:
        -----------

        species : str
            Selected species

        """
        list_atom_species = self._get_all_atoms_species(species)
        self._get_mcd_distance(list_atom_species, species)

        mcd = np.array([ at.get_array('mcd-distance').flatten() for at in list_atom_species])
        energy = np.array([ at.get_array('local-energy').flatten() for at in list_atom_species])

        mcd = mcd.flatten()
        energy = energy.flatten()
        
        plt.figure()
        energy_mcd = np.vstack([energy,mcd])
        kernel_density = gaussian_kde(energy_mcd)(energy_mcd)

        plt.scatter(energy,
                    mcd,
                    c=kernel_density,
                    alpha=0.5,
                    s=16,
                    edgecolors='grey',
                    linewidths=0.5)
        plt.xlabel(r'Local energy for {:s} atoms in eV'.format(species))
        plt.ylabel(r'MCD distances for {:s} atoms'.format(species))
        plt.savefig('energy_mcd_correlation_{:s}.pdf'.format(species),dpi=300)

    
    def volume_energy_correlation(self, species : str) -> None : 
        """Plot the correlation between local energy and mcd distance
        
        Parameters:
        -----------

        species : str
            Selected species

        """
        list_atom_species = self._get_all_atoms_species(species)
        self._get_mcd_distance(list_atom_species, species)

        volume = np.array([ at.get_array('atomic-volume').flatten() for at in list_atom_species])
        energy = np.array([ at.get_array('local-energy').flatten() for at in list_atom_species])

        volume = volume.flatten()
        energy = energy.flatten()
        
        plt.figure()
        energy_mcd = np.vstack([energy,volume])
        kernel_density = gaussian_kde(energy_mcd)(energy_mcd)

        plt.scatter(energy,
                    volume,
                    c=kernel_density,
                    alpha=0.5,
                    s=16,
                    edgecolors='grey',
                    linewidths=0.5)
        plt.xlabel(r'Local energy for {:s} atoms in eV'.format(species))
        plt.ylabel(r'Atomic volume for {:s} atoms in \AA$^3$'.format(species))
        plt.savefig('energy_volume_correlation_{:s}.pdf'.format(species),dpi=300)

########################################################
### INPUTS
########################################################
path_dfct = '/home/lapointe/WorkML/UO2Analysis/data/I1UO2/cols_test'
dic_sub_class = {'cols_test':'00_000'}
milady_compute = False
pickle_data_file = 'data_dfct_col.pickle'
pickle_model_file = 'MCD.pickle'


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
                                 weight_per_element=[0.9,0.8])
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
    analysis_dfct = Dfct_analysis_object(previous_dbmodel, extended_properties=['local-energy','atomic-volume'])
    analysis_dfct.setting_mcd_model(pickle.load(open(pickle_model_file,'rb')))
    print('... MCD models are stored ...')
    print()
    print('... Analysis for U ...')
    #analysis_dfct.mcd_energy_correlation('U')
    analysis_dfct.volume_energy_correlation('U')
    print('... Analysis for O ....')
    analysis_dfct.volume_energy_correlation('O')
    #analysis_dfct.mcd_energy_correlation('O')
    
    #ovito time ...
    from ase.io import write
    if not os.path.exists('{:s}/ovito'.format(os.getcwd())) :
        os.mkdir('{:s}/ovito'.format(os.getcwd()))
    else : 
        shutil.rmtree('{:s}/ovito'.format(os.getcwd()))
        os.mkdir('{:s}/ovito'.format(os.getcwd()))

    for key in previous_dbmodel.model_init_dic.keys() : 
        atoms_config = analysis_dfct.one_the_fly_mcd_analysis(previous_dbmodel.model_init_dic[key]['atoms'])
        custom_writer(atoms_config,'{:s}/ovito/{:s}.dump'.format(os.getcwd(),key))