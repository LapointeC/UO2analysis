import os
import numpy as np
import glob
import pickle

import random
from ase import Atoms, Atom
from milady import * 
from create_inputs import *
from milady_writer import *
from my_cfg_reader import my_cfg_reader
from sklearn.covariance import MinCovDet, EllipticEnvelope

import matplotlib.pyplot as plt

from typing import List, Dict

####################################################
## Histogramms
####################################################
class NormDescriptorHistogram : 

    def __init__(self, list_atoms : List[Atoms], nb_bin : int = None) : 
        if nb_bin is None :
            nb_bin = int(0.05*len(list_atoms))

        self.list_atoms = list_atoms
        self.nb_bin = nb_bin
        list_norm = [np.linalg.norm(at.get_array('milady-descriptors'))**2 for at in self.list_atoms]
        self.bin = {}

        self.min_dis, self.max_dis = np.amin(list_norm), np.amax(list_norm)
        increment = (self.max_dis - self.min_dis)/nb_bin
        for k in range(nb_bin) : 
            self.bin[k] = {'min':self.min_dis+k*increment,
                           'max':self.min_dis+(k+1)*increment,
                           'list_norm':[],
                           'list_atoms':[],
                           'density':None}

        self.fill_histogram()

    def fill_histogram(self) :
        for at in self.list_atoms : 
            norm_desc = np.linalg.norm(at.get_array('milady-descriptors'))**2
            for id_bin in self.bin.keys() : 
                if norm_desc > self.bin[id_bin]['min'] and norm_desc <= self.bin[id_bin]['max'] : 
                    self.bin[id_bin]['list_atoms'].append(at)
                    self.bin[id_bin]['list_norm'].append(norm_desc)
                    break

        sum_density = 0
        for id_bin in self.bin.keys() : 
            random.shuffle(self.bin[id_bin]['list_atoms'])
            self.bin[id_bin]['density'] = float(len(self.bin[id_bin]['list_norm']))/float(len(self.list_atoms))
            sum_density += float(len(self.bin[id_bin]['list_norm']))/float(len(self.list_atoms))
        
        miss_density = (1.0-sum_density)/(self.nb_bin)
        for id_bin in self.bin.keys() : 
            self.bin[id_bin]['density'] += miss_density

    def histogram_sample(self,nb_selected : int) -> List[Atoms] :
        list_atoms_selected = []
        selection_bin = np.random.choice([k for k in range(self.nb_bin)], nb_selected, p=[self.bin[id_bin]['density'] for id_bin in self.bin.keys()])
        for sl_bin in selection_bin : 
            if len(self.bin[sl_bin]['list_atoms']) > 0 :
                list_atoms_selected.append(self.bin[sl_bin]['list_atoms'][-1])
                self.bin[sl_bin]['list_atoms'].pop(-1)

        print('... Effective number of selected atoms is {:5d}/{:5d} ...'.format(len(list_atoms_selected),nb_selected))
        return list_atoms_selected

####################################################
# Dictionnary build for milady 
####################################################
class DBDictionnaryBuilder : 

    def __init__(self) : 
        self.dic : Dict[str,List[Atoms]] = {}

    def _builder(self, list_atoms : List[Atoms], list_sub_class : List[str] ) -> dict : 
        for id_at, atoms in enumerate(list_atoms) :
            self._update(atoms, list_sub_class[id_at])
        
        return self._generate_dictionnary()

    def _update(self, atoms : Atoms, sub_class : str) -> None : 
        if sub_class not in self.dic.keys() : 
            self.dic[sub_class] = [atoms]
        else : 
            self.dic[sub_class].append(atoms)     

    def _generate_dictionnary(self) -> dict :
        dictionnary = {}
        for sub_class in self.dic.keys() : 
            for id, atoms in enumerate(self.dic[sub_class]) : 
                name_poscar = '{:}_{:}'.format(sub_class,str(1000000+id+1)[1:])
                dictionnary[name_poscar] = {'atoms':atoms,'energy':None,'energy':None,'forces':None,'stress':None}

        return dictionnary

#######################################################
## MCD analysis object
#######################################################
class MCD_analysis_object : 

    def __init__(self, dbmodel : DBManager) -> None : 
        self.dic_class : Dict[str,Dict[str,List[Atoms]]] = {}
        self.mcd_model : Dict[str,MinCovDet]= {}

        for key in dbmodel.model_init_dic.keys() : 
            if key[0:6] in self.dic_class.keys() : 
                for id_at, at in enumerate(dbmodel.model_init_dic[key]['atoms']) :
                    atoms = Atoms()
                    atoms.append(at)          
                    atoms.set_array('milady-descriptors',descriptors[id_at,:].reshape(1,descriptors.shape[1]), dtype=float)
                    self.dic_class[key[0:6]][at.symbol].append(atoms)

            else : 
                species = dbmodel.model_init_dic[key]['atoms'].symbols.species()
                dic_species : Dict[str,List[Atoms]] = {sp:[] for sp in species}
                descriptors = dbmodel.model_init_dic[key]['atoms'].get_array('milady-descriptors')
                for id_at, at in enumerate(dbmodel.model_init_dic[key]['atoms']) :
                    atoms = Atoms()
                    atoms.append(at)          
                    atoms.set_array('milady-descriptors',descriptors[id_at,:].reshape(1,descriptors.shape[1]), dtype=float)
                    dic_species[at.symbol].append(atoms)

                self.dic_class[key[0:6]] = dic_species

    def _get_all_atoms_species(self, species : str) -> List[Atoms] : 
        """Create the full list of Atoms for a given species"""
        list_atoms_species = []
        for sub_class in self.dic_class.keys() : 
            list_atoms_species += self.dic_class[sub_class][species]
        return list_atoms_species

    def _fit_mcd_model(self, list_atoms : List[Atoms], species : str, contamination : float = 0.05) -> None : 
        """Build the mcd model for a given species"""
        self.mcd_model[species] = MinCovDet(support_fraction=1.0-contamination)
        #self.mcd_model[species] = EllipticEnvelope(contamination=contamination)
        descriptors_array = np.array([ atoms.get_array('milady-descriptors').flatten() for atoms in list_atoms ])
        self.mcd_model[species].fit(descriptors_array)

    def _get_mcd_distance(self, list_atoms : List[Atoms], species : str) -> List[Atoms] :
        """Compute mcd distances based for a given species and return updated Atoms objected with new array : mcd-distance"""
        for atoms in list_atoms : 
            mcd_distance = self.mcd_model[species].mahalanobis(atoms.get_array('milady-descriptors'))  
            atoms.set_array('mcd-distance',mcd_distance, dtype=float)

        return list_atoms

    def perform_mcd_analysis(self, species : str, contamination : float = 0.05) -> None : 
        """Perform MCD analysis for a given species"""
        list_atom_species = self._get_all_atoms_species(species)
        print()
        print('... Starting histogram procedure ...')
        histogram_norm_species = NormDescriptorHistogram(list_atom_species,nb_bin=100)
        list_atom_species = histogram_norm_species.histogram_sample(nb_selected=10000)
        print('... Histogram selection is done ...')
        print()

        print('... Starting MCD fit for {:} atoms ...'.format(species))
        self._fit_mcd_model(list_atom_species, species, contamination=contamination)
        print('... MCD envelop is fitted ...')
        updated_atoms = self._get_mcd_distance(list_atom_species,species)

        plt.figure()
        list_mcd = [at.get_array('mcd-distance') for at in updated_atoms]
        plt.hist(list_mcd,bins=30,density=True)
        plt.xlabel(r'MCD distance for {:s} atoms'.format(species))
        plt.ylabel(r'Probability density')
        plt.savefig('{:s}_dmcd_distribution.pdf'.format(species),dpi=300)
        plt.show()

#######################################################


########################################################
### INPUTS
########################################################
path_bulk = '/home/lapointe/WorkML/UO2Analysis/data/thermique'
dic_sub_class = {'600K':'01_000','300K':'00_000'}
milady_compute = False
pickle_file = 'dataUO2.pickle'

#path_bulk = '/home/lapointe/WorkML/UO2Analysis/data/I1UO2'

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
                          directory='mld',
                          ncpu=2)

    mld_calc.calculate(properties=['milady-descriptors'])
    print('... Milady calculation is done ...')

    if os.path.exists(pickle_file) : 
        os.remove(pickle_file)
    print('... Writing pickle object ...')
    pickle.dump(mld_calc.dbmodel, open(pickle_file,'wb'))
    print('... Pickle object is written :) ...')

else : 
    print('... Starting from the previous pickle file ...')
    previous_dbmodel = pickle.load(open(pickle_file,'rb'))
    analysis_mcd = MCD_analysis_object(previous_dbmodel)
    analysis_mcd.perform_mcd_analysis('U',contamination=0.05)