import os
import numpy as np
import glob
import pickle

import random
from ase import Atoms, Atom

from sklearn.covariance import MinCovDet, EllipticEnvelope
from sklearn.decomposition import PCA
from milady import DBManager

import scipy.stats
from scipy.stats import gaussian_kde
from sklearn.linear_model import LogisticRegression, RidgeClassifier

from ovito.io.ase import ase_to_ovito
from ovito.modifiers import VoronoiAnalysisModifier, DislocationAnalysisModifier
from ovito.pipeline import StaticSource, Pipeline

import matplotlib
matplotlib.use('Agg')

from typing import List, Dict, Any
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

####################################################
# DICTIOANNARY BUILDER FOR MILADY
####################################################
class DBDictionnaryBuilder : 
    """Build the general dictionnary object to launch Milady calcuation, specific format of the dictionnay is
    detailed in DBManager doc object"""
    def __init__(self) : 
        self.dic : Dict[str,List[Atoms]] = {}

    def _builder(self, list_atoms : List[Atoms], list_sub_class : List[str] ) -> dict : 
        """Build the dictionnary from Atoms or List[Atoms] object"""
        for id_at, atoms in enumerate(list_atoms) :
            self._update(atoms, list_sub_class[id_at])
        
        return self._generate_dictionnary()

    def _update(self, atoms : Atoms, sub_class : str) -> None : 
        """Update entries of dictionnary"""
        if sub_class not in self.dic.keys() : 
            self.dic[sub_class] = [atoms]
        else : 
            self.dic[sub_class].append(atoms)     

    def _generate_dictionnary(self) -> dict :
        """Generate dictionnary object to launch Milady calcuation, specific format of the dictionnay is
        detailed in DBManager doc object"""
        dictionnary = {}
        for sub_class in self.dic.keys() : 
            for id, atoms in enumerate(self.dic[sub_class]) : 
                name_poscar = '{:}_{:}'.format(sub_class,str(1000000+id+1)[1:])
                dictionnary[name_poscar] = {'atoms':atoms,'energy':None,'energy':None,'forces':None,'stress':None}

        return dictionnary

####################################################
## SQUARE NORM DESCRIPTORS CLASS
####################################################
class NormDescriptorHistogram : 
    """Selection method based on histogram of square descriptor norm. This class allows 
    to build small size dataset for MCD fitting with the same statistical properties than the original one"""
    
    def __init__(self, list_atoms : List[Atoms], nb_bin : int = None) :
        """Build the square descriptor norm histogram for a given dataset.
        
        Parameters:
        -----------
        list_atoms : List[Atoms]
            Full dataset contains in Atoms object

        nb_bin : int 
            Number of bin for the histogram. If nb_bin is set to None, nb_bin = int(0.05*len(list_atoms)) by default
        
        """ 
        if nb_bin is None :
            nb_bin = int(0.05*len(list_atoms))

        self.list_atoms = list_atoms
        self.nb_bin = nb_bin
        list_norm = [np.linalg.norm(at.get_array('milady-descriptors'))**2 for at in self.list_atoms]
        self.bin = {}

        self.min_dis, self.max_dis = np.amin(list_norm), np.amax(list_norm)
        increment = (self.max_dis - self.min_dis)/nb_bin
        #here is the histogram generation procedure
        for k in range(nb_bin) : 
            self.bin[k] = {'min':self.min_dis+k*increment,
                           'max':self.min_dis+(k+1)*increment,
                           'list_norm':[],
                           'list_atoms':[],
                           'density':None}

        self.fill_histogram()

    def fill_histogram(self) :
        """Fill the histogram based square norm of descriptors"""
        for at in self.list_atoms : 
            norm_desc = np.linalg.norm(at.get_array('milady-descriptors'))**2
            for id_bin in self.bin.keys() : 
                if norm_desc > self.bin[id_bin]['min'] and norm_desc <= self.bin[id_bin]['max'] : 
                    self.bin[id_bin]['list_atoms'].append(at)
                    self.bin[id_bin]['list_norm'].append(norm_desc)
                    break

        # density correction for random choice
        sum_density = 0
        for id_bin in self.bin.keys() : 
            random.shuffle(self.bin[id_bin]['list_atoms'])
            self.bin[id_bin]['density'] = float(len(self.bin[id_bin]['list_norm']))/float(len(self.list_atoms))
            sum_density += float(len(self.bin[id_bin]['list_norm']))/float(len(self.list_atoms))
        
        miss_density = (1.0-sum_density)/(self.nb_bin)
        for id_bin in self.bin.keys() : 
            self.bin[id_bin]['density'] += miss_density

    def histogram_sample(self,nb_selected : int) -> List[Atoms] :
        """Histogram selection based on random choice function
        
        Parameters:
        -----------

        nb_selected : int 
            Number of local configurations to select 

        Returns:
        --------

        List[Atoms]
            Selected Atoms objects
        """
        list_atoms_selected = []
        selection_bin = np.random.choice([k for k in range(self.nb_bin)], nb_selected, p=[self.bin[id_bin]['density'] for id_bin in self.bin.keys()])
        for sl_bin in selection_bin : 
            if len(self.bin[sl_bin]['list_atoms']) > 0 :
                list_atoms_selected.append(self.bin[sl_bin]['list_atoms'][-1])
                self.bin[sl_bin]['list_atoms'].pop(-1)

        print('... Effective number of selected atoms is {:5d}/{:5d} ...'.format(len(list_atoms_selected),nb_selected))
        return list_atoms_selected

#######################################################
## MCD ANALYSIS OBJECT
#######################################################
class MCDAnalysisObject : 

    def __init__(self, dbmodel : DBManager) -> None : 
        self.dic_class : Dict[str,Dict[str,List[Atoms]]] = {}
        self.mcd_model : Dict[str,MinCovDet]= {}
        self.pca_model : Dict[str,PCA] = {}
        self.distribution : Dict[str,Dict[str,tuple]] = {}

        def fill_dictionnary(at : Atom, id_at : int, descriptors : np.ndarray, dic : Dict[str,List[Atoms]]) -> None : 
            atoms = Atoms([at])          
            atoms.set_array('milady-descriptors',descriptors[id_at,:].reshape(1,descriptors.shape[1]), dtype=float)
            dic[at.symbol] += [atoms]
            return 

        for key in dbmodel.model_init_dic.keys() : 
            if key[0:6] in self.dic_class.keys() : 
                descriptors = dbmodel.model_init_dic[key]['atoms'].get_array('milady-descriptors')
                [ fill_dictionnary(at, id_at, descriptors, self.dic_class[key[0:6]]) \
                 for id_at, at in enumerate(dbmodel.model_init_dic[key]['atoms']) ]

            else : 
                species = dbmodel.model_init_dic[key]['atoms'].symbols.species()
                dic_species : Dict[str,List[Atoms]] = {sp:[] for sp in species}
                descriptors = dbmodel.model_init_dic[key]['atoms'].get_array('milady-descriptors')
                [ fill_dictionnary(at, id_at, descriptors, dic_species) \
                 for id_at, at in enumerate(dbmodel.model_init_dic[key]['atoms']) ]

                self.dic_class[key[0:6]] = dic_species

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

    def _fit_mcd_model(self, list_atoms : List[Atoms], species : str, contamination : float = 0.05) -> None : 
        """Build the mcd model for a given species
        
        Parameters:
        -----------

        list_atoms : List[Atoms]
            List of Atoms objects to perform MCD analysis. Each Atoms object containing only 1 Atom with its 
            associated properties ...

        species : str
            Selected species 

        contamination : float
            percentage of outlier for mcd hyper-elliptic envelop fitting

        """
        self.mcd_model[species] = MinCovDet(support_fraction=1.0-contamination)
        descriptors_array = np.array([ atoms.get_array('milady-descriptors').flatten() for atoms in list_atoms ])
        self.mcd_model[species].fit(descriptors_array)

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
        
        def local_setting_mcd(atoms : Atoms) -> None : 
            mcd_distance = self.mcd_model[species].mahalanobis(atoms.get_array('milady-descriptors'))
            atoms.set_array('mcd-distance',np.sqrt(mcd_distance), dtype=float)

        [local_setting_mcd(atoms) for atoms in list_atoms]

        return list_atoms

    def _get_pca_model(self, list_atoms : List[Atoms], species : str, n_component : int = 2) -> np.ndarray : 
        """Build PCA model from data"""
        self.pca_model[species] = PCA(n_components=n_component)
        descriptors_array = np.array([ atoms.get_array('milady-descriptors').flatten() for atoms in list_atoms ])
        return self.pca_model[species].fit_transform(descriptors_array)


    def perform_mcd_analysis(self, species : str, contamination : float = 0.05, nb_selected=10000) -> None : 
        """Perform MCD analysis for a given species"""
        list_atom_species = self._get_all_atoms_species(species)
        print()
        print('... Starting histogram procedure ...')
        histogram_norm_species = NormDescriptorHistogram(list_atom_species,nb_bin=100)
        list_atom_species = histogram_norm_species.histogram_sample(nb_selected=nb_selected)
        print('... Histogram selection is done ...')
        print()

        print('... Starting MCD fit for {:} atoms ...'.format(species))
        self._fit_mcd_model(list_atom_species, species, contamination=contamination)
        print('... MCD envelop is fitted ...')
        updated_atoms = self._get_mcd_distance(list_atom_species,species)

        #mcd distribution 
        fig, axis = plt.subplots(nrows=1, ncols=2, figsize=(14,6))
        list_mcd = [at.get_array('mcd-distance').flatten()[0] for at in updated_atoms]
        n, _, patches = axis[0].hist(list_mcd,density=True,bins=50,alpha=0.7)
        for i in range(len(patches)):
            patches[i].set_facecolor(plt.cm.viridis(n[i]/max(n)))        

        #chi2 fit
        dist = getattr(scipy.stats, 'chi2')
        param = dist.fit(list_mcd)
        fake_mcd = np.linspace(np.amin(list_mcd), np.amax(list_mcd), 1000) #
        axis[0].plot(fake_mcd, dist(*param).pdf(fake_mcd), linewidth=1.5, linestyle='dashed',color='grey')
        self.distribution[species] = {'chi2':param}

        axis[0].set_xlabel(r'MCD distance $d_{\textrm{MCD}}$ for %s atoms'%(species))
        axis[0].set_ylabel(r'Probability density')

        #pca ! 
        cm = plt.cm.get_cmap('gnuplot')
        print('')
        print('... Starting PCA analysis for {:s} atoms ...'.format(species))
        desc_transform = self._get_pca_model(list_atom_species, species, n_component=2)
        scat = axis[1].scatter(desc_transform[:,0],desc_transform[:,1],
                               c=list_mcd,
                               cmap=cm,
                               edgecolors='grey',
                               linewidths=0.5,
                               alpha=0.5)
        print('... PCA analysis is done ...'.format(species))
        axis[1].set_xlabel(r'First principal component for %s atoms'%(species))
        axis[1].set_ylabel(r'Second principal component for %s atoms'%(species))
        cbar = fig.colorbar(scat,ax=axis[1])
        cbar.set_label(r'MCD disctances $d_{\textrm{MCD}}$', rotation=270)
        plt.tight_layout()

        plt.savefig('{:s}_distribution_analysis.pdf'.format(species),dpi=300)

#######################################################
    
def custom_writer(atoms : Atoms, path : str, property : str = 'mcd-distance',**kwargs) : 

    if property == 'logistic-score' : 
        class_to_plot = kwargs['class_to_plot']

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
        atoms_property = atoms.get_array(property)
        if property == 'logistic-score' :
            atoms_property = atoms_property[:,class_to_plot]

        for id, pos in enumerate(atoms.get_positions()) : 
            w.write('{:5d} {:2d} {:3.9f} {:3.9f} {:3.9f} {:3.9f} \n'.format(id+1,dic[atoms[id].symbol],atoms_property[id],pos[0],pos[1],pos[2]))
    return