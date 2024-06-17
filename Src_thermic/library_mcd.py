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
class MCD_analysis_object : 

    def __init__(self, dbmodel : DBManager) -> None : 
        self.dic_class : Dict[str,Dict[str,List[Atoms]]] = {}
        self.mcd_model : Dict[str,MinCovDet]= {}
        self.pca_model : Dict[str,PCA] = {}
        self.distribution : Dict[str,Dict[str,tuple]] = {}

        for key in dbmodel.model_init_dic.keys() : 
            if key[0:6] in self.dic_class.keys() : 
                descriptors = dbmodel.model_init_dic[key]['atoms'].get_array('milady-descriptors')
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
        for atoms in list_atoms : 
            mcd_distance = self.mcd_model[species].mahalanobis(atoms.get_array('milady-descriptors'))  
            atoms.set_array('mcd-distance',np.sqrt(mcd_distance), dtype=float)

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

#######################################################
## Dfct analysis object
#######################################################
class Dfct_analysis_object : 

    def __init__(self, dbmodel : DBManager, extended_properties : List[str] = None, **kwargs) -> None : 
        self.dic_class : Dict[str,Dict[str,List[Atoms]]] = {}
        self.mcd_model : Dict[str,MinCovDet] = {}
        self.logistic_model : Dict[str,LogisticRegression] = {} 
        self.meta_data_model = []
        # sanity check !
        implemented_properies = {'local-energy':(1,),'atomic-volume':(1,),'coordination':(1,),'label-dfct':(1,)}
        for prop in extended_properties : 
            if prop not in implemented_properies :
                raise NotImplementedError('{} : this property is not yet implemented'.format(prop))

        for key in dbmodel.model_init_dic.keys() : 
            if key[0:6] in self.dic_class.keys() : 
                #voronoi analysis
                if 'atomic-volume' or 'coordination' in extended_properties : 
                    dbmodel.model_init_dic[key]['atoms'] = self.compute_Voronoi(dbmodel.model_init_dic[key]['atoms'])

                if 'label-dfct' in extended_properties : 
                    dic_nb_dfct = kwargs['dic_nb_dfct']
                    dbmodel.model_init_dic[key]['atoms'] = self._labeling_outlier_atoms(dbmodel.model_init_dic[key]['atoms'],dic_nb_dfct)

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
                
                if 'label-dfct' in extended_properties : 
                    dic_nb_dfct = kwargs['dic_nb_dfct']
                    dbmodel.model_init_dic[key]['atoms'] = self._labeling_outlier_atoms(dbmodel.model_init_dic[key]['atoms'],dic_nb_dfct)             

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
        return

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

    def DXA_analysis(self, atoms : Atoms, lattice_type : str, list_type : List[int], param_dxa : Dict[str,Any] = {}) : 
        ovito_config = ase_to_ovito(atoms)
        ovito_config.particles_.create_property('Structure Type',data=np.asarray(list_type))

        pipeline = Pipeline(source = StaticSource(data = ovito_config))
        dic_lattice = {'fcc':DislocationAnalysisModifier.Lattice.FCC,
                       'bcc':DislocationAnalysisModifier.Lattice.BCC,
                       'hcp':DislocationAnalysisModifier.Lattice.HCP,
                       'cubic_diamond':DislocationAnalysisModifier.Lattice.CubicDiamond,
                       'hexa_diamond':DislocationAnalysisModifier.Lattice.HexagonalDiamond}
        if lattice_type not in dic_lattice.keys() : 
            raise NotImplementedError('This lattice is not implemented : {:s}'.format(lattice_type))

        dic_param_dxa = {'circuit_stretchability':9,
                         'only_perfect_dislocations':False,
                        'trial_circuit_length':14}

        for key_dxa in param_dxa.keys() :
            dic_param_dxa[key_dxa] = param_dxa[key_dxa]

        DXA_modifier = DislocationAnalysisModifier(input_crystal_structure=dic_lattice[lattice_type],
                                                   circuit_stretchability=dic_param_dxa['circuit_stretchability'],
                                                   only_perfect_dislocations=dic_param_dxa['only_perfect_dislocations'],
                                                   trial_circuit_length=dic_param_dxa['trial_circuit_length'])
        pipeline.modifiers.append(DXA_modifier)
        data = pipeline.compute()
        for line in data.dislocations.lines:
            print("Dislocation %i: length=%f, Burgers vector=%s" % (line.id, line.length, line.true_burgers_vector))
            print(line.points)

    def setting_mcd_model(self, MCD_object : MCD_analysis_object) -> None : 
        """Loading MCD models from a previous bulk analysis
        
        Parameters:
        -----------

        MCD_object : MCD_analysis_object 
            Filled MCD_analysis_object from a bulk analysis

        """
        for species in MCD_object.mcd_model.keys() : 
            self.mcd_model[species] = MCD_object.mcd_model[species] 
        return

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
        """Plot the correlation between local volume and mcd distance
        
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

    
    def _labeling_outlier_atoms(self, atoms : Atoms, dic_nb_dfct : Dict[str,int]) -> Atoms : 
        """Make labelisation of atoms in system depending their energies
        
        Parameters:
        -----------

        atoms : Atoms 
            Atoms configuration to label 

        Returns: 
        --------

        Atoms 
            Atoms configuration with labels
        """
        
        dic_species_energy = {sp:[] for sp in atoms.symbols.species()}
        local_energy = atoms.get_array('local-energy') 
        for id_at, at in enumerate(atoms) : 
            dic_species_energy[at.symbol].append(local_energy[id_at])

        for sp in dic_species_energy : 
            array_energy_species = np.array(dic_species_energy[sp])
            array_energy_species = np.sort(array_energy_species)
            selected_energy = array_energy_species[-dic_nb_dfct[sp]]
            dic_species_energy[sp] = selected_energy
        
        label_array = []
        for id_at, at in enumerate(atoms) : 
            # case of bulk ! 
            if local_energy[id_at] < dic_species_energy[at.symbol] : 
                label_array.append(0)
            #calse of dfct ! 
            else :
                label_array.append(1)

        atoms.set_array('label-dfct',
                        np.array(label_array),
                        dtype=int)

        return atoms

    def fit_logistic_regressor(self, species : str, inputs_properties : List[str] = ['mcd-distance']) -> None : 
        """Adjust logistic regressor based on inputs_properties 
        
        Parameters:
        -----------

        species : str
            Species for the regressor 

        inputs_properties : List[str]
            List of properties used to adjust the logistic regressor, by default regressor is only based on mcd-distance

        """
        # sanity check !
        implemented_properies = {'local-energy':(1,),'atomic-volume':(1,),'coordination':(1,),'mcd-distance':(1,)}
        for prop in inputs_properties : 
            if prop not in implemented_properies :
                raise NotImplementedError('{} : this property is not yet implemented'.format(prop))


        self.logistic_model[species] = LogisticRegression()
        list_species_atoms = self._get_all_atoms_species(species)
        list_species_atoms = self._get_mcd_distance(list_species_atoms, species)

        Xdata = []
        Ytarget = []
        for atoms in list_species_atoms : 
            Ytarget.append(atoms.get_array('label-dfct')[0])
            Xdata.append([ atoms.get_array(prop)[0] for prop in inputs_properties ])
        
        Xdata = np.array(Xdata)
        Ytarget = np.array(Ytarget)
        self.logistic_model[species].fit(Xdata,Ytarget)
        
        #update meta logistic model
        if len(self.meta_data_model) == 0 :
            self.meta_data_model = inputs_properties

        print('Score for {:s} logistic regressor is : {:1.4f}'.format(species,self.logistic_model[species].score(Xdata,Ytarget)))
        return 

    def _predict_logistic(self, species : str, array_desc : np.ndarray) -> np.ndarray : 
        return self.logistic_model[species].predict_proba(array_desc)

    def one_the_fly_logistic_analysis(self, atoms : Atoms) -> Atoms :
        """Build one the fly mcd distances
        
        Parameters:
        -----------

        atoms : Atoms 
            Atoms object containing a given configuration

        Returns:
        --------

        Atoms 
            Updated Atoms object with new array "logistic-score" which corresponding to (N,nb_class)
            matrix (L) where N is the number of atom in the configuration and nb_class is the number of defect class
            then L_ij corresponding to the probability to have atom i to be part of the class j 
        """

        # setting extra arrays for atoms 
        dic_prop = {}
        for prop in self.meta_data_model :
            try : 
                atoms.get_array(prop) 
            except :  
                if prop == 'mcd-distance' : 
                    atoms = self.one_the_fly_mcd_analysis(atoms)
                if prop == 'atomic-volume' : 
                    atoms= self.compute_Voronoi(atoms)
            
            dic_prop[prop] = atoms.get_array(prop)

        
        list_logistic_score = []
        for id_at, at in enumerate(atoms) :
            array_data = np.array([[  dic_prop[prop][id_at] for prop in dic_prop.keys() ]]) 
            list_logistic_score.append( self._predict_logistic(at.symbol,array_data).flatten() )

        atoms.set_array('logistic-score',
                        np.array(list_logistic_score),
                        dtype=float)

        return atoms
    
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

#######################################################
## Defect class
#######################################################
class Cluster : 
    """Cluster class which contains all data about atomic defects found in a given configuration
    This class contains the present list of methods : 
        - append : update defect cluster with new atom object 
        - update_extension : update the spatial extension of the cluster
        - get_volume : return the volume of the cluster
        - estimation_dfct_number : return an estimation of the number of defect inside the cluster (working for point defects...)

    """
    def __init__(self, atom : Atom, rcut : float, array_property : Dict[str,Any] = {}) -> None : 
        """Init method cluster class 
        
        Parameters:
        -----------

        atom : Atom 
            First atom object indentifies to be part of the cluster 

        rcut : float 
            Initial size of the cluster 
        
        array_property : Dict[str,Any]
            Dictionnnary which contains additional data about atom in the cluster (atomic volume, mcd distance ...)

        """
        self.array_property = array_property
        self.atoms_dfct = Atoms()
        self.atoms_dfct.append(atom)
        self.rcut = rcut
        self.size = rcut
        self.elliptic = (1.0/(self.size**2))*np.eye(3)
        self.center = self.atoms_dfct.positions

    def append(self, atom : Atom, array_property : Dict[str,Any] = {}, elliptic : str ='iso') -> None : 
        """Append new atom in the cluster
        
        Parameters:
        -----------

        atom : Atom 
            New atom to put in the cluster 
        
        array_property : Dict[str,Any]
            Dictionnnary which contains additional data about atom in the cluster (atomic volume, mcd distance ...)

        """       
        self.atoms_dfct.append(atom)
        self.center = self.atoms_dfct.get_center_of_mass()
        if elliptic == 'iso' :
            self.size = self.update_extension()
            self._isotropic_extension()
        if elliptic == 'aniso' : 
            self._anistropic_extension()
        
        for prop in array_property.keys() : 
            self.array_property[prop] += array_property[prop]

    def update_extension(self) -> float : 
        """Update the spatial extension of the cluster 
        Should be changed for non isotropic defects ..."""
        return max([self.rcut, np.amax( [np.linalg.norm(pos - self.center) for pos in self.atoms_dfct.positions ] )])

    def _isotropic_extension(self) -> None : 
        """Distance covariance estimatior for isotropic clusters"""
        self.elliptic = (1.0/(self.size**2))*np.eye(3)
    
    def _anistropic_extension(self) -> None : 
        """Distance covariance estimatior for anisotropic clusters"""
        covariance = (self.atoms_dfct.positions - self.center).T@(self.atoms_dfct.positions - self.center)
        self.elliptic = np.linalg.pinv(covariance)

        # check the minimal isotropic raduis for each direction
        for i in range(self.elliptic.shape[0]) : 
            if self.elliptic[i,i] > 1.0/(self.rcut**2) :
                self.elliptic[i,i] = 1.0/(self.rcut**2)

    def get_elliptic_distance(self, atom : Atom) -> float :
        """Compute distance to the elliptic covariance distances envelop
        
        Parameters:
        -----------

        atom : Atom
            Atom object to compute the elliptic distance

        Returns:
        --------

        float : Elliptic distance
        """
        return np.sqrt((atom.position.flatten()-self.center.flatten())@self.elliptic@(atom.position.flatten()- self.center.flatten()).T)

    def get_volume(self) -> float : 
        """Get an estimation of the cluster volume 
        
        Returns:
        --------

        float : cluster volume 
        """
        return np.sum(self.array_property['atomic-volume'])
    
    def estimate_dfct_number(self, mean_atomic_volume : float) -> float :
        """Get an estimation of number of atom inside the cluster
        
        Returns:
        --------

        float : number of atom estimation
        """
        return len(self.atoms_dfct) - self.get_volume()/mean_atomic_volume