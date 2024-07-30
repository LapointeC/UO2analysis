import os, sys
import numpy as np
import glob
import pickle
import shutil

sys.path.append(os.getcwd())

from ase import Atoms, Atom
from milady import * 
from create_inputs import *
from milady_writer import *

from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors

matplotlib.use('Agg')
from library_mcd import DBDictionnaryBuilder, MCDAnalysisObject

from ovito.io.ase import ase_to_ovito
from ovito.pipeline import Pipeline, PythonSource, PipelineSourceInterface, StaticSource
from ovito.io import *
from ovito.data import DataCollection

from ovito.modifiers import VoronoiAnalysisModifier, DislocationAnalysisModifier, SelectTypeModifier, CommonNeighborAnalysisModifier

from typing import List, Dict
from PySide6.QtCore import QEventLoop
from PySide6.QtWidgets import QApplication

from typing import List, Dict
plt.rcParams['text.usetex'] = True


if not QApplication.instance():
    app = QApplication(sys.argv)

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
        if len(self.atoms_dfct.positions) > 1 :
            covariance = (self.atoms_dfct.positions - self.center).T@(self.atoms_dfct.positions - self.center)/(len(self.atoms_dfct.positions)-1)
            self.elliptic = np.linalg.pinv(covariance)

            id_to_remove = []
            # check the minimal isotropic raduis for each direction
            for i in range(self.elliptic.shape[0]) : 
                if np.linalg.norm(self.elliptic[i,:]) > 1.0/(self.rcut**2) :
                    for j in range(self.elliptic.shape[1]) : 
                        if i == j :
                            self.elliptic[i,j] = 1.0/(self.rcut**2)
                        else : 
                            id_to_remove.append([i,j])
                            id_to_remove.append([j,i])

            for id_rem in id_to_remove : 
                self.elliptic[id_rem[0],id_rem[1]] = 0.0
        
        else : 
            self.elliptic = (1.0/(self.size**2))*np.eye(3)

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

#######################################################
## Dfct analysis object
#######################################################
class Dfct_analysis_object : 

    def __init__(self, dbmodel : DBManager, extended_properties : List[str] = None, **kwargs) -> None : 
        self.dic_class : Dict[str,Dict[str,List[Atoms]]] = {}
        self.mcd_model : Dict[str,MinCovDet] = {}
        self.distribution_model : Dict[str,Dict[str,tuple]] = {}
        self.logistic_model : Dict[str,LogisticRegression] = {} 
        self.meta_data_model = []
        self.dfct : Dict[str,Dict[str,Cluster]] = {'vacancy':{},'interstial':{},'dislocation':{}}

        # sanity check !
        implemented_properies = {'local-energy':(1,),'atomic-volume':(1,),'coordination':(1,),'label-dfct':(1,)}
        if extended_properties is not None : 
            for prop in extended_properties : 
                if prop not in implemented_properies :
                    raise NotImplementedError('{} : this property is not yet implemented'.format(prop))

        for key in dbmodel.model_init_dic.keys() : 
            if key[0:6] in self.dic_class.keys() : 
                if extended_properties is None : 
                    extended_properties = []

                #voronoi analysis
                if 'atomic-volume' in extended_properties or 'coordination' in extended_properties : 
                    dbmodel.model_init_dic[key]['atoms'] = self.compute_Voronoi(dbmodel.model_init_dic[key]['atoms'])

            else : 
                if extended_properties is None : 
                    extended_properties = []
                # voronoi analysis
                if 'atomic-volume' in extended_properties or 'coordination' in extended_properties : 
                    dbmodel.model_init_dic[key]['atoms'] = self.compute_Voronoi(dbmodel.model_init_dic[key]['atoms'])           

        return

    def update_dfct(self, key_dfct : str, atom : Atom, array_property : Dict[str,Any] = {}, rcut : float = 4.0, elliptic : str = 'iso') :
        """Method to update defect inside dictionnary"""
        if self.dfct[key_dfct] == {} : 
            self.dfct[key_dfct]['0'] = Cluster(atom, rcut, array_property=array_property)
        
        else : 

            key_closest = np.argmin([np.linalg.norm( atom.position - cluster.center) for _, cluster in self.dfct[key_dfct].items()])
            if self.dfct[key_dfct][key_closest].get_elliptic_distance(atom) < 1.5 :
                self.dfct[key_dfct][key_closest].append(atom, array_property=array_property, elliptic = elliptic)
            
            else : 
                next_index = str(max([int(key) for key in self.dfct[key_dfct].keys()]) + 1)
                self.dfct[key_dfct][next_index] = Cluster(atom, rcut, array_property=array_property)           

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


    def setting_mcd_model(self, MCD_object : MCDAnalysisObject) -> None : 
        """Loading MCD models from a previous bulk analysis
        
        Parameters:
        -----------

        MCD_object : MCD_analysis_object 
            Filled MCD_analysis_object from a bulk analysis

        """
        def local_mcd_model(species, MCD_object : MCDAnalysisObject) -> None :
            self.mcd_model[species] = MCD_object.mcd_model[species]
        
        #for species in MCD_object.mcd_model.keys() : 
        #    self.mcd_model[species] = MCD_object.mcd_model[species] 

        [ local_mcd_model(species, MCD_object) for species in MCD_object.mcd_model.keys() ]
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

        def local_setting_mcd(atoms : Atoms) -> None : 
            mcd_distance = self.mcd_model[species].mahalanobis(atoms.get_array('milady-descriptors'))
            atoms.set_array('mcd-distance',np.sqrt(mcd_distance), dtype=float)

        [local_setting_mcd(atoms) for atoms in list_atoms]

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
        list_mcd = [ np.sqrt(self.mcd_model[at.symbol].mahalanobis(descriptor[id_at,:].reshape(1,descriptor.shape[1]))) for id_at, at in enumerate(atoms)  ]
        
        atoms.set_array('mcd-distance',
                        np.array(list_mcd).reshape(len(list_mcd),),
                        dtype=float)

        return atoms

    def _get_pca_model(self, atoms : Atoms, n_component : int = 2) -> np.ndarray : 
        """Build PCA model from data"""
        pca_model = PCA(n_components=n_component)
        descriptors_array = atoms.get_array('milady-descriptors')
        return pca_model.fit_transform(descriptors_array)

    def mcd_distribution(self, atoms : Atoms, species : str, nb_bin : int = 20, threshold : float = 0.05) -> None : 
        """Perform MCD analysis for a given species"""

        #mcd distribution 
        fig, axis = plt.subplots(nrows=1, ncols=2, figsize=(14,6))
        list_mcd = atoms.get_array('mcd-distance').flatten()
        n, _, patches = axis[0].hist(list_mcd,density=True,bins=nb_bin,alpha=0.7)
        for i in range(len(patches)):
            patches[i].set_facecolor(plt.cm.viridis(n[i]/max(n)))

        axis[0].axvline(threshold*np.amax(list_mcd),min(n),max(10*n),
                   color='black',
                   linewidth=1.0,
                   zorder=10)
        axis[0].set_xlabel(r'MCD distance $d_{\textrm{MCD}}$ for %s atoms'%(species))
        axis[0].set_ylabel(r'Probability density')
        axis[0].set_yscale('log')

        #pca ! 
        cm = plt.cm.get_cmap('gnuplot')
        print('')
        print('... Starting PCA analysis for {:s} atoms ...'.format(species))
        desc_transform = self._get_pca_model(atoms, n_component=2)
        scat = axis[1].scatter(desc_transform[:,0],desc_transform[:,1],
                               c=list_mcd,
                               cmap=cm,
                               edgecolors='grey',
                               linewidths=0.5,
                               alpha=0.5,
                               rasterized=True)
        print('... PCA analysis is done ...'.format(species))
        axis[1].set_xlabel(r'First principal component for %s atoms'%(species))
        axis[1].set_ylabel(r'Second principal component for %s atoms'%(species))
        cbar = fig.colorbar(scat,ax=axis[1])
        cbar.set_label(r'MCD disctances $d_{\textrm{MCD}}$', rotation=270)
        plt.tight_layout()

        plt.savefig('{:s}_dfct_distribution_analysis.pdf'.format(species),dpi=300)

    def voronoi_distribution(self, atoms : Atoms, species : str, nb_bin : int = 20) -> None :
        """Plot atomic volume distribution ..."""
        atomic_volume = atoms.get_array('atomic-volume')
        plt.figure()
        n, _, patches = plt.hist(atomic_volume,density=True,bins=nb_bin,alpha=0.7)
        for i in range(len(patches)):
            patches[i].set_facecolor(plt.cm.viridis(n[i]/max(n)))
        
        plt.vlines(np.mean(atomic_volume),min(n),max(10*n),
                   color='black',
                   linestyles='dashed',
                   linewidth=1.0,
                   zorder=10)
        plt.xlabel(r'Atomic volume in $\AA$ for %s atoms'%(species))
        plt.ylabel(r'Probability density')
        plt.yscale('log')
        plt.savefig('atomic_vol.pdf', dpi=300)
        return

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

    def brut_force_vacancy_analysis(self, atoms : Atoms, mcd_threshold : float, elliptic : str = 'iso') -> None : 
        """Brut force analysis to localised vacancies (based on mcd score and atomic volume)
        
        Parameters:
        -----------

        atoms : Atoms 
            Atoms object to analyse 

        mcd_treshold : float
            Ratio mcd/max(mcd) to consider the presence of atomic defect

        """
        atomic_volume = atoms.get_array('atomic-volume').flatten()
        mcd_distance = atoms.get_array('mcd-distance').flatten()
        
        max_mcd = np.amax(mcd_distance)
        mean_atomic_volume = np.mean(atomic_volume)

        for id_atom, atom in enumerate(atoms) : 
            if mcd_distance[id_atom] > mcd_threshold*max_mcd and atomic_volume[id_atom] > mean_atomic_volume :
                self.update_dfct('vacancy', atom, array_property={'atomic-volume':[atomic_volume[id_atom]]}, rcut=4.0, elliptic = elliptic)
            
        for key in self.dfct['vacancy'].keys() :
            center = self.dfct['vacancy'][key].center.flatten()
            print('Vacancy cluster {:s} : nb vacancy {:2.1f}, positions : {:2.3f} {:2.3f} {:2.3f}'.format(key,
                                                                                                            self.dfct['vacancy'][key].estimate_dfct_number(mean_atomic_volume),
                                                                                                            center[0],
                                                                                                            center[1],
                                                                                                            center[2]))
            #print('Vacancy cluster {:s} : nb vacancy {:2.1f}'.format(key,self.dfct['vacancy'][key].estimate_dfct_number(mean_atomic_volume)))

    def brut_force_dislo_analysis(self, atoms : Atoms, mcd_threshold : float, lattice : str) -> None :
        atomic_volume = atoms.get_array('atomic-volume').flatten()
        mcd_distance = atoms.get_array('mcd-distance').flatten()
        
        max_mcd = np.amax(mcd_distance)
        mean_atomic_volume = np.mean(atomic_volume)

        list_type = []

        for id_atom, atom in enumerate(atoms) : 
            if mcd_distance[id_atom] > mcd_threshold*max_mcd and atomic_volume[id_atom] < mean_atomic_volume :
                list_type.append(0)
            else :
                list_type.append(3)
        
        print('my DXA')
        self.DXA_analysis(atoms, lattice, list_type)
        print('basic DXA')
        self.basic_DXA_analysis(atoms, lattice)
        exit(0)
###########################################################################
def change_species(atoms : Atoms, species : List[str]) -> Atoms : 
    """Just give the right type for atoms after lammps file reading ..."""
    for id_at, at in enumerate(atoms) : 
        at.symbol = species[id_at]
    return atoms 

########################################################
## FRAME INTERFACE FOR OVITO
########################################################
class FrameOvito(PipelineSourceInterface) : 
    """Define the PipelineSourceInterface given to ovito pipeline to 
    plot frame by frame the confiugrations"""
    def __init__(self, list_atoms : List[Atoms]) :
        self.list_atoms = list_atoms
        self.lenght = len(list_atoms)
        
    def compute_trajectory_length(self, **kwargs):
        """Needed method for pipeline"""
        return self.lenght
            
    def create(self, data : DataCollection, *, frame : int, **kwargs) :
        """Creator for DataCollection object inside the ovito pipeline"""
        data_atoms : DataCollection = ase_to_ovito(self.list_atoms[frame])
        data.particles = data_atoms.particles
        data.cell = data_atoms.cell

class MCDModifier :
    """Ovito modifier function for visual logistic selection"""
    def __init__(self, init_transparency : float = 0.9, threshold_mcd : float = 0.5, color_map : str = 'viridis') :
        self.init_transparency = init_transparency
        self.threshold_mcd = threshold_mcd
        self.color_map = plt.cm.get_cmap(color_map)

    def AssignColors(self, frame : int, data : DataCollection) :
        particules_type = data.particles['Particle Type'][:]
        color_array = np.empty((len(particules_type),3))
        for id ,type in enumerate(particules_type) :
            color_array[id, :] = self.dic_rgb[type]
        data.particles_.create_property('Color',data=color_array)

    def PerformMCDVolumeSelection(self, frame : int, data : DataCollection) :
        mcd_score = data.particles['mcd-distance'][:]
        atomic_volume = data.particles['atomic-volume'][:]

        mean_volume = np.mean(atomic_volume)
        max_mcd = np.amax(mcd_score)

        color_array = np.empty((len(mcd_score),3))
        array_transparency = np.empty((mcd_score.shape[0],))
        for id ,mcd in enumerate(mcd_score) :
            if (mcd/max_mcd) > self.threshold_mcd and atomic_volume[id] < mean_volume :
                array_transparency[id] = 0.2*(1.0 - mcd/max_mcd)
                color_array[id,:] = colors.to_rgb(self.color_map(mcd/max_mcd))
            else :
                array_transparency[id] = self.init_transparency
                color_array[id,:] = colors.to_rgb('grey')
        data.particles_.create_property('Transparency',data=array_transparency)
        data.particles_.create_property('Color',data=color_array)

    def PerformMCDSelection(self, frame : int, data : DataCollection) :
        mcd_score = data.particles['mcd-distance'][:]
        max_mcd = np.amax(mcd_score)

        color_array = np.empty((len(mcd_score),3))
        array_transparency = np.empty((mcd_score.shape[0],))
        for id ,mcd in enumerate(mcd_score) :
            if (mcd/max_mcd) > self.threshold_mcd :
                array_transparency[id] = 0.2*(1.0 - mcd/max_mcd)
                color_array[id,:] = colors.to_rgb(self.color_map(mcd/max_mcd))
            else :
                array_transparency[id] = self.init_transparency
                color_array[id,:] = colors.to_rgb('grey')
        data.particles_.create_property('Transparency',data=array_transparency)
        data.particles_.create_property('Color',data=color_array)

########################################################
### INPUTS
########################################################
#path_dfct = '/home/lapointe/ToMike/data/shortlab-2021-Ti-Wigner-04b16ab/3_SIM/1_Annealing/Files/A92-10-8000-570.xyz'
path_dfct = '/home/lapointe/WorkML/TiAnalysis/pka_data'
dic_sub_class = {'pka_data':'00_000'}
milady_compute = False
pickle_data_file = 'data_pka.pickle'
pickle_model_file = 'MCD.pickle'


if milady_compute : 
    Db_dic_builder = DBDictionnaryBuilder()

    md_list = glob.glob('{:s}/*.xyz'.format(path_dfct))
    print('... Loading {:4d} configurations file for descriptors calculation ...'.format(len(md_list)))
    for md_file in md_list : 
        corresponding_sub_class = dic_sub_class[md_file.split('/')[-2]]
        try : 
            md_atoms = read(md_file,format='lammps-dump-text')
        except : 
            md_atoms = read(md_file,format='lammps-data',style='atomic')
        
        md_atoms = change_species(md_atoms,['Ti' for _ in range(len(md_atoms))])
        Db_dic_builder._update(md_atoms,corresponding_sub_class)

    # Full setting for milady
    dbmodel = DBManager(model_ini_dict=Db_dic_builder._generate_dictionnary())
    print('... All configurations have been embeded in Atoms object ...')
    optimiser = Optimiser.Milady(fix_no_of_elements=1,
                                 chemical_elements=['Ti'],
                                 desc_forces=False)
    regressor = Regressor.ComputeDescriptors(write_design_matrix=False)
    descriptor = Descriptor.BSO4(r_cut=5.0,j_max=4.0,lbso4_diag=False)

    # command setup for milady
    os.environ['MILADY_COMMAND'] = '/home/lapointe/Git/mld_build_intel/bin/milady_main.exe'
    os.environ['MPI_COMMAND'] = 'mpirun -np'

    # launch milady for descriptor computation
    print('... Starting Milady ...')
    mld_calc = Milady(optimiser,
                          regressor,
                          descriptor,
                          dbmodel=dbmodel,
                          directory='mld_pka',
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
    print('... Loading previous MCD model ...')
    dfct_analysis = Dfct_analysis_object(previous_dbmodel,extended_properties=['atomic-volume'])
    dfct_analysis.setting_mcd_model(pickle.load(open(pickle_model_file,'rb')))
    print('... MCD model is set ...')
    print()
    print('... Starting analysis ...')
    for id, key_conf in enumerate(previous_dbmodel.model_init_dic.keys()) : 
        print(' ... Analysis for : {:s} ...'.format(key_conf))
        atom_conf = dfct_analysis.one_the_fly_mcd_analysis(previous_dbmodel.model_init_dic[key_conf]['atoms'])
        dfct_analysis.voronoi_distribution(atom_conf,'Ti',nb_bin=50)
        print('... Starting vacancies analysis ...')
        dfct_analysis.brut_force_vacancy_analysis(atom_conf,0.3, elliptic = 'iso')
        print('... Starting DXA analysis ...')
        dfct_analysis.brut_force_dislo_analysis(atom_conf,0.05,'hcp')
        print(' ... MCD distances are filled for {:s} ...'.format(key_conf))

        if id == 0 : 
            print('... MCD distribution analysis ...')
            dfct_analysis.mcd_distribution(atom_conf,'Ti',nb_bin=100,threshold=0.05)
            atoms_config : List[Atoms] = [atom_conf]
            break
        #else :
        #    atoms_config.append(atom_conf) 

    frame_object = FrameOvito(atoms_config)
    logistic_modifier = MCDModifier(init_transparency=1.0,
                                    threshold_mcd=0.05,
                                    color_map='viridis')

    pipeline_config = Pipeline(source = PythonSource(delegate=frame_object))
    pipeline_config.modifiers.append(logistic_modifier.PerformMCDVolumeSelection)
    for frame in range(pipeline_config.source.num_frames) :
        data = pipeline_config.compute(frame)

    pipeline_config.add_to_scene()