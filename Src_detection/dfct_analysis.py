import numpy as np
import os
import more_itertools
from sklearn.linear_model import LogisticRegression
from sklearn.covariance import MinCovDet
from sklearn.mixture import BayesianGaussianMixture
from sklearn.neighbors import KernelDensity

from ase import Atoms, Atom

from typing import Dict, List , Any
from library_mcd import DBManager, MCDAnalysisObject
from cluster import Cluster, ClusterDislo
from dislocation_object import DislocationObject, reference_structure

from ovito.io.ase import ase_to_ovito
from ovito.modifiers import VoronoiAnalysisModifier, DislocationAnalysisModifier, CoordinationAnalysisModifier
from ovito.pipeline import StaticSource, Pipeline


#######################################################
## Dfct analysis object
#######################################################
class DfctAnalysisObject : 

    def __init__(self, dbmodel : DBManager, extended_properties : List[str] = None, **kwargs) -> None : 
        self.dic_class : Dict[str,Dict[str,List[Atoms]]] = {}
        self.mcd_model : Dict[str,MinCovDet] = {}
        self.gmm : Dict[str,BayesianGaussianMixture] = {}
        self.logistic_model : Dict[str,LogisticRegression] = {} 
        self.distribution : Dict[str,KernelDensity] = {}
        self.meta_data_model = []
        self.dfct : Dict[str, Dict[str, Cluster | ClusterDislo]] = {'vacancy':{},'interstial':{},'dislocation':{},'other':{}}

        self.mean_atomic_volume = None 

        def fill_dictionnary(key : str, at : Atom, id_at : int, descriptors : np.ndarray, extended_properties : List[str], dic : Dict[str,List[Atoms]]) -> None : 
            atoms = Atoms([at])          
            atoms.set_array('milady-descriptors',descriptors[id_at,:].reshape(1,descriptors.shape[1]), dtype=float)
            if extended_properties is not None : 
                for property in extended_properties : 
                    property_value = dbmodel.model_init_dic[key]['atoms'].get_array(property)[id_at]
                    atoms.set_array(property,
                                    property_value.reshape(1,),
                                    dtype=float)
                dic[at.symbol] += [atoms]
            return 

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
                [ fill_dictionnary(key, at, id_at, descriptors, extended_properties, self.dic_class[key[0:6]]) \
                 for id_at, at in enumerate(dbmodel.model_init_dic[key]['atoms']) ]
                
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
                [ fill_dictionnary(key, at, id_at, descriptors, extended_properties, dic_species) \
                 for id_at, at in enumerate(dbmodel.model_init_dic[key]['atoms']) ]

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
        voro = VoronoiAnalysisModifier(compute_indices = True,
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

    def StructureEstimator(self, atoms : Atoms, rcut : float = 5.0, nb_bin = 100) -> reference_structure : 
        """Agnostic reference structure identifier for dislocation analysis. This method is 
        based on CNA analysis from ```Ovito```.
        
        Parameters:
        -----------

        atoms : Atoms 
            Atoms system to analyse

        rcut : float 
            Cutoff radius for CNA analysis

        nb_bin : int 
            Number of bins used to build the RDF histogram

        Returns:
        --------

        reference_structure 
            ```reference_structure``` dictionnary needed for dislocation analysis

        """
        dictionnary_equivalent = {'bcc': lambda x : 2.0*x/np.sqrt(3.0)*np.eye(3),
                                  'fcc': lambda x : np.sqrt(2.0)*x*np.eye(3)}
        inv_dict_struct = {'other':0,
                           'fcc':1,
                           'hcp':2,
                           'bcc':3,
                           'ico':4}

        ovito_config = ase_to_ovito(atoms)
        pipeline = Pipeline(source = StaticSource(data = ovito_config))
        CNA = CoordinationAnalysisModifier(cutoff = rcut, number_of_bins = nb_bin) 
        pipeline.modifiers.append(CNA)
        data = pipeline.compute()
        rdf = data.tables['coordination-rdf'].xy()
        particule_type = data.particles['Coordination'][:]

        # guessing the structure from CNA ...
        types, count_types = np.unique(particule_type, return_counts=True)
        structure = inv_dict_struct[types[np.argmax(count_types)]]

        if structure not in dictionnary_equivalent.keys() : 
            raise NotImplementedError(f'{structure} is not implemented ...')

        #compute the maximum of radial distribution function
        max_idx = np.argmax(rdf[:,1])
        maximum_r = rdf[max_idx,0]

        return {'structure':structure, 'unit_cell':dictionnary_equivalent[structure](maximum_r)}

    def DXA_analysis(self, atoms : Atoms, lattice_type : str, param_dxa : Dict[str,Any] = {}) -> None : 
        """Perform classical DXA analysis from ```Ovito``` ...
        
        Parameters:
        -----------

        atoms : Atoms 
            Atoms system to analyse

        lattice_type : str 
            Reference cristallographic lattice of the system 

        param_dxa : Dict[str,Any]
            Optional dictionnary of parameters for DXA (see https://www.ovito.org/docs/current/reference/pipelines/modifiers/dislocation_analysis.html)
        """
        ovito_config = ase_to_ovito(atoms)
        #ovito_config.particles_.create_property('Structure Type',data=np.asarray(list_type))

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
        for species, models in MCD_object.mcd_model.items() : 
            self.mcd_model[species] = models 
            self.distribution[species] = MCD_object.distribution[species]
        return

    def setting_gmm_model(self, MCD_object : MCDAnalysisObject) -> None : 
        """Loading MCD models from a previous bulk analysis
        
        Parameters:
        -----------

        MCD_object : MCD_analysis_object 
            Filled MCD_analysis_object from a bulk analysis

        """
        for species, gmm in MCD_object.gmm.items() : 
            self.gmm[species] = gmm
            self.distribution[species] = gmm
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
        def local_filling(list_at_sp : List[Atoms], class_ml : Dict[str,Atoms], species) -> None :
            list_at_sp += class_ml[species]
            return 

        list_atoms_species = []
        [ local_filling(list_atoms_species,class_ml,species) for _, class_ml in self.dic_class.items() ]

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

    def _get_gmm_distance(self, list_atoms : List[Atoms], species : str) -> List[Atoms] :
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
        
        def mahalanobis_gmm(model : BayesianGaussianMixture, X : np.ndarray) -> np.ndarray : 
            mean_gmm = model.means_
            invcov_gmm = model.precisions_
            array_distance = np.empty((X.shape[0],invcov_gmm.shape[0]))
            for i in range(array_distance.shape[1]):
                array_distance[:,i] = np.sqrt((X-mean_gmm[i,:])@invcov_gmm[i,:,:]@(X-mean_gmm[i,:]).T)
            return array_distance

        for atoms in list_atoms : 
            gmm_distance = mahalanobis_gmm(self.gmm[species],atoms.get_array('milady-descriptors')) 
            atoms.set_array('gmm-distance',gmm_distance, dtype=float)

        return list_atoms

    def mahalanobis_gmm(self, model : BayesianGaussianMixture, X : np.ndarray) -> np.ndarray : 
        """Predict the distance array associated to gaussian mixture. Element i of the array 
        corresponding to d_i(X) the distance from X to the center of Gaussian i
        
        Parameters
        ----------

        model : BayesianGaussianMixture
            Gaussian mixture model 

        X : np.ndarray 
            Data to compute distances 

        Returns 
        -------

        np.ndarray 
            Distances array

        """
        mean_gmm = model.means_
        invcov_gmm = model.precisions_
        array_distance = np.empty((X.shape[0],invcov_gmm.shape[0]))
        for i in range(array_distance.shape[1]):
            array_distance[:,i] = np.sqrt((X-mean_gmm[i,:])@invcov_gmm[i,:,:]@(X-mean_gmm[i,:]).T)
        return array_distance

    def one_the_fly_mcd_analysis(self, atoms : Atoms) -> Atoms :
        """Build one the fly mcd distances
        
        Parameters:
        -----------

        atoms : Atoms 
            Atoms object containing a given configuration
        """

        descriptor = atoms.get_array('milady-descriptors')
        list_mcd = [ np.sqrt(self.mcd_model[at.symbol].mahalanobis(descriptor[id_at,:].reshape(1,descriptor.shape[1]))) for id_at, at in enumerate(atoms)  ]
        list_proba = [ self.distribution[at.symbol].score(list_mcd[id_at]) for id_at, at in enumerate(atoms)]

        atoms.set_array('mcd-distance',
                        np.array(list_mcd).reshape(len(list_mcd),),
                        dtype=float)
        atoms.set_array('probabilty',
                        np.array(list_proba).reshape(len(list_proba),),
                        dtype=float)

        return atoms

    def one_the_fly_gmm_analysis(self, atoms : Atoms) -> Atoms :
        """Build one the fly gmm distances
        
        Parameters:
        -----------

        atoms : Atoms 
            Atoms object containing a given configuration
        """

        list_gmmd = []
        list_proba = []
        descriptor = atoms.get_array('milady-descriptors')
        dim_gmm = None 
        for id_at, at in enumerate(atoms) : 
            descriptor_at = descriptor[id_at,:].reshape(1,descriptor.shape[1])
            gmmd = self.mahalanobis_gmm( self.gmm[at.symbol], descriptor_at )
            score = self.gmm[at.symbol].score(descriptor_at)
            if dim_gmm is None : 
                dim_gmm = gmmd.shape[1]

            list_gmmd += [gmmd]
            list_proba += [score]

        atoms.set_array('gmm-distance',
                        np.array(list_gmmd).reshape(len(list_gmmd),dim_gmm),
                        dtype=float)
        atoms.set_array('probability',                        
                        np.array(list_proba).reshape(len(list_proba),1),
                        dtype=float)
        return atoms
    
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
        
        label_array = [ 0 if local_energy[id_at] < dic_species_energy[at.symbols] else 1 \
                       for id_at, at in enumerate(atoms) ]

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
        implemented_properies = {'local-energy':(1,),'atomic-volume':(1,),'coordination':(1,),'mcd-distance':(1,),'gmm-distance':(1,)}
        for prop in inputs_properties : 
            if prop not in implemented_properies :
                raise NotImplementedError('{} : this property is not yet implemented'.format(prop))


        self.logistic_model[species] = LogisticRegression()
        list_species_atoms = self._get_all_atoms_species(species)

        if 'mcd-distance' in inputs_properties :
            list_species_atoms = self._get_mcd_distance(list_species_atoms, species)

        if 'gmm-distance' in inputs_properties : 
            list_species_atoms = self._get_gmm_distance(list_species_atoms, species)

        Xdata = []
        Ytarget = []
        for atoms in list_species_atoms : 
            Ytarget.append(atoms.get_array('label-dfct')[0])
            miss_shaped_X = [ atoms.get_array(prop)[0].tolist() for prop in inputs_properties ]
            Xdata.append(list(more_itertools.collapse(miss_shaped_X)))

        Xdata = np.array(Xdata)
        Ytarget = np.array(Ytarget)
        self.logistic_model[species].fit(Xdata,Ytarget)
        
        #update meta logistic model
        if len(self.meta_data_model) == 0 :
            self.meta_data_model = inputs_properties

        print('Score for {:s} logistic regressor is : {:1.4f}'.format(species,self.logistic_model[species].score(Xdata,Ytarget)))
        return 

    def _predict_logistic(self, species : str, array_desc : np.ndarray) -> np.ndarray : 
        """Predict the logistic score for a given array of descriptor
        
        Parameters:
        -----------

        species : str 
            Species associated to the descritpor array 

        array_desc : np.ndarray 
            Descriptor array (M,D)

        Returns:
        --------

        np.ndarray 
            Associated logistic score probabilities array (M,N_c) where N_c is the number of logistic classes
        """
        return self.logistic_model[species].predict_proba(array_desc)

    def one_the_fly_logistic_analysis(self, atoms : Atoms) -> Atoms :
        """Perfrom logistic regression analysis
        
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
                if prop == 'gmm-distance' : 
                    atoms = self.one_the_fly_gmm_analysis(atoms)

            dic_prop[prop] = atoms.get_array(prop)

        
        list_logistic_score = []
        for id_at, at in enumerate(atoms) :
            miss_shaped_data = [ dic_prop[prop][id_at].tolist() for prop in dic_prop.keys() ]
            array_data = np.array([ list(more_itertools.collapse(miss_shaped_data)) ])
            list_logistic_score.append( self._predict_logistic(at.symbol,array_data).flatten() )

        atoms.set_array('logistic-score',
                        np.array(list_logistic_score),
                        dtype=float)

        return atoms

###############################################################
### UPDATING POINT DEFECT CLUSTERS PART 
###############################################################
    def update_dfct(self, key_dfct : str, atom : Atom, array_property : Dict[str,Any] = {}, rcut : float = 4.0, elliptic : str = 'iso') -> None :
        """Method to update defect inside dictionnary
        
        Parameters:
        -----------

        key_dfct : str 
            Type of defect to update (this method manage only ```vacancy``` and ```interstitial```)

        atom : Atom
            Atom object to update in a ```Cluster``` object

        array_property : Dict[str,Any]
            Dictionnnary which contains additional data about atom in the cluster (atomic volume, mcd distance ...)

        rcut : float 
            Cut off raduis used as initial size for new ```Cluster```

        elliptic : str 
            Type of size estimation for ```Cluster```. 
            ```iso``` : isotropic cluster 
            ```aniso``` : elliptic cluster

        """
        if self.dfct[key_dfct] == {} : 
            self.dfct[key_dfct]['0'] = Cluster(atom, rcut, array_property=array_property)
        
        else : 
            key_closest = np.argmin([np.linalg.norm( atom.position - cluster.center) for _, cluster in self.dfct[key_dfct].items()])
            if self.dfct[key_dfct][key_closest].get_elliptic_distance(atom) < 1.5 :
                self.dfct[key_dfct][key_closest].append(atom, array_property=array_property, elliptic = elliptic)
            
            else : 
                next_index = str(max([int(key) for key in self.dfct[key_dfct].keys()]) + 1)
                self.dfct[key_dfct][next_index] = Cluster(atom, rcut, array_property=array_property) 

    def _aggregate_cluster(c1 : Cluster, c2 : Cluster) -> Cluster : 
        """Local method to aggregate two ```Cluster``` objects
        
        Parameters:
        -----------

        c1 : Cluster
            First ```Cluster``` to aggregate

        c2 : Cluster 
            Second ```Cluster``` to aggregate

        Returns:
        --------

        Cluster
            Aggregated ```Cluster``` from c1 and c2
        
        """
        atom_c2 = c2.atoms_dfct
        c1.atoms_dfct += atom_c2
        c1.center = c1.atoms_dfct.get_center_of_mass()
        c1.update_extension()
        
        return c1

    def AggregateClusters(self, dic_cluster : Dict[str,Cluster] ) -> Dict[str,Cluster] : 
        """General aggregation method for point defect cluster
        
        Parameters:
        -----------

        dic_cluster : Dict[str,Cluster]
            Dictionnary of ```Cluster``` to aggregate

        Returns:
        --------

        Dict[str,Cluster]
            Aggreagated dictionnary of ```Cluster```
        """
        updated_dict_cluster : Dict[str,Cluster] = {'0':dic_cluster['0']}
        for key, cluster in dic_cluster.items() : 
            if key == '0' : 
                continue
            else :
                composite_idx_distance = [ [np.linalg.norm(cluster.center - c.center),idx] for idx, c in updated_dict_cluster.items()] 
                closest = np.argmin(composite_idx_distance[:,0]) 
                min_dist, min_idx = composite_idx_distance[closest]
                if min_dist < cluster.size + updated_dict_cluster[min_idx].size : 
                    updated_dict_cluster[min_idx] = self._aggregate_cluster(updated_dict_cluster[min_idx], cluster)
                
                else : 
                    updated_dict_cluster[key] = cluster
            
        return updated_dict_cluster 

    def AggregateAllClusters(self) -> None :
        """Automatic method to aggregate ```Cluster``` execpt dislocations !"""
        for dfct, dic_cluster_dfct in self.dfct.items():
            if len(dic_cluster_dfct) > 0 and dfct != 'dislocation' :
                dic_cluster_dfct = self.AggregateClusters(dic_cluster_dfct)
        
        return 

###############################################################
### GLOBAL ANALYSIS PART 
###############################################################
    def VacancyAnalysis(self, atoms : Atoms, mcd_threshold : float, elliptic : str = 'iso') -> None : 
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
        self.mean_atomic_volume = mean_atomic_volume

        # build the mask
        mask = ( mcd_distance > mcd_threshold*max_mcd ) & (atomic_volume > mean_atomic_volume)
        idx2do = np.where(mask)[0]

        for id_atom in idx2do :  
            atom = atoms[id_atom]
            self.update_dfct('vacancy', atom, array_property={'atomic-volume':[atomic_volume[id_atom]]}, rcut=4.0, elliptic = elliptic)

        return

    def InterstialAnalysis(self, atoms : Atoms, mcd_threshold : float, elliptic : str = 'iso') -> None : 
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
        self.mean_atomic_volume = mean_atomic_volume

        # build the mask
        mask = ( mcd_distance > mcd_threshold*max_mcd ) & (atomic_volume < mean_atomic_volume)
        idx2do = np.where(mask)[0]

        for id_atom in idx2do :  
            atom = atoms[id_atom]
            self.update_dfct('interstial', atom, array_property={'atomic-volume':[atomic_volume[id_atom]]}, rcut=4.0, elliptic = elliptic)

        return     
        
    def DislocationAnalysis(self, atoms : Atoms, mcd_threshold : float,
                            rcut_extended : float = 4.0,
                            rcut_full : float = 5.0,
                            rcut_neigh : float = 5.0,
                            reference_structure : reference_structure = None,
                            params_dislocation : Dict[str,float | np.ndarray] = {}) -> None : 
        """Brut force analysis to localised vacancies (based on mcd score and atomic volume)
        
        Parameters:
        -----------

        atoms : Atoms 
            Atoms object to analyse 

        mcd_treshold : float
            Ratio mcd/max(mcd) to consider the presence of atomic defect

        """
        
        if rcut_extended > rcut_full : 
            raise ValueError(f'First buffer region is larger than second buffer region ! ({rcut_extended} > {rcut_full})')

        atomic_volume = atoms.get_array('atomic-volume').flatten()
        mcd_distance = atoms.get_array('mcd-distance').flatten()
        
        max_mcd = np.amax(mcd_distance)
        mean_atomic_volume = np.mean(atomic_volume)
        self.mean_atomic_volume = mean_atomic_volume

        # build the mask
        mask = ( mcd_distance > mcd_threshold*max_mcd ) & (atomic_volume < mean_atomic_volume)
        idx2do = np.where(mask)[0]
        dislo_system : Atoms = atoms[idx2do]

        idx_extended = [ id for id, at in enumerate(atoms) if np.amin( [ np.linalg.norm(at.position - at_dis.position) for at_dis in dislo_system] ) < rcut_extended ]
        idx_full = [ id for id, at in enumerate(atoms) if np.amin( [ np.linalg.norm(at.position - at_dis.position) for at_dis in dislo_system] ) < rcut_full ]
        extended_system = atoms[idx_extended]
        full_system = atoms[idx_full]

        if reference_structure is None :
            reference_structure = self.StructureEstimator(atoms, rcut = 5.0, nb_bin = 200)

        dislocation_obj = DislocationObject(dislo_system,
                                            extended_system,
                                            full_system,
                                            rcut_neigh,
                                            reference_structure = reference_structure)

        if len(params_dislocation) == 0 : 
            params_dislocation = {'rcut_line' : 3.5, 
                                  'rcut_burger' : 4.5, 
                                  'rcut_cluster' : 5.0,
                                  'scale_cluster' : 1.2,
                                  'descriptor' : None, 
                                  'smoothing_line' : {'nb_averaging_window':3}}
        
        if params_dislocation['descriptor'] is not None : 
            params_dislocation['descriptor'] = atoms.get_array('milady-descriptor')[idx2do]

        dislocation_obj.BuildDislocations(params_dislocation['rcut_line'],
                                         params_dislocation['rcut_burger'],
                                         params_dislocation['rcut_cluster'],
                                         params_dislocation['scale_cluster'],
                                         params_dislocation['descriptor'],
                                         params_dislocation['smoothing_line'])

        return     
        
###########################################
#### WRITING PART 
###########################################

    def GetAllPointDefectData(self, path2write : os.PathLike[str] = './point_dfct.data') -> None : 
        """Extracting data from point defect analysis...
        
        Parameters:
        -----------

        path2write : os.PathLike[str]
            Path to write data file
        """
        with open(path2write,'w') as f_data : 
            f_data.write('Here is data analysis for point defects ... \n')
            for dfct in ['vacancy', 'interstial'] : 
                f_data.write(f' {dfct} analysis : I found {len(self.dfct[dfct])} clusters \n')
                print(f' {dfct} analysis : I found {len(self.dfct[dfct])} clusters \n')
                for key_dfct in self.dfct[dfct].keys() :
                    center = self.dfct[dfct][key_dfct].center.flatten()
                    f_data.write('{:s} cluster {:s} : nb dfct {:2.1f}, positions : {:2.3f} {:2.3f} {:2.3f} \n'.format(dfct,
                                                                                                            key_dfct,
                                                                                                            self.dfct[dfct][key_dfct].estimate_dfct_number(self.mean_atomic_volume),
                                                                                                            center[0],
                                                                                                            center[1],
                                                                                                            center[2]))
                    print('{:s} cluster {:s} : nb dfct {:2.1f}, positions : {:2.3f} {:2.3f} {:2.3f}'.format(dfct,
                                                                                                            key_dfct,
                                                                                                            self.dfct[dfct][key_dfct].estimate_dfct_number(self.mean_atomic_volume),
                                                                                                            center[0],
                                                                                                            center[1],
                                                                                                            center[2]))
                f_data.write('\n')
                print()
        return 

    def GetDislocationsData(self, path2write : os.PathLike[str] = './dislocation.data', only_average_data : bool = False) -> None : 
        """Extracting data from dislocation analysis...
        
        Parameters:
        -----------

        path2write : os.PathLike[str]
            Path to write data file

        only_average_data : bool 
            If ```only_average_data``` is set to False, all data about dislocation analysis are writen in data file
        """

        def array2str(array : np.ndarray) -> str : 
            return "".join(array)

        with open(path2write,'w') as f_data : 
            f_data.write('Here is data analysis for dislocation ... \n')
            f_data.write(f' dislocation analysis : I found {len(self.dfct['dislocation'])} dislocations \n')
            print(f' dislocation analysis : I found {len(self.dfct['dislocation'])} dislocations ')
            for key_dislo in self.dfct['dislocation'].keys() :
                center = self.dfct['dislocation'][key_dislo].center.flatten()
                lenght, segments = self.dfct['dislocation'][key_dislo].get_lenght_line()
                av_burger, burgers = self.dfct['dislocation'][key_dislo].get_burger_vector_line()
                av_caracter, caracters = self.dfct['dislocation'][key_dislo].get_caracter_line()

                print('Dislocation cluster {:s} : nb dfct {:2.1f}, positions : {:2.3f} {:2.3f} {:2.3f}, lenght line : {:3.2f}, average burger : {:2.2f}, average caracter : {:1.2f}'.format(key_dislo,
                                                                                        self.dfct['dislocation'][key_dislo].estimate_dfct_number(self.mean_atomic_volume),
                                                                                        center[0],
                                                                                        center[1],
                                                                                        center[2]),
                                                                                        lenght,
                                                                                        av_burger,
                                                                                        av_caracter) 
                f_data.write('Dislocation cluster {:s} : nb dfct {:2.1f}, positions : {:2.3f} {:2.3f} {:2.3f}, lenght line : {:3.2f}, average burger : {:2.2f}, average caracter : {:1.2f} \n'.format(key_dislo,
                                                                                    self.dfct['dislocation'][key_dislo].estimate_dfct_number(self.mean_atomic_volume),
                                                                                    center[0],
                                                                                    center[1],
                                                                                    center[2]),
                                                                                    lenght,
                                                                                    av_burger,
                                                                                    av_caracter)
                    
                if not only_average_data : 
                    #nightmare begins ...
                    str_data = ['segments | burgers | caracters \n']
                    str_data += [ f'{array2str(segments[i])} | {array2str(burgers[i])} | {caracters[i]} \n' for i in range(len(segments)) ]
                    f_data.write(''.join(str_data))
            
            f_data.write('\n')
            print()
        
        return 