import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.covariance import MinCovDet

from ase import Atoms, Atom

from typing import Dict, List , Any, TypedDict
from library_mcd import DBManager, MCDAnalysisObject
from cluster import Cluster, ClusterDislo

from ovito.io.ase import ase_to_ovito
from ovito.modifiers import VoronoiAnalysisModifier, DislocationAnalysisModifier
from ovito.pipeline import StaticSource, Pipeline


#######################################################
## Dfct analysis object
#######################################################
class DfctAnalysisObject : 

    def __init__(self, dbmodel : DBManager, extended_properties : List[str] = None, **kwargs) -> None : 
        self.dic_class : Dict[str,Dict[str,List[Atoms]]] = {}
        self.mcd_model : Dict[str,MinCovDet] = {}
        self.logistic_model : Dict[str,LogisticRegression] = {} 
        self.meta_data_model = []
        self.dfct : Dict[str, Dict[str, Cluster]] = {'vacancy':{},'interstial':{},'dislocation':{},'other':{}}

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
        for species, models in MCD_object.mcd_model.items() : 
            self.mcd_model[species] = models 
        return

    def setting_mcd_model(self, MCD_object : MCDAnalysisObject) -> None : 
        """Loading MCD models from a previous bulk analysis
        
        Parameters:
        -----------

        MCD_object : MCD_analysis_object 
            Filled MCD_analysis_object from a bulk analysis

        """
        def local_mcd_model(species : str, model : MinCovDet) -> None :
            self.mcd_model[species] = model
        
        [ local_mcd_model(species, model) for species, model in MCD_object.mcd_model.items() ]
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
        implemented_properies = {'local-energy':(1,),'atomic-volume':(1,),'coordination':(1,),'mcd-distance':(1,)}
        for prop in inputs_properties : 
            if prop not in implemented_properies :
                raise NotImplementedError('{} : this property is not yet implemented'.format(prop))


        self.logistic_model[species] = LogisticRegression()
        list_species_atoms = self._get_all_atoms_species(species)
        list_species_atoms = self._get_mcd_distance(list_species_atoms, species)

        Xdata = [ atoms.get_array('label-dfct')[0] for atoms in list_species_atoms]
        Ytarget = [ [ atoms.get_array(prop)[0] for prop in inputs_properties ] for atoms in list_species_atoms ]
        
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

        
        list_logistic_score = [ self._predict_logistic(at.symbol, np.array([[  dic_prop[prop][id_at] for prop in dic_prop.keys() ]]) ).flatten() \
                                for id_at, at in enumerate(atoms) ]
        atoms.set_array('logistic-score',
                        np.array(list_logistic_score),
                        dtype=float)

        return atoms

###############################################################
### ANALYSIS PART 
###############################################################
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

    def _aggregate_cluster(c1 : Cluster, c2 : Cluster) -> Cluster : 
        atom_c2 = c2.atoms_dfct
        c1.atoms_dfct += atom_c2
        c1.center = c1.atoms_dfct.get_center_of_mass()
        c1.update_extension()
        return c1

    def AggregateClusters(self, dic_cluster : Dict[str,Cluster] ) -> Dict[str,Cluster] : 
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
        for dfct, dic_cluster_dfct in self.dfct.items():
            if len(dic_cluster_dfct) > 0 and dfct != 'dislocation' :
                dic_cluster_dfct = self.AggregateClusters(dic_cluster_dfct)
        
        return 
    
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

        # build the mask
        mask = ( mcd_distance > mcd_threshold*max_mcd ) & (atomic_volume > mean_atomic_volume)
        idx2do = np.where(mask)[0]

        for id_atom in idx2do :  
            atom = atoms[id_atom]
            self.update_dfct('vacancy', atom, array_property={'atomic-volume':[atomic_volume[id_atom]]}, rcut=4.0, elliptic = elliptic)

        return

        #for key in self.dfct['vacancy'].keys() :
        #    center = self.dfct['vacancy'][key].center.flatten()
        #    print('Vacancy cluster {:s} : nb vacancy {:2.1f}, positions : {:2.3f} {:2.3f} {:2.3f}'.format(key,
        #                                                                                                        self.dfct['vacancy'][key].estimate_dfct_number(mean_atomic_volume),
        #                                                                                                        center[0],
        #                                                                                                        center[1],
        #                                                                                                        center[2]))

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

        # build the mask
        mask = ( mcd_distance > mcd_threshold*max_mcd ) & (atomic_volume < mean_atomic_volume)
        idx2do = np.where(mask)[0]

        for id_atom in idx2do :  
            atom = atoms[id_atom]
            self.update_dfct('interstial', atom, array_property={'atomic-volume':[atomic_volume[id_atom]]}, rcut=4.0, elliptic = elliptic)

        return     
        
        #for key in self.dfct['vacancy'].keys() :
        #    center = self.dfct['vacancy'][key].center.flatten()
        #    print('Interstial cluster {:s} : nb vacancy {:2.1f}, positions : {:2.3f} {:2.3f} {:2.3f}'.format(key,
        #                                                                                                    self.dfct['vacancy'][key].estimate_dfct_number(mean_atomic_volume),
        #                                                                                                    center[0],
        #                                                                                                    center[1],
        #                                                                                                    center[2]))