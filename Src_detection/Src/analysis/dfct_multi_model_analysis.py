import numpy as np
import os
import matplotlib.pyplot as plt
import more_itertools

from ase import Atoms, Atom
from typing import Dict, List, Any, Tuple
from ..metrics import MCDModel, GMMModel, PCAModel, LogisticRegressor
from ..clusters import Cluster, ClusterDislo, DislocationObject, reference_structure, NanoCluster
from ..mld import DBManager
from ..tools import timeit, build_extended_neigh_

from ovito.io.ase import ase_to_ovito
from ovito.modifiers import VoronoiAnalysisModifier, DislocationAnalysisModifier, CoordinationAnalysisModifier
from ovito.pipeline import StaticSource, Pipeline

#######################################################
## Dfct analysis object
#######################################################
class DfctMultiAnalysisObject : 
    """TODO write doc"""
    def __init__(self, dbmodel : DBManager, extended_properties : List[str] = None, **kwargs) -> None : 
        self.dic_class : Dict[str,Dict[str,List[Atoms]]] = {}
        self.mcd_models : Dict[str,MCDModel] = {}
        self.gmm_models : Dict[str,GMMModel] = {}
        self.pca_models : Dict[str,PCAModel] = {}
        self.logistic_models : Dict[str,LogisticRegressor] = {} 

        self.dfct : Dict[str, Dict[int, Cluster | ClusterDislo]] = {'vacancy':{},
                                                                    'interstial':{},
                                                                    'dislocation':{},
                                                                    'A15':{},
                                                                    'C15':{},
                                                                    'other':{}}

        def fill_dictionnary_fast(ats : Atoms, dic : Dict[str,List[Atoms]]) :
            symbols = ats.get_chemical_symbols()
            for sym in dic.keys() : 
                mask = list(map( lambda b : b == sym, symbols))
                ats2keep = ats[mask]
                dic[sym] += [ats2keep]
            return 

        # sanity check !
        implemented_properies = {'local-energy':(1,),'atomic-volume':(1,),'coordination':(1,),'label-dfct':(1,)}
        for prop in extended_properties : 
            if prop not in implemented_properies :
                raise NotImplementedError('{} : this property is not yet implemented'.format(prop))

        for key in dbmodel.model_init_dic.keys() : 
            key_dic = key[0:6]
            if key_dic in self.dic_class.keys() : 
                #voronoi analysis
                if 'atomic-volume' or 'coordination' in extended_properties : 
                    dbmodel.model_init_dic[key]['atoms'] = self.compute_Voronoi(dbmodel.model_init_dic[key]['atoms'])

                if 'label-dfct' in extended_properties : 
                    dic_nb_dfct = kwargs['dic_nb_dfct']
                    dbmodel.model_init_dic[key]['atoms'] = self._labeling_outlier_atoms(dbmodel.model_init_dic[key]['atoms'],dic_nb_dfct)

                fill_dictionnary_fast(dbmodel.model_init_dic[key]['atoms'], self.dic_class[key_dic])

            else : 
                # voronoi analysis
                if 'atomic-volume' or 'coordination' in extended_properties : 
                    dbmodel.model_init_dic[key]['atoms'] = self.compute_Voronoi(dbmodel.model_init_dic[key]['atoms'])
                
                if 'label-dfct' in extended_properties : 
                    dic_nb_dfct = kwargs['dic_nb_dfct']
                    dbmodel.model_init_dic[key]['atoms'] = self._labeling_outlier_atoms(dbmodel.model_init_dic[key]['atoms'],dic_nb_dfct)             

                species = dbmodel.model_init_dic[key]['atoms'].symbols.species()
                dic_species : Dict[str,List[Atoms]] = {sp:[] for sp in species}

                fill_dictionnary_fast(dbmodel.model_init_dic[key]['atoms'], dic_species)
                self.dic_class[key_dic] = dic_species
        return

    def compute_Voronoi(self, atoms : Atoms) -> Atoms : 
        """Compute atomic volume and coordination based on Ovito Voronoi analysis
        
        Parameters
        ----------

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
        
        Parameters
        ----------

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

    def VoronoiDistribution(self, atoms : Atoms, species : str, nb_bin : int = 20) -> Tuple[float, float] :
        """Plot atomic volume distribution for a given system 
        
        Parameters
        ----------

        atoms : Atoms 
            Atoms system to perform the Voronoi distribution analysis

        species : str
            Species of the system

        nb_bin : int 
            Number of bin used for histogram

        Returns:
        --------

        float 
            mean Voronoi volume of the system 
        
        float 
            std of Voronoi volume of the system

        """
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
        plt.tight_layout()
        plt.savefig('atomic_vol.pdf', dpi=300)

        return np.mean(atomic_volume), np.std(atomic_volume)

    def MCDDistribution(self, atoms : Atoms, species : str, nb_bin : int = 20, threshold : float = 0.05) -> float : 
        """Perform MCD analysis for a given species
        
        Parameters
        ----------

        atoms : Atoms 
            Atoms system to analyse

        species : str 
            Species of the Atoms system

        nb_bin : int 
            Number of bin for histogram

        threshold : float 
            Percentage of mcd distance used as threshold

        Returns:
        --------

        float 
            Bound for mcd distance
        """

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
        desc_transform = self.pca_model._get_pca_model(atoms, species, n_component=2)
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
        return threshold*np.amax(list_mcd)

    def DXA_analysis(self, atoms : Atoms, lattice_type : str, param_dxa : Dict[str,Any] = {}) -> None : 
        """Perform classical DXA analysis from ```Ovito``` ...
        
        Parameters
        ----------

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

    def setting_mcd_models(self, path_pkl : os.PathLike[str]) -> None : 
        """Loading MCD models from a previous bulk analysis
        
        Parameters
        ----------

        MCD_object : MCD_analysis_object 
            Filled MCD_analysis_object from a bulk analysis

        """
        pkl_model_paths = [f'{path_pkl}/{f}' for f in os.listdir(path_pkl)]
        for pkl_model in pkl_model_paths : 
            model_mcd = MCDModel(pkl_model)
            self.mcd_models[model_mcd.name] = model_mcd
        return

    def setting_gmm_models(self, path_pkl : os.PathLike[str]) -> None : 
        """Loading GMM models from a previous bulk analysis
        
        Parameters
        ----------

        MCD_object : MCD_analysis_object 
            Filled MCD_analysis_object from a bulk analysis

        """
        pkl_model_paths = [f'{path_pkl}/{f}' for f in os.listdir(path_pkl)]
        for pkl_model in pkl_model_paths : 
            model_gmm = GMMModel(pkl_model)
            self.gmm_models[model_gmm.name] = model_gmm
        return

    def _get_all_atoms_species(self, species : str) -> List[Atoms] : 
        """Create the full list of Atoms for a given species
        
        Parameters
        ----------

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

    def one_the_fly_mcd_analysis(self, atoms : Atoms) -> Atoms :
        """Build one the fly mcd distances
        
        Parameters
        ----------

        atoms : Atoms 
            Atoms object containing a given configuration
        """

        descriptor = atoms.get_array('milady-descriptors')
        
        full_species = atoms.get_chemical_symbols()

        species = np.unique(full_species)
        full_array_mcd = np.zeros(descriptor.shape[0])
        full_array_proba = np.zeros(descriptor.shape[0])
        for m in self.mcd_models.keys() :
            for s in species : 
                mask_s = list(map( lambda b : b == s, full_species))
                mcd_s = self.mcd_models[m].models[s]['mcd'].mahalanobis(descriptor[mask_s]) 
                proba_s = self.mcd_models[m].models[s]['distribution'].score(mcd_s)

                full_array_mcd[mask_s] = mcd_s
                full_array_proba[mask_s] = proba_s

            atoms.set_array(f'mcd-distance-{m}',
                            np.array(full_array_mcd).reshape(len(full_array_mcd),),
                            dtype=float)
            atoms.set_array(f'probabilty-{m}',
                            np.array(full_array_proba).reshape(len(full_array_proba),),
                            dtype=float)    

        return atoms

    def one_the_fly_gmm_analysis(self, atoms : Atoms) -> Atoms :
        """Build one the fly gmm distances
        
        Parameters
        ----------

        atoms : Atoms 
            Atoms object containing a given configuration
        """

        descriptor = atoms.get_array('milady-descriptors')
        
        full_species = atoms.get_chemical_symbols()

        species = np.unique(full_species)
        for m in self.gmm_models.keys() :
            full_array_proba = np.zeros((self.gmm_models[m].n_components,descriptor.shape[0]))
            full_array_gmmd = np.zeros(descriptor.shape[0])
            for s in species : 
                mask_s = list(map( lambda b : b == s, full_species))
                gmmd_s = self.gmm_models[m].mahalanobis_gmm(s, descriptor[mask_s])
                proba_s = self.gmm_models[m]._predict_probability(gmmd_s, s)

                full_array_gmmd[mask_s] = gmmd_s
                full_array_proba[mask_s] = proba_s

            atoms.set_array(f'gmm-distance-{m}',
                            np.array(gmmd_s),
                            dtype=float)
            atoms.set_array(f'probability-{m}',                        
                            np.array(full_array_proba),
                            dtype=float)

        return atoms
    
    def _labeling_outlier_atoms(self, atoms : Atoms, dic_nb_dfct : Dict[str,int]) -> Atoms : 
        """Make labelisation of atoms in system depending their energies
        
        Parameters
        ----------

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
        
        label_array = [ 0 if local_energy[id_at] < dic_species_energy[at.symbol] else 1 \
                       for id_at, at in enumerate(atoms) ]

        atoms.set_array('label-dfct',
                        np.array(label_array),
                        dtype=int)

        return atoms

    def fit_logistic_regressor(self, species : str,
                                name_model : str,
                                name_regressor : str,
                                inputs_properties : List[str] = ['mcd-distance']) -> None : 
        """Adjust logistic regressor based on inputs_properties 
        
        Parameters
        ----------

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


        #self.logistic_model[species] = LogisticRegression()
        list_species_atoms = self._get_all_atoms_species(species)

        if 'mcd-distance' in inputs_properties :
            list_species_atoms = self.mcd_models[name_model]._get_mcd_distance(list_species_atoms, species)

        if 'gmm-distance' in inputs_properties : 
            list_species_atoms = self.gmm_models[name_model]._get_gmm_distance(list_species_atoms, species)

        Xdata = []
        Ytarget = []
        for atoms in list_species_atoms : 
            Ytarget.append(atoms.get_array('label-dfct')[0])
            miss_shaped_X = [ atoms.get_array(prop)[0].tolist() for prop in inputs_properties ]
            Xdata.append(list(more_itertools.collapse(miss_shaped_X)))

        Xdata = np.array(Xdata)
        Ytarget = np.array(Ytarget)
        
        self.logistic_models[name_regressor] = LogisticRegressor()
        self.logistic_models[name_regressor]._fit_logistic_model(Xdata, Ytarget, species, inputs_properties)

        print('Score for {:s} logistic regressor is : {:1.4f}'.format(species,self.logistic_model.models[species]['logistic_regressor'].score(Xdata,Ytarget)))
        return 

    def one_the_fly_logistic_analysis(self, atoms : Atoms) -> Atoms :
        """Perfrom logistic regression analysis
        
        Parameters
        ----------

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
        for m in self.logistic_models.keys() :
            dic_prop = {}
            new_prop = []
            for prop in self.logistic_models[m]._get_metadata() :
                try : 
                    atoms.get_array(prop) 
                except :  
                    if prop == 'mcd-distance' : 
                        atoms = self.one_the_fly_mcd_analysis(atoms)
                        mcd_model = [f'mcd-distance-{key}' for key in self.mcd_models.keys()]
                        new_prop += mcd_model

                    if prop == 'atomic-volume' : 
                        atoms = self.compute_Voronoi(atoms)
                        new_prop += prop

                    if prop == 'gmm-distance' : 
                        atoms = self.one_the_fly_gmm_analysis(atoms)
                        gmm_model = [f'gmm-distance-{key}' for key in self.gmm_models.keys()]
                        new_prop += gmm_model

            for prop in new_prop : 
                dic_prop[prop] = atoms.get_array(prop)


            list_logistic_score = []
            for id_at, at in enumerate(atoms) :
                miss_shaped_data = [ dic_prop[prop][id_at].tolist() for prop in dic_prop.keys() ]
                array_data = np.array([ list(more_itertools.collapse(miss_shaped_data)) ])
                list_logistic_score.append( self.logistic_models[m]._predict_logistic(at.symbol,array_data).flatten() )

            atoms.set_array(f'logistic-score-{m}',
                            np.array(list_logistic_score),
                            dtype=float)

        return atoms

###############################################################
### UPDATING POINT DEFECT CLUSTERS PART 
###############################################################
    def update_dfct(self, key_nano : str, atom : Atom, array_property : Dict[str,Any] = {}, rcut : float = 4.0, elliptic : str = 'iso') -> None :
        """Method to update defect inside dictionnary
        
        Parameters
        ----------

        key_nano : str 
            Type of nanophase to update

        atom : Atom
            Atom object to update in a ```Cluster``` object

        array_property : Dict[str,Any]
            Dictionnnary which contains additional data about atom in the cluster (atomic volume, mcd distance ...)

        rcut : float 
            Cut off raduis used as initial size for new ```Cluster```

        """
        if self.dfct[key_nano] == {} or key_nano not in self.dfct.keys() : 
            self.dfct[key_nano][0] = Cluster(atom, rcut, array_property=array_property)
        
        else : 
            key_closest = np.argmin([np.linalg.norm( atom.position - cluster.center) for _, cluster in self.dfct[key_nano].items()])
            key_closest = [key for key in self.dfct[key_nano]][key_closest]
            if self.dfct[key_nano][key_closest].get_elliptic_distance(atom) < 1.5 :
                self.dfct[key_nano][key_closest].append(atom, array_property=array_property, elliptic = elliptic)
            
            else : 
                next_index = max([int(key) for key in self.dfct[key_nano].keys()]) + 1
                self.dfct[key_nano][next_index] = Cluster(atom, rcut, array_property=array_property) 

    def update_nanophase(self, key_dfct : str, atom : Atom, array_property : Dict[str,Any] = {}, rcut : float = 4.0) -> None :
        """Method to update defect inside dictionnary
        
        Parameters
        ----------

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
            self.dfct[key_dfct][0] = NanoCluster(atom, rcut, array_property=array_property)
        
        else : 
            key_closest = np.argmin([np.linalg.norm( atom.position - cluster.center) for _, cluster in self.dfct[key_dfct].items()])
            key_closest = [key for key in self.dfct[key_dfct]][key_closest]
            if self.dfct[key_dfct][key_closest].get_elliptic_distance(atom) < 1.5 :
                self.dfct[key_dfct][key_closest].append(atom, array_property=array_property, elliptic = 'iso')
            
            else : 
                next_index = max([int(key) for key in self.dfct[key_dfct].keys()]) + 1
                self.dfct[key_dfct][next_index] = NanoCluster(atom, rcut, array_property=array_property) 


    def _aggregate_cluster(c1 : Cluster, c2 : Cluster) -> Cluster : 
        """Local method to aggregate two ```Cluster``` objects
        
        Parameters
        ----------

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
        
        Parameters
        ----------

        dic_cluster : Dict[str,Cluster]
            Dictionnary of ```Cluster``` to aggregate

        Returns:
        --------

        Dict[str,Cluster]
            Aggreagated dictionnary of ```Cluster```
        """
        updated_dict_cluster : Dict[int,Cluster] = {'0':dic_cluster['0']}
        for key, cluster in dic_cluster.items() : 
            if key == 0 : 
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
    def NanophasesAnalysis(self, atoms : Atoms,
                           kind_phases : List[str], 
                           threshold_p : np.ndarray = None) -> None : 
        """Brut force analysis to localised nanophases in ```Atoms``` system
        
        Parameters
        ----------

        atoms : Atoms 
            Atoms object to analyse 

        kind_phases : List[str]
            List of phase to test in analysis

        threshold_p : np.ndarray 
            Threshold probabilities array to classift nanophases
        """

        #sanity check 
        for phase in kind_phases : 
            if phase not in self.dfct.keys() : 
                raise NotImplementedError(f'... Looking for not implemented nanophase : {phase} ...')

        atomic_volume = atoms.get_array('atomic-volume').flatten()
        mean_atomic_volume = np.mean(atomic_volume)
        self.mean_atomic_volume = mean_atomic_volume        

        for phases in kind_phases : 
            probability = atoms.get_array(f'probability-{phases}')
            distance = atoms.get_array(f'mcd-distances-{phases}')
            mask = (atomic_volume < mean_atomic_volume) & probability > threshold_p
            idx2do = np.where(mask)[0]

            for id_atom in idx2do :  
                atom = atoms[id_atom]
                self.update_nanophase(phases, atom, array_property={'atomic-volume':[atomic_volume[id_atom]],
                                                                    f'mcd-distance-{phases}':[distance[id_atom]]}, rcut=4.0)

        return 

    def VacancyAnalysis(self, atoms : Atoms, mcd_threshold : float, elliptic : str = 'iso') -> None : 
        """Brut force analysis to localised vacancies (based on mcd score and atomic volume)
        
        Parameters
        ----------

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
        
        Parameters
        ----------

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
        
        Parameters
        ----------

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

        dislo_system, extended_system, full_system = build_extended_neigh_(atoms, idx2do, rcut_extended, rcut_full)

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

    def PointDefectAnalysisFunction(self, atoms : Atoms,
                           selection_funct : function, 
                           function_dictionnary : Dict[str,Any],
                           kind : str = 'vacancy',
                           elliptic : str = 'iso') -> None : 
        """Brut force analysis to localised point defect based on given selection function
        
        Parameters
        ----------

        atoms : Atoms 
            Atoms object to analyse 

        selection_funct : function
            Selection function for defects

        function_dictionnary : Dict[str,Any]
            Dictionnary containing selection data for defect (see examples)

        kind : str
            Type of defect

        elliptic : str 
            Type of cluster aggregation (iso => isotropic, aniso => anisotropic)

        """

        #sanity check 
        if kind not in self.dfct.keys() : 
            raise NotImplementedError(f'... Looking for not implemented defect : {kind} ...')

        dictionnary_methods = {'vacancy': lambda a, array, rcut, elliptic: self.update_dfct('vacancy',a,
                                                                                            array_property=array, 
                                                                                            rcut=rcut,
                                                                                            elliptic=elliptic),
                               'interstitial': lambda a, array, rcut, elliptic: self.update_dfct('interstitial',a,
                                                                                            array_property=array, 
                                                                                            rcut=rcut,
                                                                                            elliptic=elliptic),    
                               'C15': lambda a, array, rcut, e : self.update_nanophase('C15',a,
                                                                                    array_property=array,
                                                                                    rcut=rcut),

                               'A15': lambda a, array, rcut, e : self.update_nanophase('A15',a,
                                                                                    array_property=array,
                                                                                    rcut=rcut)}

        selected_idx = selection_funct(atoms,function_dictionnary)
        full_properties = {key:atoms.get_array(key) for key in function_dictionnary}

        for id_atom in selected_idx :  
            atom = atoms[id_atom]
            array_properties = {key:full_properties[key][id_atom,:] for key in function_dictionnary}
            dictionnary_methods[kind](atom, array_properties, 4.0, elliptic)

        return 

###########################################
#### WRITING PART 
###########################################

    def GetAllPointDefectData(self, path2write : os.PathLike[str] = './point_dfct.data') -> None : 
        """Extracting data from point defect analysis...
        
        Parameters
        ----------

        path2write : os.PathLike[str]
            Path to write data file
        """
        with open(path2write,'w') as f_data : 
            f_data.write('Here is data analysis for point defects ... \n')
            for dfct in ['vacancy', 'interstial','C15','A15'] : 
                if len(self.dfct[dfct]) == 0 :
                    continue

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
        
        Parameters
        ----------

        path2write : os.PathLike[str]
            Path to write data file

        only_average_data : bool 
            If ```only_average_data``` is set to False, all data about dislocation analysis are writen in data file
        """

        def array2str(array : np.ndarray) -> str : 
            return "".join(array)

        with open(path2write,'w') as f_data : 
            f_data.write('Here is data analysis for dislocation ... \n')
            f_data.write(f' dislocation analysis : I found {len(self.dfct["dislocation"])} dislocations \n')
            print(f' dislocation analysis : I found {len(self.dfct["dislocation"])} dislocations ')
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