import numpy as np
import os
import matplotlib.pyplot as plt

from ase import Atoms, Atom
from ase.io import write
from typing import Dict, List, Any, Tuple
from types import FunctionType
from ..metrics import PCAModel, LogisticRegressor, MetaModel
from ..clusters import Cluster, ClusterDislo, DislocationObject, reference_structure, NanoCluster
from ..mld import DBManager
from ..tools import timeit, build_extended_neigh_

from ovito.io.ase import ase_to_ovito
from ovito.modifiers import VoronoiAnalysisModifier, DislocationAnalysisModifier, CoordinationAnalysisModifier
from ovito.pipeline import StaticSource, Pipeline

#######################################################
## GENERAL DEFECT ANALYSIS OBJECT
#######################################################
class DfctMultiAnalysisObject : 
    """Generic analysis object to fit reference distance models
    Reference models will be stored in ```self.meta_model``` under ```MetaModel``` format
    Defect analysis will be stored in ```self.dfct``` which has the following structure:
        - ```Dict[str, Dict[int, Cluster | ClusterDislo]]```
    """
    def __init__(self, dbmodel : DBManager = None, 
                 extended_properties : List[str] = None, **kwargs) -> None : 
        self.dic_class : Dict[str,Dict[str,List[Atoms]]] = {}
        self.metamodel : MetaModel = MetaModel()
        self.pca_models : Dict[str,PCAModel] = {}
        self.logistic_models : Dict[str,LogisticRegressor] = {} 

        self.dfct : Dict[str, Dict[int, Cluster | ClusterDislo]] = {'vacancy':{},
                                                                    'interstial':{},
                                                                    'dislocation':{},
                                                                    'A15':{},
                                                                    'C15':{},
                                                                    'other':{}}

        self.mean_atomic_volume : Dict[str, float] = {}

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

        if dbmodel is not None : 
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

    def DraftData(self, 
                  db_model : DBManager, 
                  extended_properties : List[str] = None,
                  extension : str = 'cfg') -> None : 
        """Write draftly data coming from distance analysis
        
        Parameters
        ----------

        db_model : ```DBManager``` 
            ```DBManager``` object to write 
        
        extended_properties : List[str] 
            List of extended properties to consider for analysis

        extension : str
            Type of extension for geometry files
        """
        
        # seem to be useless for cfg ...
        list_whole_properties = []
        dic_equiv = {'MCD':'mcd-distance',
                     'GMM':'gmm-distance',
                     'MAHA':'mahalanobis-distance'}
        for key_m in self.metamodel.meta_data.keys() : 
            kind = dic_equiv[self.metamodel.meta_kind[key_m]]
            name = self.metamodel.meta_data[key_m]
            list_whole_properties.append(f'{kind}-{name}')
        list_whole_properties += extended_properties

        print(list_whole_properties)

        for key, val in db_model.model_init_dic.items() : 
            write(f'{key}.{extension}', 
                  val['atoms'], 
                  format=extension)

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

    def VoronoiDistribution(self, atoms : Atoms, 
                            name_distribution : str,
                            species : str, 
                            nb_bin : int = 20) -> Tuple[float, float] :
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
        
        atoms_sp = self._get_all_atoms_species_list([atoms], species)[0]
        atomic_volume = atoms_sp.get_array('atomic-volume')
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
        plt.savefig(f'atomic_vol_{name_distribution}_{species}.png', dpi=300)

        self.mean_atomic_volume[species] = np.mean(atomic_volume)

        return self.mean_atomic_volume[species], np.std(atomic_volume)

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

    def setting_metamodel(self, path_pkl : os.PathLike[str]) -> None : 
        """Loading MetaModel from a previous analysis
        
        Parameters
        ----------

        path_pkl : os.PathLike[str]
            Path to the previous MetaModel 

        """
        self.metamodel._load_pkl(path_pkl)
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

    def _get_all_atoms_species_list(self, list_ats : List[Atoms], species : str) -> List[Atoms] : 
        """Fast method to extract ```Atoms``` object with specific species from a given ```List[Atoms]``` """
        def fast_species(ats : Atoms, species : str) -> Atoms : 
            symbols = ats.get_chemical_symbols()
            mask_species = list(map( lambda b : b == species, symbols))
            return ats[mask_species]

        return [fast_species(ats, species) for ats in list_ats]
    
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

###############################################################
### UPDATING POINT DEFECT CLUSTERS PART 
###############################################################
    def update_dfct(self, key_nano : str, 
                    atom : Atom, 
                    array_property : Dict[str,Any] = {}, 
                    rcut : float = 4.0, 
                    elliptic : str = 'iso') -> None :
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

    def update_nanophase(self, key_dfct : str, 
                         atom : Atom, 
                         array_property : Dict[str,Any] = {}, 
                         rcut : float = 4.0) -> None :
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


########################
#### DEFECT ANALYSIS
########################
    def PointDefectAnalysisFunction(self, atoms : Atoms,
                           selection_funct : FunctionType, 
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
                               
                               'other': lambda a, array, rcut, elliptic: self.update_dfct('other',a,
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

    def DislocationAnalysisFunction(self, atoms : Atoms,
                            selection_function : FunctionType,
                            function_dictionnary : Dict[str,Any],
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

        
        selected_idx = selection_function(atoms,function_dictionnary)
        dislo_system, extended_system, full_system = build_extended_neigh_(atoms, selected_idx, rcut_extended, rcut_full)

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
            params_dislocation['descriptor'] = atoms.get_array('milady-descriptor')[selected_idx]

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

    def GetAllPointDefectData(self,
                              species : str,
                              path2write : os.PathLike[str] = './point_dfct.data') -> None : 
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
                                                                                                            self.dfct[dfct][key_dfct].estimate_dfct_number(self.mean_atomic_volume[species]),
                                                                                                            center[0],
                                                                                                            center[1],
                                                                                                            center[2]))
                    print('{:s} cluster {:s} : nb dfct {:2.1f}, positions : {:2.3f} {:2.3f} {:2.3f}'.format(dfct,
                                                                                                            key_dfct,
                                                                                                            self.dfct[dfct][key_dfct].estimate_dfct_number(self.mean_atomic_volume[species]),
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