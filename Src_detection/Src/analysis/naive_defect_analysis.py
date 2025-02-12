import numpy as np
import more_itertools

from ase import Atoms 
from ..tools.neighbour import build_extended_neigh_
from typing import List, Dict, Tuple


from ..clusters import DislocationObject, reference_structure
from ..metrics import LogisticRegressor

from .dfct_analysis import DfctMultiAnalysisObject

###############################################################
### GLOBAL ANALYSIS PART 
###############################################################
def fit_logistic_regressor(dfct_obj : DfctMultiAnalysisObject,
                           species : str,
                           name_model : str,
                           inputs_properties : List[str] = ['mcd-distance']) -> LogisticRegressor : 
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
    list_species_atoms = dfct_obj._get_all_atoms_species(species)

    if 'mcd-distance' in inputs_properties :
        list_species_atoms = dfct_obj.metamodel._get_statistical_distances(list_species_atoms, 
                                                                           name_model,
                                                                           'MCD',
                                                                           species)

    if 'gmm-distance' in inputs_properties : 
        list_species_atoms = dfct_obj.metamodel._get_statistical_distances(list_species_atoms, 
                                                                           name_model,
                                                                           'GMM',
                                                                           species)

    Xdata = []
    Ytarget = []
    for atoms in list_species_atoms : 
        Ytarget.append(atoms.get_array('label-dfct')[0])
        miss_shaped_X = [ atoms.get_array(prop)[0].tolist() for prop in inputs_properties ]
        Xdata.append(list(more_itertools.collapse(miss_shaped_X)))

    Xdata = np.array(Xdata)
    Ytarget = np.array(Ytarget)
        
    logistic = LogisticRegressor()
    logistic._fit_logistic_model(Xdata, Ytarget, species, inputs_properties)

    print('Score for {:s} logistic regressor is : {:1.4f}'.format(species,logistic.models[species]['logistic_regressor'].score(Xdata,Ytarget)))
    return logistic

def one_the_fly_logistic_analysis(logistic_models : Dict[str, LogisticRegressor],
                                  atoms : Atoms) -> Atoms :
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
    for m in logistic_models.keys() :
        dic_prop = {}
        for prop in logistic_models[m]._get_metadata() :
            atoms.get_array(prop) 


            list_logistic_score = []
            for id_at, at in enumerate(atoms) :
                miss_shaped_data = [ dic_prop[prop][id_at].tolist() for prop in dic_prop.keys() ]
                array_data = np.array([ list(more_itertools.collapse(miss_shaped_data)) ])
                list_logistic_score.append( logistic_models[m]._predict_logistic(at.symbol,array_data).flatten() )

            atoms.set_array(f'logistic-score-{m}',
                            np.array(list_logistic_score),
                            dtype=float)

        return atoms

def NanophasesAnalysis(defect_obj : DfctMultiAnalysisObject,
                       atoms : Atoms,
                       kind_phases : List[str], 
                       threshold_p : np.ndarray = None) -> DfctMultiAnalysisObject : 
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
        if phase not in defect_obj.dfct.keys() : 
            raise NotImplementedError(f'... Looking for not implemented nanophase : {phase} ...')

    atomic_volume = atoms.get_array('atomic-volume').flatten()
    mean_atomic_volume = np.mean(atomic_volume)
    defect_obj.mean_atomic_volume = mean_atomic_volume        

    for phases in kind_phases : 
        probability = atoms.get_array(f'probability-{phases}')
        distance = atoms.get_array(f'mcd-distances-{phases}')
        mask = (atomic_volume < mean_atomic_volume) & probability > threshold_p
        idx2do = np.where(mask)[0]
    
        for id_atom in idx2do :  
            atom = atoms[id_atom]
            defect_obj.update_nanophase(phases, atom, array_property={'atomic-volume':[atomic_volume[id_atom]],
                                                                f'mcd-distance-{phases}':[distance[id_atom]]}, rcut=4.0)

    return defect_obj

def VacancyAnalysis(defect_obj : DfctMultiAnalysisObject, 
                    atoms : Atoms, 
                    mcd_threshold : float, 
                    elliptic : str = 'iso') -> DfctMultiAnalysisObject : 
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
    defect_obj.mean_atomic_volume = mean_atomic_volume
    
    # build the mask
    mask = ( mcd_distance > mcd_threshold*max_mcd ) & (atomic_volume > mean_atomic_volume)
    idx2do = np.where(mask)[0]
    
    for id_atom in idx2do :  
        atom = atoms[id_atom]
        defect_obj.update_dfct('vacancy', atom, array_property={'atomic-volume':[atomic_volume[id_atom]]}, rcut=4.0, elliptic = elliptic)
    
    return defect_obj

def InterstialAnalysis(defect_obj : DfctMultiAnalysisObject, 
                       atoms : Atoms, 
                       mcd_threshold : float, 
                       elliptic : str = 'iso') -> DfctMultiAnalysisObject : 
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
    defect_obj.mean_atomic_volume = mean_atomic_volume
    # build the mask
    mask = ( mcd_distance > mcd_threshold*max_mcd ) & (atomic_volume < mean_atomic_volume)
    idx2do = np.where(mask)[0]
    
    for id_atom in idx2do :  
        atom = atoms[id_atom]
        defect_obj.update_dfct('interstial', atom, array_property={'atomic-volume':[atomic_volume[id_atom]]}, rcut=4.0, elliptic = elliptic)
    
    return defect_obj
        
def DislocationAnalysis(defect_obj : DfctMultiAnalysisObject,
                        atoms : Atoms, 
                        mcd_threshold : float,
                        rcut_extended : float = 4.0,
                        rcut_full : float = 5.0,
                        rcut_neigh : float = 5.0,
                        reference_structure : reference_structure = None,
                        params_dislocation : Dict[str,float | np.ndarray] = {}) -> Tuple[DfctMultiAnalysisObject, DislocationObject] : 
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
    defect_obj.mean_atomic_volume = mean_atomic_volume
    
    # build the mask
    mask = ( mcd_distance > mcd_threshold*max_mcd ) & (atomic_volume < mean_atomic_volume)
    idx2do = np.where(mask)[0]
    dislo_system, extended_system, full_system = build_extended_neigh_(atoms, idx2do, rcut_extended, rcut_full)
    
    if reference_structure is None :
        reference_structure = defect_obj.StructureEstimator(atoms, rcut = 5.0, nb_bin = 200)
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
    
    return defect_obj, dislocation_obj