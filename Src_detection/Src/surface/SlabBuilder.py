import numpy as np
from .SurfaceRemesh import SurfaceRemesher

# ase time + pymatgen
from ase import Atoms, Atom
from ase import io
import ase.io

try:
    from .local_pymatgen.io.ase import AseAtomsAdaptor
    from .local_pymatgen.core.surface import SlabGenerator
except ImportError:
    raise ImportError('No pymatgen :( I can not continue')

from typing import List, Tuple, TypedDict

dic_radius = {'Ac': 1.95, 'Ag': 1.6, 'Al': 1.25, 'Am': 1.75, 'Ar': 0.71, 'As': 1.15, 'At': 'no data', 'Au': 1.35, 'B': 0.85, 'Ba': 2.15, 'Be': 1.05, 'Bi': 1.6, 'Bk': 'no data', 'Br': 1.15, 'C': 0.7, 'Ca': 1.8, 'Cd': 1.55, 'Ce': 1.85, 'Cf': 'no data', 'Cl': 1.0, 'Cm': 'no data', 'Co': 1.35, 'Cr': 1.4, 'Cs': 2.6, 'Cu': 1.35, 'Dy': 1.75, 'Er': 1.75, 'Es': 'no data', 'Eu': 1.85, 'F': 0.5, 'Fe': 1.4, 'Fm': 'no data', 'Fr': 'no data', 'Ga': 1.3, 'Gd': 1.8, 'Ge': 1.25, 'H': 0.25, 'He': 'no data', 'Hf': 1.55, 'Hg': 1.5, 'Ho': 1.75, 'I': 1.4, 'In': 1.55, 'Ir': 1.35, 'K': 2.2, 'Kr': 'no data', 'La': 1.95, 'Li': 1.45, 'Lr': 'no data', 'Lu': 1.75,
              'Md': 'no data', 'Mg': 1.5, 'Mn': 1.4, 'Mo': 1.45, 'N': 0.65, 'Na': 1.8, 'Nb': 1.45, 'Nd': 1.85, 'Ne': 'no data', 'Ni': 1.35, 'No': 'no data', 'Np': 1.75, 'O': 0.6, 'Os': 1.3, 'P': 1.0, 'Pa': 1.8, 'Pb': 1.8, 'Pd': 1.4, 'Pm': 1.85, 'Po': 1.9, 'Pr': 1.85, 'Pt': 1.35, 'Pu': 1.75, 'Ra': 2.15, 'Rb': 2.35, 'Re': 1.35, 'Rh': 1.35, 'Rn': 'no data', 'Ru': 1.3, 'S': 1.0, 'Sb': 1.45, 'Sc': 1.6, 'Se': 1.15, 'Si': 1.1, 'Sm': 1.85, 'Sn': 1.45, 'Sr': 2.0, 'Ta': 1.45, 'Tb': 1.75, 'Tc': 1.35, 'Te': 1.4, 'Th': 1.8, 'Ti': 1.4, 'Tl': 1.9, 'Tm': 1.75, 'U': 1.75, 'V': 1.35, 'W': 1.35, 'Xe': 'no data', 'Y': 1.8, 'Yb': 1.75, 'Zn': 1.35, 'Zr': 1.55}


def ComputeSurfaceAtomTermination(list_atom_surface: List[Atom]) -> float:
    """Compute the atomic surface on a termination based on experimental atomic 
    radius

    Parameters
    ----------

    List of atom on the surface 

    Returns 
    -------

    float 
        Atomic surface estimation based on hard spheres

    """
    surface = 0.0
    for at_surface in list_atom_surface:
        surface += np.pi*(dic_radius[at_surface.symbol])**2

    return surface


def CheckSurfaceCompacity(slab: Atoms, list_atom_surface: List[Atom], tolerance_compacity: float, compacity_ref: float) -> bool:
    """Compare the atomic surface compacity with a criterion on maximum possible atomic compacity
    key parameter is tolerance_compacity

    Parameters 
    ----------

    slab : Atoms 
        ASE object containing the slab system

    list_atom_surface : List[Atom] 
        List of Atom system which have been identified to be at the surface termination

    tolerance_compacity : float 
        percentage of the theoritical compacity needed to have a dense termination

    compacity_ref : float 
        Reference 3D compacity

    Returns 
    -------

    bool 
        True if the termination is considered to be dense and False otherwise

    """
    surface_total = np.linalg.det(slab.cell[:2, :2])
    surface_atom = ComputeSurfaceAtomTermination(list_atom_surface)
    ratio_surface = surface_atom/surface_total
    if ratio_surface > tolerance_compacity*compacity_ref**(2.0/3.0):
        return True
    else:
        return False

class DictSym(TypedDict): 
    bool : bool
    replica : int 
    rcut : float

class DictRemesh(TypedDict):
    bool: bool
    rcut: float
    beta: float


class DictDense(TypedDict):
    bool: bool
    tolerance: float
    compacity: float


def BuildDenseHighTermination(composition: List[str], aseslab: Atoms, tolerance_surface: float, check_sym: DictSym, alpha_list: List[float], tolerance_compacity: float, compacity_ref: float, remesher: DictRemesh) -> Tuple[Atoms, bool]:
    """Build the new high dense termination for a give slab system

    Parameters
    ----------

    composition : List[str]
        List of elements of the system

    aseslab : Atoms
        ASE object containing the slab system

    tolerance_surface : float
        Heigh tolorence to consider an atom at the surface

    check_sym : Dict[bool,List[float]]
        Dictionnary to build replication in order to take into account periodic boundary condition
        List contains number of replica and rcut for descriptor in AA

    alpha_list : List[float]
        List of alpha power for radial descriptor of termination

    tolerance_compacity : float
        Theoritical 3D compacity 

    remesher : DictRemesh 
        Dictionnary containing data for remeshing on rough surfaces

    Returns 
    -------

    Atoms 
        ASE object containing the slab system with the new termination

    bool 
        Boolean which ckeck if the procedure built a dense termination

    """
    dense_slab = False
    bool_convergence = True
    while not dense_slab:
        _, _, atom_haut = BuildCompositionDistanceVectorSlab(
            composition, aseslab, terminaison='haut', tolerance=tolerance_surface, sym_check=check_sym, alpha_list=alpha_list, remesher=remesher)
        local_dense_bool = CheckSurfaceCompacity(
            aseslab, atom_haut, tolerance_compacity, compacity_ref)
        if local_dense_bool:
            dense_slab = True
        else:
            aseslab = AtomDeleter(aseslab, atom_haut, copy=False)

    return aseslab, bool_convergence


def CompositionFromBulk(file: str) -> List[str]:
    """Reading bulk file to find elements in the bulk

    Parameters
    ----------

    file : str 
        Path to the bulk file to read

    Returns 
    -------

    List[str]
        List of elements in bulk system

    """
    composition = []
    bulk = io.read(file)
    for el in bulk.get_chemical_symbols():
        if el not in composition:
            composition.append(el)

    return composition


def BuildCompositionDistanceVectorSlab(composition: List[str], slab: Atoms, terminaison='haut', tolerance=1e-2, sym_check: DictSym = {'bool':False,'replica':1, 'rcut':3.0}, alpha_list: List[float] = None, remesher: DictRemesh = {'bool': False, 'rcut': 0.0, 'beta': 1.0}) -> Tuple[List[float], List[float], List[Atom]]:
    """Build 2 vectors for one of the terminaison of the slab (you can choose the terminaison by
    changing terminaison key word : haut and bas)
    - list_compo ==> is a list that gives the composition of each element (N_el(terminaison)/N_tot(terminaison))
    for the terminaison
    - list_distance ==> is a list that gives sum of interatomic distances in the terminaison 
     (sum_i sum_j d_ij)/N_tot(terminaison)
    - list_atom_surface ==> is the list of Atom objects corresponding to the atoms in the terminaison

    Parameters
    ----------

    composition : List[str]
        List of element in the bulk system

    slab : Atoms
        ASE object containing the slab system

    terminaison : str
        Type of terminaison 'haut' or 'bas'

    tolerance : float
        High tolerance for an atom to be considered in the surface

    sym_check : Dict[bool,List[float]]
        Dictionnary to build replication in order to take into account periodic boundary condition
        List contains number of replica and rcut for descriptor in AA

    alpha_list : List[float]
        List of alpha power for radial descriptor of termination

    remesher : DictRemesh 
        Dictionnary containing data for remeshing on rough surfaces

    Returns
    -------

    List[float]
        List that gives the composition of each element (N_el(terminaison)/N_tot(terminaison))
    for the terminaison

    List[float]
        List that gives sum of interatomic distances in the terminaison 
     (sum_i sum_j d_ij)/N_tot(terminaison)

    List[Atom]
        List of Atom objects corresponding to the atoms in the terminaison        

    """
    extremum_z_position = None
    if terminaison == 'haut':
        extremum_z_position = np.amax(slab.get_positions()[:, 2])
    if terminaison == 'bas':
        extremum_z_position = np.amin(slab.get_positions()[:, 2])

    list_compo: List[float] = [0.0 for k in range(len(composition))]
    list_position: List[float] = []
    list_atom_surface: List[Atom] = []

    sum_compo = 0.0
    if alpha_list is None:
        sum_distance = 0.0
    else:
        sum_distance = [0.0 for k in range(len(alpha_list)+1)]

    for atom in slab:
        if abs(atom.position[2] - extremum_z_position) < tolerance:
            el_atom = atom.symbol
            sum_compo += 1
            list_compo[composition.index(el_atom)] += 1
            list_position.append(atom.position[:2])
            list_atom_surface.append(atom)

    if remesher['bool']:
        local_remesh = SurfaceRemesher(list_atom_surface, remesher['rcut'])
        card_remesh, list_atom_surface_remesh = local_remesh.remesh_surface(
            list_atom_surface, terminaison=terminaison)
        if card_remesh > 0:
            list_atom_surface = list_atom_surface_remesh

    """build the replicated position to take into account translation symmetries"""

    if sym_check['bool']:
        cell = slab.cell[:2, :2]
        list_position_replicated = []
        repli = sym_check['replica']
        rcut = sym_check['rcut']
        for proj_pos in list_position:
            for e1 in range(-int((repli-1)/2), int((repli-1)/2)+1):
                for e2 in range(-int((repli-1)/2), int((repli-1)/2)+1):
                    list_position_replicated.append(
                        proj_pos + (np.asarray([e1, e2]))@cell)

        list_compo = [el/sum_compo for el in list_compo]
        compt = 0
        for pos_i in list_position:
            for pos_j in list_position_replicated:
                if np.linalg.norm(pos_i-pos_j) < rcut:
                    if alpha_list is None:
                        sum_distance += np.linalg.norm(pos_i-pos_j)
                    else:
                        sum_distance[0] += np.linalg.norm(pos_i-pos_j)
                        for id_alpha, alpha in enumerate(alpha_list):
                            sum_distance[id_alpha +
                                         1] += np.linalg.norm(pos_i-pos_j)**alpha
                    compt += 1

        if alpha_list is None:
            sum_distance *= 1.0/compt
            sum_distance: List[float] = [sum_distance]
        else:
            sum_distance: List[float] = [
                el/float(compt) for el in sum_distance]

    else:
        list_compo = [el/sum_compo for el in list_compo]
        for pos_i in list_position:
            for pos_j in list_position:
                if np.linalg.norm(pos_i-pos_j) < rcut:
                    if alpha_list is None:
                        sum_distance += np.linalg.norm(pos_i-pos_j)
                    else:
                        sum_distance[0] += np.linalg.norm(pos_i-pos_j)
                        for id_alpha, alpha in enumerate(alpha_list):
                            sum_distance[id_alpha +
                                         1] += np.linalg.norm(pos_i-pos_j)**alpha

        if alpha_list is None:
            sum_distance *= 1.0/(len(list_position)**2)
            sum_distance: List[float] = [sum_distance]
        else:
            sum_distance: List[float] = [
                el/float(len(list_position)**2) for el in sum_distance]

    return list_compo, sum_distance, list_atom_surface


def AtomDeleter(slab: Atoms, list_index: List[Atom], copy=True) -> Atoms:
    """Delete the Atom objects in the slab corresponding to the list_index
    Option copy allows to create a copy of the slab without deleted atoms or delete directly 
    in the original slab object

    Parameters
    ----------

    slab : Atoms 
        ASE object containing the slab system

    list_index : List[Atom]
        List of Atom object to delete

    copy : bool 
        True if you want to delete atoms on copy, False if you directly modify the slab

    Returns 
    -------

    Atoms 
        Slab system where atoms have been removed

    """
    slab_to_save = None
    if copy == True:
        slab_to_save = slab.copy()
    if copy == False:
        slab_to_save = slab

    del slab_to_save[[atom.index for atom in list_index]]

    return slab_to_save


def DeepLevelTerminaison(niveau: int, composition: List[str], slab: Atoms, terminaison='haut', tolerance=1e-2, sym: DictSym  = {'bool':False, 'replica':1, 'rcut':3.0}, alpha_list=None, remesher={'bool': False, 'rcut': 0.0, 'beta': 1.0}) -> Tuple[List[List[float]], List[List[float]]]:
    """Build the compostion vector and the list of sum distance for a number of layer corresponding to niveau
    if level niveau = 2, list_composition and list_sum are computed for the two first layers of the terminaison

    Parameters
    ----------

    niveau : int
        Number of termination levels to compare

    composition : List[str]
        List containing all elements of the bulk

    slab : Atoms 
        ASE object containing slab system

    terminaison : str
        Type of termination to build 'haut' or 'bas'

    tolerance : float
        High tolerance for an atom to be considered in the surface

    sym_check : Dict[bool,List[float]]
        Dictionnary to build replication in order to take into account periodic boundary condition
        List contains number of replica and rcut for descriptor in AA

    alpha_list : List[float]
        List of alpha power for radial descriptor of termination

    remesher : DictRemesh 
        Dictionnary containing data for remeshing on rough surfaces 

    Returns 
    -------

    List[List[float]]
        List of list of compostion vector of each level of termination

    List[List[float]]
        List of list of radial descritpor vector of each level of termination


    """
    list_composition: List[List[float]] = []
    list_sum: List[List[float]] = []
    slab_to_work = slab.copy()
    for k in range(niveau):
        list_compo_k, sum_dis_k, index_atom = BuildCompositionDistanceVectorSlab(
            composition, slab_to_work, terminaison, tolerance, sym_check=sym, alpha_list=alpha_list, remesher=remesher)
        list_composition.append(list_compo_k)
        list_sum += sum_dis_k

        slab_to_work = AtomDeleter(slab_to_work, index_atom, copy=False)

    return list_composition, list_sum


def BuilderSurfaceOriented(path_bulk_file: str, 
                           orientation: List[float], 
                           h_slab: float, 
                           vaccum: float, 
                           composition: List[str], 
                           niveau: int, 
                           z_slab_ang: float, 
                           tolerance_z: float, 
                           path_writing: str, 
                           tolerance_surface: float = 1e-2, 
                           check_sym : DictSym = {'bool':False, 'replica':1, 'rcut':3.0}, 
                           alpha_list: List[float] = None, 
                           tolerance_descripteur: float = 1e-2, 
                           dense_check: DictDense = {'bool': False, 'tolerance': None, 'compacity': None}, 
                           remesher: DictRemesh = {'bool': False, 'rcut': 0.0, 'beta': 1.0},
                           tolerance_pymatgen : float = 1e-12) -> None:
    """Build inequivalent slabs from a given bulk file for a given set of Miller indexes (orientation)

    Parameters 
    ----------

    path_bulk_file : str 
        Path to the bulk file 

    orientation : List[float]
        HKL indexes for slab orientation

    h_slab : float
        Slab heigh generated by pymatgen

    vaccum : float
        Size of the vaccum generated by pymatgen

    composition : List[str]
        List of elements in the bulk system

    niveau : int
        Number of termination levels to compare inequivalent terminations

    z_slab_ang : float 
        Constraint on slab heigh after symmetrisation procedure

    tolerance_z : float
        Heigh tolerance for an atom to be considered in the surface

    check_sym : Dict[bool, List[float]]
        Dictionnary to build replication in order to take into account periodic boundary condition
        List contains number of replica and rcut for descriptor in AA

    alpha_list : List[float]
        List of alpha power for radial descriptor of termination

    tolerance_descriptor : float
        Tolerance to consider that two surface descriptors are different

    dense_check: DictDense
        Dictionnnary containing data for dense termination constraints

    remesher : DictRemesh 
        Dictionnary containing data for remeshing on rough surfaces

    """
    bulk = io.read(path_bulk_file)
    bulk_pymatgen = AseAtomsAdaptor.get_structure(bulk)
    gen = SlabGenerator(bulk_pymatgen, orientation, h_slab, vaccum, lll_reduce=False,
                        center_slab=True, in_unit_planes=True, max_normal_search=10)
    print('Slab is generated')

    """here tol have to be really small to ensure the convergence of the method"""
    list_slabs: List[Atoms] = gen.get_slabs(
        ftol=1e-3, tol=tolerance_pymatgen)  # keep these values !
    print('==> I found %s slabs' % (str(len(list_slabs))))

    slab_to_keep: List[Atoms] = []
    global_compo: List[List[List[float]]] = []
    global_distance: List[List[float]] = []

    for j, slab in enumerate(list_slabs):
        symetric_slab = False
        aseslab = AseAtomsAdaptor.get_atoms(slab)
        aseslab.wrap(pbc=[1, 1, 1])

        if dense_check['bool']:
            aseslab, bool_density = BuildDenseHighTermination(
                composition, aseslab, tolerance_surface, check_sym, alpha_list, dense_check['tolerance'], dense_check['compacity'], remesher=remesher)
            if not bool_density:
                print(
                    'The something wrong with the surface density of the slab... (index %s)' % (str(j)))
                continue

        _, _, _ = BuildCompositionDistanceVectorSlab(
            composition, aseslab, terminaison='haut', tolerance=tolerance_surface, sym_check=check_sym, alpha_list=alpha_list, remesher=remesher)
        high_compo_haut, high_distance_haut = DeepLevelTerminaison(
            niveau, composition, aseslab, terminaison='haut', tolerance=tolerance_surface, sym=check_sym, alpha_list=alpha_list)

        bool_size = True

        while not symetric_slab:
            try:
                _, _, atom_bas = BuildCompositionDistanceVectorSlab(
                    composition, aseslab, terminaison='bas', tolerance=tolerance_surface, sym_check=check_sym, alpha_list=alpha_list, remesher=remesher)
                high_compo_bas, high_distance_bas = DeepLevelTerminaison(
                    niveau, composition, aseslab, terminaison='bas', tolerance=tolerance_surface, sym=check_sym, alpha_list=alpha_list)

            except:
                bool_size = False
                break

            size_slab = np.amax(aseslab.get_positions()[
                                :, 2]) - np.amin(aseslab.get_positions()[:, 2])

            norm_distance = np.sum([abs(high_distance_bas[k]-high_distance_haut[k])
                                   for k in range(len(high_distance_haut))])/len(high_distance_haut)

            if high_compo_bas == high_compo_haut and norm_distance < 1e-2 and abs(size_slab - z_slab_ang) < tolerance_z:
                symetric_slab = True
            else:
                aseslab = AtomDeleter(aseslab, atom_bas, copy=False)

        if not bool_size:
            print(
                'The something wrong with the size of the slab... (index %s)' % (str(j)))
            continue

        aseslab.center(vacuum=vaccum, axis=2)
        slab_to_keep.append(aseslab)

        decimale = int(-np.log10(tolerance_descripteur))
        if high_compo_haut not in global_compo:
            global_compo.append(high_compo_haut)
            slab_to_keep.append(aseslab)
            global_distance.append([round(dis, decimale)
                                   for dis in high_distance_haut])

        else:
            round_high_distance_haut = [
                round(dis, decimale) for dis in high_distance_haut]
            if round_high_distance_haut not in global_distance:
                global_distance.append(round_high_distance_haut)
                slab_to_keep.append(aseslab)

    print('==> Reducting to %s non equivalent slabs !' %
          (str(len(slab_to_keep))))
    name_orientation = ''
    for value in orientation:
        name_orientation += '%s' % (str(value))

    for id_slab, slab in enumerate(slab_to_keep):
        ase.io.write('%s/%s_model%s.POSCAR' % (path_writing, name_orientation,
                     str(id_slab)), slab, sort=True, vasp5=True, direct=True, format='vasp')

    return
