import os
from difflib import SequenceMatcher
from typing import List

from ..surface_energy.DataSurfaceObject import DataSurface

def ReadConvergenceFile(path_convergence_file : os.PathLike[str], 
                        object : DataSurface) -> DataSurface :
    """Read convergence logfile and fill DataSurface object for analysis
    
    Parameters 
    ----------

    path_convergence_file : str
        Path to convergence file 

    object : DataSurface
        DataSurface object to fill

    Returns 
    -------

    DataSurface 
        Filled DataSurface object
    
    """
    r = open(path_convergence_file,'r').readlines() 
    compt = 0
    for l in r : 
        compt += 1
        if compt > 1 : 
            """first is to identify bulk, element or slab """
            tmp = l.split('==>')
            path = tmp[-1].split('/')
            
            """slab case"""
            if path[-3] == 'slab' : 
                name_slab = path[-2]
        
                """find energy"""
                tmp_energy = l.split('>=>')[1]
                energy_slab = float(tmp_energy.split()[0])
                
                """find composition"""
                composition = [[],[]]
                tmp_element = l.split('<=>')[1].split(',')
                for k in range(len(tmp_element)) :
                    split_tmp_element = tmp_element[k].split() 
                    composition[0].append(split_tmp_element[0])
                    composition[1].append(split_tmp_element[1])

                object.add_entry_slab(name_slab,energy_slab,composition)
            
            """bulk case"""
            if path[-3] == 'bulk' : 
                name_bulk = path[-2]
        
                """find energy"""
                tmp_energy = l.split('>=>')[1]
                energy_bulk = float(tmp_energy.split()[0])
                
                """find the composition"""
                composition = [[],[]]
                tmp_element = l.split('<=>')[1].split(',')
                for k in range(len(tmp_element)) :
                    split_tmp_element = tmp_element[k].split() 
                    composition[0].append(split_tmp_element[0])
                    composition[1].append(split_tmp_element[1])

                object.add_entry_bulk(name_bulk,energy_bulk,composition)           

            """simple elements case"""
            if path[-4] == 'elements' : 
                name_element = path[-3]
                type = path[-2]

                """find energy"""
                tmp_energy = l.split('>=>')[1]
                nb_atom_energy = float(tmp_energy.split()[0])

                """find the composition"""
                tmp_element = l.split('<=>')[1].split(',')
                nb_atom = float(tmp_element[0].split()[1])

                object.add_entry_el(name_element,type,nb_atom_energy/nb_atom)                         

    return object


def NearestMode(list_mode : List[str], 
                mode : str, 
                nb_poss : int) -> List[str]:
    """Find the nearest mode to the one in list_mode based on str metric
    
    Parameters
    ----------

    list_mode : List[str]
        List of possible modes

    mode : str
        Str to compute the nearest mode 

    nb_poss : int
        Number of possible mode proposed

    Returns 
    -------

    List[str]
        List of possible modes

    """
    list_score = []
    for mode_implemented_i in list_mode : 
        score_i = 0.0
        compt = 0
        if len(mode.split('_')) > 1 : 
            words_mode = mode.split('_')
            words = mode_implemented_i.split('_')
            for word_mode in words_mode : 
                for word in words :
                    compt += 1 
                    score_i += SequenceMatcher(None,word,word_mode).ratio()

        else : 
            words = mode_implemented_i.split('_')
            for word in words : 
                compt += 1
                score_i += SequenceMatcher(None,word,mode).ratio()
        
        score_i *= 1.0/compt
        list_score.append(score_i)
    
    """sorting"""
    for id_1,score_1 in enumerate(list_score) :
        for id_2, score_2 in enumerate(list_score) :
            if score_1 > score_2 : 
                list_score[id_1], list_score[id_2] = list_score[id_2], list_score[id_1]
                list_mode[id_1], list_mode[id_2] = list_mode[id_2], list_mode[id_1]

    return list_mode[:nb_poss]


def print_usage():
    usage = """
    This code allows to : (i) generate slabs with given orientation from a bulk file, (ii) launch 
    energy calculation to compute parametric gamma surface energy and (iii) look at the stability
    of several slabs  :
        python3 main.py path_input_file mode

    ............................ Slab builder inputs .........................................
        - path_bulk : path to the bulk file (POSCAR format) which is used to create slabs
        - vac_and_height : size of the vaccum and the slab heigh in Angstroms (the code use a 
        function from pymatgen library to do so)
        - list_miller_indexes : list of Miller indexes which defined the orientations of slabs
        - path_writing_slab : directory where the slab files will be written (if the directory 
        does not exist, it will be created)
        - niveau : number of layers compared in the slab for symmetrisation and non equivalence
        procedure
        - ang_constraint : 2 floats have to be given : (i) the heigh of the slab needed and (ii)
        the tolerance of the heigh, both are in Angstroms

    ............................... VASP .................................................
        - path_comput : path of the comutation directory, if it does not it will be created
        - path_inputs_vasp : path of vasp inputs
        - speed_convergence : fast or normal, choose between 2 INCARs and do gamma point calculation
        for fast argument
        - density_k : density of k-points in reciprocal space defined such as \rho = b/N_k
        if you are running with speed_convergence fast option, this density is not read

    ............................. Clusters................................................
    This code is working for Occigen, Jean Zay, Irene, Topaze cluster
        - cluster : which cluster you use : occigen, jean_zay or irene
        - nb_proc : the number of processors you want to use for one simulation
        - time_limit : wall time for each simulations (in seconds)
        - keep : boolean, option needed to work after keep_and_move option

    ............................. Slab energy launcher options .............................
        - path_cif : path for the .cif files for simple elements composing the bulk 
        and slabs 
        - path_bulk : path to the bulk file (POSCAR format) which is used to create slabs
        - slab_orientation : orientation of the normal vector of the surface for slabs (in x,y,z basis), 
        this key word build an accurate k-points grid for slabs

    ............................. Extract surface energy options ..........................
        - path_comput : path of the comutation directory, this path is needed to read the results 
        of DFT calculations
        - path_bulk : path to the bulk file (POSCAR format), only name of the bulk file has to 
        be constitent with path_comput directories
        - el_to_remove : element removed to calculate the parametric surface energy
        - path_slab_files : path to the slabs files, only names of slab files have to 
        be constitent with path_comput directories
        - path_to_hyperplane_data : path of writing for the surface energy results. An other file 
        called hp.data will be written in the same directory and can be used to plot the results

    ................................... Plot and stability ................................
        - path_to_hyperplane_data : path of writing for the surface energy results. The code will 
        read the hp.data that is written in the same directory
        - list_slab_to_plot : name of slabs to plot, stability analysis will be done only on 
        those slabs
        - discretisation : number of mu increment for each mu dimension use to perform the stability
        analysis


    ............................. Slab builder mode ..............................................
    keyword : slab_builder

    This mode will generate slabs from a given bulk structure. This option used a pymatgen function to
    generate a collection of slab for given orientations (list_miller_indexes) then the code will symmetrise
    and keep only non equivalent slabs. The symmetrisation and the non equivalence are based on two
    terminaison criteria : (i) the composition ( el_1, el_2, ..., el_N )/sum_i^N el_i for N in terminaison and
    (ii) the following descriptor D = 1.0/(sum_i^N)^2 sum_i sum_j r_{ij} where r_{ij} is the distance between the 
    j^th and the i^th atom of the terminaison.

    Slabs are written at the path_writing_slab and can be used as inputs for the other modes. 

                                        ! WARNING ! 
    In order to calculate the surface energy, a volume relaxation has to be performed.
    For slabs this relaxation is not possible and the volume used the equilibirum bulk volume. So, 
    A VOLUME RELAXED BULK STRUCUTURE HAS TO USED AS INPUT :))


    ............................. Launching calculation mode .........................................
    keyword : slab_energy_launcher

                    ! Launching mode needs all the slab configurations to be used !

    This mode launchs all VASP calculations for : (i) the pure elements (bulk and single calculations 
    are performed, for bulk calculation a volume relaxation is done), (ii) bulk alloy (a volume relaxation
    is performed) and (iii) slabs (no volume relaxation is performed)


    ............................. Just Check ..............................................
    keyword : just_check

                    ! Just Check mode needs to use the launch mode FIRST to work !

    This mode allows to check the convergence of all VASP calculations, results of 
    convergency are written in check_conv.log in the path_comput directory

    ............................. Check and relaunch .......................................
    keyword : check_and_relaunch

                 ! Check and relaunch mode needs to use the launch mode FIRST to work !

    This mode allows to check the convergence of all VASP calculations for unconverged 
    calculations


    .................................. Relaunch .............................................
    keyword : relaunch

                ! Relaunch mode needs to use the launch mode FIRST to work !

    This mode allows to relaunch all the calculations by direct reading of input file, you can change the
    k-points density and the mode (fast or normal) of the simulation


    ........................... Computation of surface energy ...............................
    keyword : extract_surface_energy

                ! Extract mode needs to use the launch mode FIRST to work !
    
    Parametric surface energy is computed based on the DFT results written in check_conv.log files.
    Surface energies are hyperplanes in the (mu_1,mu_2,..,mu_{N-1})  oplus  gamma space (here mu_{el_to_remove}
    is excluded from the mu_i). Normal of hyperplane is computed for each slabs and written in hp.data file.

    path_to_hyperplane_data is written to summerised the results on gamma surface, hp.data file is also written 
    and can be used to plot and perform stability analysis between several slabs.

    ................................. Plot and stability  ....................................

                ! Plot mode needs to use the extract mode FIRST to work !

    Plot the hypersurface generated in (mu_1,mu_2,..,mu_{N-1})  oplus  gamma space for the given subset of 
    slabs (list_slab_to_plot). Probability analysis is also performed on uniform grid in (mu_1,mu_2,..,mu_{N-1}),
    the number of discretisation steps for each mu dimension can be adjusted with discretisation key word. 
    
        ! KEEP in our mind that the stability analysis scale as mathcal{O}(discretision^{N-1}) !
    
    Analysis and plot are based on reading the hp.data file.
    """

    print(usage)
    return