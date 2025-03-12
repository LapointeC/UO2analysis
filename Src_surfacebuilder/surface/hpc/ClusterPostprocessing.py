import os, re
import numpy as np
from ..ase.ASEVasp import WriteIncarASE
from ..parser.SurfaceParser import AseVaspParser

from typing import List

def format_time(time : int) -> List[str]:
    """Change the time format from seconds to [hour, minute, seconds] for some clusters
    
    Parameters
    ----------

    time : int
        time in seconds 

    Returns 
    -------

    List[str]
        time in [hour, minute, seconds]
    """

    time_list = []
    convertion_tiem = [3600,60,1]
    for temps in convertion_tiem :
        time_list.append(str(100+time//temps)[1:])
        time = time%temps

    return time_list


def ClearVaspInputs(work_path : os.PathLike[str]) -> None :
    """Clear Vasp inputs
    
    Parameters 
    ----------

    work_path : str 
        Path to clear

    """

    os.system('rm -r %s/POTCAR'%(work_path))
    os.system('rm -r %s/INCAR'%(work_path))
    return


def ClearVaspOutputs(work_path : os.PathLike[str]) -> None:
    """Clear all Vasp outputs
    
    Parameters 
    ----------

    work_path : str 
        Path to clear   
    
    """

    os.system('rm -r %s/WAVECAR'%(work_path))
    os.system('rm -r %s/CHGCAR'%(work_path))
    os.system('rm -r %s/PCDAT'%(work_path))
    os.system('rm -r %s/IBZKPT'%(work_path))
    os.system('rm -r %s/XDATCAR'%(work_path))
    os.system('rm -r %s/OSZICAR'%(work_path))
    os.system('rm -r %s/EIGENVAL'%(work_path))
    os.system('rm -r %s/PROCAR'%(work_path))
    os.system('rm -r %s/DOSCAR'%(work_path))
    os.system('rm -r %s/vasprun.xml'%(work_path))

    return


def PrintKPointsGrid(density : float, 
                     cell_param : np.ndarray, 
                     mode : str = 'fast') -> None :
    """Print the Kpoints grid that will be used for calculations
    
    Parameters 
    ----------

    density : float 
        Kpoints density in reciprocal space => density = 1.0/(lattice parameters * N Kpoints)

    cell_param : np.ndarray
        Supercell associated to the system 

    mode : str 
        Mode used for Kpoints setting => 'fast' = gamma points, 'normal' = grid based on density
    """

    if mode == 'normal' :
        kx = 1.0/np.sqrt(np.linalg.norm(cell_param[0,:])**2)/density
        ky = 1.0/np.sqrt(np.linalg.norm(cell_param[1,:])**2)/density
        kz = 1.0/np.sqrt(np.linalg.norm(cell_param[2,:])**2)/density       
        print('K-points grid is set to %2d x %2d x %2d (without taking into account surface)'%(kx,ky,kz))

    if mode == 'fast' :
        print('K-points grid is set to Gamma point 1 x 1 x 1')

    return

def SetKpoints(density : float, 
               cell_param : np.ndarray, 
               speed : str) -> List[int]:
    """Setup Kpoints grid that will be used for calculations
    
    Parameters 
    ----------

    density : float 
        Kpoints density in reciprocal space => density = 1.0/(lattice parameters * N Kpoints)

    cell_param : np.ndarray
        Supercell associated to the system 

    speed : str 
        Mode used for Kpoints setting => 'fast' = gamma points, 'normal' = grid based on density
    
    Returns
    -------

    List[int] 
        Kpoints grid to setup
    """
    if speed == 'normal' :
        kx = 1.0/np.sqrt(np.linalg.norm(cell_param[0,:])**2)/density
        ky = 1.0/np.sqrt(np.linalg.norm(cell_param[1,:])**2)/density
        kz = 1.0/np.sqrt(np.linalg.norm(cell_param[2,:])**2)/density
        return [int(np.ceil(kx)), int(np.ceil(ky)), int(np.ceil(kz))]

    if speed == 'fast' : 
        return [1,1,1]


def SetKpointsSlab(density : float, 
                   cell_param : np.ndarray, 
                   speed : str, 
                   orientation_slab : List[int] = [0,0,1]) -> List[int]:
    """Setup Kpoints grid that will be used for slab calculations
    
    Parameters 
    ----------

    density : float 
        Kpoints density in reciprocal space => density = 1.0/(lattice parameters * N Kpoints)

    cell_param : np.ndarray
        Supercell associated to the system 

    speed : str 
        Mode used for Kpoints setting => 'fast' = gamma points, 'normal' = grid based on density
    
    orientation_slab : List[int]
        Slab orientation, non periodic direction is set by 1 and periodic is set by 0

    Returns
    -------

    List[int] 
        Kpoints grid to setup including non periodic boundary conditions following slab direction
    """  
    if speed == 'normal' :
        kx = 1.0/np.sqrt(np.linalg.norm(cell_param[0,:])**2)/density
        ky = 1.0/np.sqrt(np.linalg.norm(cell_param[1,:])**2)/density
        kz = 1.0/np.sqrt(np.linalg.norm(cell_param[2,:])**2)/density
        kpts = [int(np.ceil(kx)), int(np.ceil(ky)), int(np.ceil(kz))]

        kx_slab = (1-kpts[0])*orientation_slab[0]+kpts[0]
        ky_slab = (1-kpts[1])*orientation_slab[1]+kpts[1]
        kz_slab = (1-kpts[2])*orientation_slab[2]+kpts[2]
        return [kx_slab, ky_slab, kz_slab]

    if speed == 'fast' :
        return [1, 1, 1]


def SetRelaunch(path : os.PathLike[str]) -> None :
    """Set up the new inputs where calculations are not converged
    This function keep a keep of OUTCAR, POSCAR and OSZICAR for all iteration (file_iteration)

    Parameters
    ----------

    path : str 
        Path to relaunch
    """
    iteration = 1
    if not os.path.exists('%s/iteration.data'%(path)):
        w = open('%s/iteration.data'%(path),'w')
        w.write('0')
        w.close()

    if os.path.exists('%s/iteration.data'%(path)):
        r = open('%s/iteration.data'%(path),'r').readlines()
        iteration = int(r[0])

    iteration += 1
    os.system('mv %s/POSCAR %s/POSCAR_%s'%(path,path,str(iteration)))

    if os.stat('%s/CONTCAR'%(path)).st_size != 0 :
        os.system('cp %s/CONTCAR %s/POSCAR'%(path,path))
        os.system('cp %s/CONTCAR %s/CONTCAR_%s'%(path,path,str(iteration)))

    elif os.stat('%s/CONTCAR'%(path)).st_size == 0 :
        list_poscar = [el for el in os.listdir('%s'%(path)) if (el.startswith('CONTCAR') and os.stat('%s/%s'%(path,el)).st_size > 0)]
        index = [int(el.split('_')[-1]) for el in list_poscar]
        print(index)
        if index == [] :
            os.system('cp %s/POSCAR_1 %s/POSCAR'%(path,str(iteration),path))

        else :
            max_index = max(index)
            print('cp %s/CONTCAR_%s %s/POSCAR'%(path,str(max_index),path))
            #exit(0)
            os.system('cp %s/CONTCAR_%s %s/POSCAR'%(path,str(max_index),path))
            os.system('cp %s/CONTCAR %s/CONTCAR_%s'%(path,path,str(iteration)))

    os.system('mv %s/OUTCAR %s/OUTCAR_%s'%(path,path,str(iteration)))
    os.system('mv %s/OSZICAR %s/OSZICAR_%s'%(path,path,str(iteration)))

    os.system('rm %s/iteration.data'%(path))
    w = open('%s/iteration.data'%(path),'w')
    w.write('%s'%(str(iteration)))
    w.close()

    return

def SetupVaspASE(parser_vasp : AseVaspParser, 
                 input_path : os.PathLike[str], 
                 work_path : os.PathLike[str], 
                 density : float ,
                 cell : np.ndarray, 
                 speed : str, 
                 el : str,
                 **kwargs) -> int:
    """Setup all VASP inputs for calculations 
    
    Parameters
    ----------

    parser_vasp : AseVaspParser
        Dictionnary of parameters to setup vasp calculation
    
    inputs_path : str
        Path to POTCAR for VASP

    work_path : str 
        Path to launch new calculation

    density : float 
        Kpoints density in reciprocal space => density = 1.0/(lattice parameters * N Kpoints)

    cell_param : np.ndarray
        Supercell associated to the system 

    speed : str 
        Mode used for Kpoints setting => 'fast' = gamma points, 'normal' = grid based on density

    el : str 
        Name of the elements asssociated to the calculation (allows to find the right POTCAR)


    Returns
    -------

    int     
        Proposal for Kpoints parallelisation

    """

    os.system('ln -s %s/POTCAR_%s %s/POTCAR'%(input_path,el,work_path))
    kpoints =  SetKpoints(density,cell,work_path,speed)
    if speed == 'normal' :
        new_setup = {'kpar':np.amin(kpoints),'kpoints':kpoints}
        if 'new_setup' in kwargs.keys():
            tmp=kwargs['new_setup']
            for key in tmp.keys() :
                new_setup[key] = tmp[key]
        WriteIncarASE(work_path,parser_vasp,new_setup=new_setup)

    if speed == 'fast' :
        new_setup = {'kpar':1,'kpoints':kpoints}
        if 'new_setup' in kwargs.keys():
            tmp=kwargs['new_setup']
            for key in tmp.keys() :
                new_setup[key] = tmp[key]
        WriteIncarASE(work_path,parser_vasp,new_setup=new_setup)
        kpar = 1

    return kpar

def SetupVaspSlabASE(parser_vasp : AseVaspParser, 
                     input_path : os.PathLike[str], 
                     work_path : os.PathLike[str], 
                     density : float, 
                     cell : np.ndarray, 
                     speed : str, 
                     slab_orientation : List[int],
                     **kwargs) -> int :
    """Setup all VASP inputs for calculations 
    
    Parameters
    ----------

    parser_vasp : AseVaspParser
        Dictionnary of parameters to setup vasp calculation
    
    inputs_path : str
        Path to POTCAR for VASP

    work_path : str 
        Path to launch new calculation

    density : float 
        Kpoints density in reciprocal space => density = 1.0/(lattice parameters * N Kpoints)

    cell_param : np.ndarray
        Supercell associated to the system 

    speed : str 
        Mode used for Kpoints setting => 'fast' = gamma points, 'normal' = grid based on density

    el : str 
        Name of the elements asssociated to the calculation (allows to find the right POTCAR)

    orientation_slab : List[int]
        Slab orientation, non periodic direction is set by 1 and periodic is set by 0

    Returns
    -------

    int     
        Proposal for Kpoints parallelisation

    """

    os.system('ln -s %s/POTCAR %s/POTCAR'%(input_path,work_path))
    kpoints = SetKpointsSlab(density,cell,work_path,speed,slab_orientation)
    if speed == 'normal' :
        kpar = np.amin([el for el in kpoints if el > 1])
        new_setup = {'kpar':kpar,'kpoints':kpoints}
        if 'new_setup' in kwargs.keys():
            tmp=kwargs['new_setup']
            for key in tmp.keys() :
                new_setup[key] = tmp[key]
        WriteIncarASE(work_path,parser_vasp,new_setup=new_setup)

    if speed == 'fast' :
        new_setup = {'kpar':1,'kpoints':[1, 1, 1]}
        if 'new_setup' in kwargs.keys():
            tmp=kwargs['new_setup']
            for key in tmp.keys() :
                new_setup[key] = tmp[key]
        WriteIncarASE(work_path,parser_vasp,new_setup=new_setup)
        kpar = 1

    return kpar


def WritingSlurm(path_simu : os.PathLike[str], 
                 full_slurm : str, 
                 name : str, 
                 nb_proc : int, 
                 kpar : int) -> None:
    """Write the slurm file submission
    
    Parameters 
    ----------

    path_simu : str 
        Path to launch

    full_slurm : str
        Full string containing all slurm file

    name : str 
        Name of the job

    nb_proc : int
        Number of processor per kpar unit

    kpar : int 
        Number of parallel in reciprocal space grid
 
    """
    full_slurm = re.sub('NAME',name,full_slurm)
    full_slurm = re.sub('NPROC',str(nb_proc*kpar))
    jsub = open('%s/surface.slurm'%(path_simu),'w')
    jsub.write(full_slurm)
    jsub.close()

    return