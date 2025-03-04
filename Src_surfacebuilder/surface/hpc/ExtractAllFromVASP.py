import numpy as np
import mmap

from ase.io import read 
from typing import List, Tuple

dic_el = {"H":1.008,"He":4.003,"Li":6.941,"Be":9.012,"B":10.811,"C":12.011,"N ":14.007,"O":15.999,"F":18.998,"Ne":20.180,"Na":22.990,"Mg":24.305,"Al":26.982,"Si":28.086,"P":30.974,"S":32.065,"Cl":35.453,"Ar":39.948,"K":39.098,"Ca":40.078,"Sc":44.956,"Ti":47.867,"V":50.942,"Cr":51.996,"Mn":54.938,"Fe":55.845,"Co":58.933,"Ni":58.693,"Cu":63.546,"Zn":65.390,"Ga":69.723,"Ge":72.640,"As":74.922,"Se":78.960,"Br":79.904,"Kr":83.800,"Rb":85.468,"Sr":87.620,"Y":88.906,"Zr":91.224,"Nb":92.906,"Mo":95.940,"Tc":98.000,"Ru":101.070,"Rh":102.906,"Pd":106.420,"Ag":107.868,"Cd":112.411,"In":114.818,"Sn":118.710,"Sb":121.760,"Te":127.600,"I":126.905,"Xe":131.293,"Cs":132.906,"Ba":137.327,"La":138.906,"Ce":140.116,"Pr":140.908,"Nd":144.240,"Pm":145.000,"Sm":150.360,"Eu":151.964,"Gd":157.250,"Tb":158.925,"Dy":162.500,"Ho":164.930,"Er":167.259,"Tm":168.934,"Yb":173.040,"Lu":174.967,"Hf":178.490,"Ta":180.948,"W":183.840,"Re":186.207,"Os":190.230,"Ir":192.217,"Pt":195.078,"Au":196.967,"Hg":200.590,"Tl":204.383,"Pb":207.200,"Bi":208.980,"Po":209.000,"At":210.000,"Rn":222.000,"Fr":223.000,"Ra":226.000,"Ac":227.000,"Th":232.038,"Pa":231.036,"U ":238.029,"Np":237.000,"Pu":244.000,"Am":243.000,"Cm":247.000,"Bk":247.000,"Cf":251.000,"Es":252.000,"Fm":257.000,"Md":258.000,"No":259.000,"Lr":262.000,"Rf":261.000,"Db":262.000,"Sg":266.000,"Bh":264.000,"Hs":277.000,"Mt":268.000}

def GetElementsFromVasp(VASP_file : str) -> Tuple[List[str],List[int]] :
    """Get the element list and number from VASP
    
    Parameters
    ----------

    VASP_file : str 
        Path to the VASP file (generally OUTCAR)

    Returns 
    -------

    List[str]
        List of elements
    
    List[int]
        Corresponding number of element in the system

    """
    ase_object = read(VASP_file)
    full_list_el = [s for s in ase_object.symbols]
    dic_el = {}
    for el in full_list_el : 
        if el in dic_el.keys() : 
            dic_el[el] += 1
        
        else :
            dic_el[el] = 0

    return [key for key in dic_el], [dic_el[key] for key in dic_el]


def GetMagnetisation(outVASP : str) -> np.ndarray :
    """Get magnetisation from VASP out file 
    
    Parameters 
    ----------

    outVASP : str 
        Path to the VASP output file 

    Returns 
    -------

    np.ndarray 
        Magentisation vector

    """
    ase_object = read(outVASP)
    return ase_object.get_magnetic_moment()


def CheckConvergence(outVASP : str) -> Tuple[bool,bool]:
    """Check the convergence criterion on VASP calculation 
    
    Parameters
    ----------

    outVASP : str
        PAth to VASP output file 

    Returns 
    -------

    bool 
        True if the accuracy is reached, False otherwise 

    bool 
        True if the structural minimisation criterion is reached, False otherwise

    """
    bool_acc = False
    bool_struc_min = False
    with open(outVASP,'r') as f:
        m=mmap.mmap(f.fileno(),0,prot=mmap.PROT_READ)
        if m.find(b'reached required accuracy') > -1 :
            bool_acc = True
            if m.find(b'reached required accuracy - stopping structural energy minimisation') > -1 :
                bool_struc_min = True

    return bool_acc, bool_struc_min


def GetEnergyFromVasp(outVASP : str) -> float :
    """Get energy from VASP out file 
    
    Parameters 
    ----------

    outVASP : str 
        Path to the VASP output file 

    Returns 
    -------

    float
        Energy of the system

    """  
    ase_object = read(outVASP)
    return ase_object.get_potential_energy()



def GetNatomFromVasp(outVASP : str) :
    """Get number of atom from VASP out file 
    
    Parameters 
    ----------

    outVASP : str 
        Path to the VASP output file 

    Returns 
    -------

    int
        Number of atom in the system

    """  
    ase_object = read(outVASP)
    return ase_object.positions.shape[0]


def ForcesAndPosFromVasp(outVASP : str) -> Tuple[np.ndarray, np.ndarray]:
    """Get postions and forces vectors from VASP out file 
    
    Parameters 
    ----------

    outVASP : str 
        Path to the VASP output file 

    Returns 
    -------

    np.ndarray 
        Positions array of the system

    np.ndarray 
        Force array of the system

    """  
    ase_object = read(outVASP)
    return ase_object.positions, ase_object.get_forces()


def GetStressFromVASP(outVASP : str) -> np.ndarray :
    """Get stress from VASP out file
    
    Parameters 
    ----------

    outVASP : str 
        Path to the VASP output file 

    Returns 
    -------

    np.ndarray 
        Stress matrix

    """  
    ase_object = read(outVASP)
    return ase_object.get_stress()


def GetVolumeFromVASP(outVASP : str) -> float :
    """Get volume from VASP out file
    
    Parameters 
    ----------

    outVASP : str 
        Path to the VASP output file 

    Returns 
    -------

    float
        Volume of the system

    """  
    ase_object = read(outVASP)
    return ase_object.get_volume()


def WritePoscarMilady(outVASP : str, path : str) -> None :
    """Write Milady poscar from VASP output file 
    
    Parameters
    ----------

    outVASP : str
        Path to the VASP output file 

    path : str
        Writing path for the Milady poscar
    
    """    
    list_el, nb_at = GetElementsFromVasp(outVASP)
    ase_object = read(outVASP)
    energy = ase_object.get_potential_energy()
    cell = ase_object.cell[:]
    pos = ase_object.positions
    forces = ase_object.get_forces()
    stress = ase_object.get_stress()

    w_poscar = open(path,'w')
    first_line = '111 %2d '%(len(list_el))
    for el in list_el :
        first_line += '%s %3d '%(el,int(round(dic_el[el],0)))
    first_line += '%4.9f %4.9f %4.9f \n'%(energy,energy,energy)
    w_poscar.write(first_line)
    w_poscar.write('1.0000000\n')

    # cell
    for i in range(cell.shape[0]):
        w_poscar.write('%8.9f %8.9f %8.9f \n'%(cell[i,0],cell[i,1],cell[i,2]))
    line_el = ''
    line_nb_at = ''
    for k,el in enumerate(list_el) :
        line_el += '%s '%(el)
        line_nb_at += '%3d '%(nb_at[k])
    w_poscar.write('%s\n'%(line_el))
    w_poscar.write('%s\n'%(line_nb_at))
    w_poscar.write('Cartesian \n')

    # pos
    for j in range(pos.shape[0]):
        w_poscar.write('%8.9f %8.9f %8.9f \n'%(pos[j,0],pos[j,1],pos[j,2]))
    w_poscar.write('\n')

    # forces
    for k in range(forces.shape[0]):
        w_poscar.write('%8.9f %8.9f %8.9f \n'%(forces[k,0],forces[k,1],forces[k,2]))
    w_poscar.write('\n')

    #stress
    w_poscar.write('%5.5f %5.5f %5.5f %5.5f %5.5f %5.5f \n'%(stress[0,0],stress[1,1],stress[2,2],stress[1,2],stress[0,2],stress[0,1]))
    w_poscar.write('\n')
    w_poscar.write('0')

    return