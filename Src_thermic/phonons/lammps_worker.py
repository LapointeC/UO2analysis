import numpy as np
import os

from lammps import lammps
from ctypes import c_double
from ase.io import write

from ase import Atoms


class LammpsWorker : 
    def __init__(self, work_directory : str, atoms : Atoms) -> None :
        self.work_directory = work_directory
        self.system = atoms 
        self.lammps_instance : lammps = None 
        self.lammps_script = None

    def SetInputsScripts(self, kind_potential : str = 'eam/alloys', 
                         potential : str = 'pot.fs',
                         species : str = 'Fe',
                         name_lammps_file : str = 'in.lmp') -> None :
        
        self.lammps_script = """
            boundary p p p
            units metal
            atom_style atomic
            atom_modify map array sort 0 0.0
            read_data  {:s}
            pair_style {:s}
            pair_coeff * * {:s} {:s}
            run 0
            thermo 10
            run 0
        """.format(name_lammps_file,kind_potential,potential,species)
        print(self.lammps_script)

    def DumpAtomsSystem(self, name_file : str = 'in.lmp') -> None : 
        write('{:s}/{:s}'.format(self.work_directory,name_file),self.system, format='lammps-data')

    def InitLammpsInstance(self) -> None : 
        self.lammps_instance = lammps(cmdargs="-screen none".split())

    def CloseLammpsInstance(self) -> None : 
        self.lammps_instance.close()

    def ReadInputLines(self) -> None : 
        #with open(file,'r') as f_lammps : 
        #    for command in f_lammps : 
        #        self.lammps_instance.command(command)
        if self.lammps_script is None : 
            raise RuntimeError('Lammps can not run without input script !')
        self.lammps_instance.commands_string(self.lammps_script)

    def UpdateSystem(self, atoms : Atoms ) -> None :
        self.system = atoms 

    def UpdateLammpsSystem(self) -> None : 
        position = (3*len(self.system)*c_double)()
        for id_pos, pos_xyz in enumerate(self.system.positions) :
            for xi in range(len(pos_xyz)):
                position[3*id_pos+xi] = pos_xyz[xi]

        self.lammps_instance.scatter_atoms("x",1,3,position)        

    def Force_i_on_j(self, i : int, j : int, displacement : np.ndarray) -> np.ndarray : 
        self.lammps_instance.command('run 0')
        positions_in_lammps = self.lammps_instance.extract_atom('x',3)

        for xi in range(len(displacement)): #apply displacement
            positions_in_lammps[i][xi] += displacement[xi]
        self.lammps_instance.command('run 0')
        force = self.lammps_instance.extract_atom('f',3)
        force_i_on_j = force[j][0:3]
        for xi in range(len(displacement)): #back to the initial configuration
            positions_in_lammps[i][xi] += - displacement[xi]

        return force_i_on_j
    
    def GetLammpsEnergy(self) -> float :
        self.lammps_instance.command('variable e equal pe')
        self.lammps_instance.command('run 0')
        return self.lammps_instance.extract_compute("thermo_pe",0,0)