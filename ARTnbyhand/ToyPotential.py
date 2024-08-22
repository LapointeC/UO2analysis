import numpy as np
from ase import Atoms, Atom
import math
from typing import List

class CoulombianCalc : 
    def __init__(self, system : Atoms) : 
        self.system = system
        self.epsilon0 = 5.5263e-3 #in e^2 eV \AA
        self.dict_charge = {'C':2.0,'H':1.0,'N':3.0,'Co':1.0}

    def update_system(self, system : Atoms) : 
        """Update the ASE object containing the system
        
        Parameters 
        ----------

        system : Atoms
            ASE object containing the system
        """
        self.system = system

    def build_replicate(self,vect_replicate : List[float]) -> Atoms :
        """Build the periodic replication of the system for straightfoward calculations...
        
        Parameters 
        ----------

        vect_replicate : List[float]
            Periodic replication vector

        Returns 
        -------

        Atoms 
            Replicated system contains in ASE object
        """ 
        self.replicated_system = self.system.copy()
        for at in self.system : 
            for x_repli in range(int(-(vect_replicate[0]-1)/2),int((vect_replicate[0]-1)/2)+1):
                for y_repli in range(int(-(vect_replicate[1]-1)/2),int((vect_replicate[1]-1)/2)+1):
                    for z_repli in range(int(-(vect_replicate[2]-1)/2),int((vect_replicate[2]-1)/2)+1):
                        if not((x_repli ==0) and (y_repli ==0) and (z_repli ==0)) :
                            new_atom : Atom = at.copy()
                            new_atom.position = at.position + np.array([x_repli, y_repli, z_repli])@self.system.cell[:,:]
                            self.replicated_system.append(new_atom)
        
        return self.replicated_system

    def compute_electro(self,pos_i : np.ndarray,pos_j : np.ndarray) -> float :
        """Compute the A.U electrostatic energy between two atomic coordinates
        
        Parameters
        ----------

        pos_i : np.ndarray
            atomic coordinate i 
        
        pos_j : np.ndarray 
            atomic coordinate j 


        Returns 
        -------

        float 
            A.U coulombian energy
        """
        denum = np.linalg.norm(pos_i-pos_j)
        if denum < 1e-3 :
            return 0.0
        else :
            return 1.0/(denum*4*np.pi*self.epsilon0)

    def compute_electro_forces_j_sur_i(self,pos_i : np.ndarray, pos_j : np.ndarray) -> np.ndarray :
        """Compute the A.U electrostatic forces between two atomic coordinates
        
        Parameters
        ----------

        pos_i : np.ndarray
            atomic coordinate i 
        
        pos_j : np.ndarray 
            atomic coordinate j 


        Returns 
        -------

        np.ndarray
            A.U coulombian forces
        """
        denum = np.linalg.norm(pos_j-pos_i)**3
        if denum < 1e-3 :
            return np.array([0.0,0.0,0.0])
        else :
            return -4*np.pi*self.epsilon0*(pos_j-pos_i)/denum

    def compute_coulombian_energy(self) -> float :
        """Compute the total Couloumbian energy for the system 
        
        Returns 
        -------

        float 
            Total Couloumbian enegy
        """
        energy = 0.0
        replicated_system = self.build_replicate([3,3,1])
        for i,pos_i in enumerate(self.system.positions) :
            for j,pos_j in enumerate(replicated_system.positions) :
                if i == j :
                    continue
                else :
                    energy += 0.5*self.dict_charge[self.system[i].symbol]*self.dict_charge[replicated_system[j].symbol]*self.compute_electro(pos_i,pos_j)
        return energy

    def compute_coulombian_forces(self) -> np.ndarray :
        """Compute the total Couloumbian forces for the system 
        
        Returns 
        -------

        float 
            Total Couloumbian forces"""
        force = []
        replicated_system = self.build_replicate([3,3,1])
        for i,pos_i in enumerate(self.system.positions) :
            force_i = np.array([0.0,0.0,0.0])
            for j,pos_j in enumerate(replicated_system.positions) :
                if i == j :
                    continue
                else :
                    force_i += self.dict_charge[self.system[i].symbol]*self.dict_charge[replicated_system[j].symbol]*self.compute_electro_forces_j_sur_i(pos_i,pos_j)

            force.append(force_i)
        return np.array([force])