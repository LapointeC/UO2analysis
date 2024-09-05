import numpy as np
from scipy.linalg import expm

from typing import Dict

#######################
## C15
#######################
class Octa : 
    def __init__(self) : 
        self.center = np.zeros(3)
        self.link = np.zeros((3, 4))
        self.atom = np.zeros((3, 4))
        self.removeatom = np.zeros((3, 6))
        self.gaos = np.zeros((3, 3, 4))

class MetaOctahedron:
    def __init__(self):
        self.typeOcta : Dict[str, Octa] = {'type1':Octa(),
                                           'type2':Octa()}

        #Type 1
        # The coordinates of the tetrahedron
        self.typeOcta['type1'].atom[:, 0] = np.array([-1, -1, -1])  # 000
        self.typeOcta['type1'].atom[:, 1] = np.array([ 1,  1, -1])  # 220
        self.typeOcta['type1'].atom[:, 2] = np.array([ 1, -1,  1])  # 202
        self.typeOcta['type1'].atom[:, 3] = np.array([-1,  1,  1])  # 022

        # The other six atoms which must be removed:
        self.typeOcta['type1'].removeatom[:, 0] = np.array([ 0,  0, -2])
        self.typeOcta['type1'].removeatom[:, 1] = np.array([ 0,  2,  0])
        self.typeOcta['type1'].removeatom[:, 2] = np.array([ 0,  0,  2])
        self.typeOcta['type1'].removeatom[:, 3] = np.array([ 0, -2,  0])
        self.typeOcta['type1'].removeatom[:, 4] = np.array([-2,  0,  0])
        self.typeOcta['type1'].removeatom[:, 5] = np.array([ 2,  0,  0])

        # The four directions which can be linked
        self.typeOcta['type1'].link[:, 0] = np.array([ 1,  1,  1])
        self.typeOcta['type1'].link[:, 1] = np.array([-1, -1,  1])
        self.typeOcta['type1'].link[:, 2] = np.array([ 1, -1, -1])
        self.typeOcta['type1'].link[:, 3] = np.array([-1,  1, -1])

        self.typeOcta['type1'].gaos[:, :, 0] = np.array([
            [ 1,  1, -1],
            [ 1, -1,  1],
            [-1,  1,  1]
        ])
        self.typeOcta['type1'].gaos[:, :, 1] = np.array([
            [-1, -1, -1],
            [-1,  1,  1],
            [ 1, -1,  1]
        ])
        self.typeOcta['type1'].gaos[:, :, 2] = np.array([
            [-1,  1,  1],
            [-1, -1, -1],
            [ 1,  1, -1]
        ])
        self.typeOcta['type1'].gaos[:, :, 3] = np.array([
            [ 1, -1,  1],
            [ 1,  1, -1],
            [-1, -1, -1]
        ])

        self.typeOcta['type1'].center = np.array([1, 1, 1])

        #Type 2 initialisation
        self.typeOcta['type2'].center = np.array([2, 2, 2])

        # The second cube has the center in 222 and is rotated 90 degrees counterclockwise
        rotation_z = self.rotation_axis_matrix(np.array([0.0, 0.0, 1.0]))
        
        # tetrahedron
        self.typeOcta['type2'].atom = rotation_z(0.5*np.pi) @ self.typeOcta['type1'].atom

        # gaos 
        self.typeOcta['type2'].gaos = rotation_z(0.5*np.pi) @ self.typeOcta['type1'].gaos

        # link 
        self.typeOcta['type2'].link = rotation_z(0.5*np.pi) @ self.typeOcta['type1'].link

        # to remove atoms 
        self.typeOcta['type2'].removeatom = rotation_z(0.5*np.pi) @ self.typeOcta['type1'].removeatom

    def rotation_axis_matrix(self, axis : np.ndarray) :
        """build the rotation matrix method arrond a given axis"""
        return lambda theta : expm(np.cross(np.eye(3), axis/np.linalg.norm(axis)*theta))
    

#######################
## A15
#######################
class Ico :
    def __init__(self) : 
        self.center = np.zeros(3)
        self.link = np.zeros((3, 4))
        self.sia = np.zeros((3, 12))
        self.removeatom = np.zeros((3, 6))
        self.add = np.zeros((3, 1))

class MetaIcosahedron:
    def __init__(self):
        self.typeIco : Dict[str, Ico] = {'type1':Ico(),
                                         'type2':Ico()}


        # Remove atoms (each one on a face)
        self.typeIco['type1'].removeatom[:, 0] = np.array([-1.0,  0.0,  0.0])  # 011
        self.typeIco['type1'].removeatom[:, 1] = np.array([ 1.0,  0.0,  0.0])  # 211
        self.typeIco['type1'].removeatom[:, 2] = np.array([ 0.0,  0.0, -1.0])  # 110
        self.typeIco['type1'].removeatom[:, 3] = np.array([ 0.0,  0.0,  1.0])  # 112
        self.typeIco['type1'].removeatom[:, 4] = np.array([ 0.0, -1.0,  0.0])  # 101
        self.typeIco['type1'].removeatom[:, 5] = np.array([ 0.0,  1.0,  0.0])  # 121

        # Atom in the center
        self.typeIco['type1'].add[:, 0] = np.array([0.0, 0.0, 0.0])  # 111

        # The other 12 atoms in SIAs position
        self.typeIco['type1'].sia[:, 0] = np.array([-1.0,  0.0,  0.5])  # 011
        self.typeIco['type1'].sia[:, 1] = np.array([-1.0,  0.0, -0.5])  # 011
        self.typeIco['type1'].sia[:, 2] = np.array([ 1.0,  0.0,  0.5])  # 211
        self.typeIco['type1'].sia[:, 3] = np.array([ 1.0,  0.0, -0.5])  # 211
        self.typeIco['type1'].sia[:, 4] = np.array([ 0.0,  0.5, -1.0])  # 110
        self.typeIco['type1'].sia[:, 5] = np.array([ 0.0, -0.5, -1.0])  # 110
        self.typeIco['type1'].sia[:, 6] = np.array([ 0.0,  0.5,  1.0])  # 112
        self.typeIco['type1'].sia[:, 7] = np.array([ 0.0, -0.5,  1.0])  # 112
        self.typeIco['type1'].sia[:, 8] = np.array([ 0.5, -1.0,  0.0])  # 101
        self.typeIco['type1'].sia[:, 9] = np.array([-0.5, -1.0,  0.0])  # 101
        self.typeIco['type1'].sia[:,10] = np.array([ 0.5,  1.0,  0.0])  # 121
        self.typeIco['type1'].sia[:,11] = np.array([-0.5,  1.0,  0.0])  # 121

        # Type 2
        # Atom in the center
        self.typeIco['type2'].add[:, 0] = np.array([0.0, 0.0, 0.0])  # 111
        
        #rotation matrix along 111
        rotation_along_111_matrix = self.rotation_axis_matrix(np.array([1.0, 1.0, 1.0]))
        
        # remove atoms 
        self.typeIco['type2'].removeatom = rotation_along_111_matrix(np.pi/3.0) @ self.typeIco['type1'].removeatom

        # sia
        self.typeIco['type2'].sia = rotation_along_111_matrix(np.pi/3.0) @ self.typeIco['type1'].sia

    def rotation_axis_matrix(self, axis : np.ndarray) :
        """build the rotation matrix method arrond a given axis"""
        return lambda theta : expm(np.cross(np.eye(3), axis/np.linalg.norm(axis)*theta))