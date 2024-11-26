import os
from ase.calculators.vasp import Vasp
from ase import Atoms
from .SurfaceParser import AseVaspParser

def WriteIncarASE(pathINCAR : str, parser_object : AseVaspParser,**kwargs) -> None:
    """Write something about this function..."""
    os.chdir(pathINCAR)

    if 'new_setup' in kwargs.keys():
        new_setup=kwargs['new_setup']
        for key_word in new_setup.keys() :
            parser_object.parameters[key_word] = new_setup[key_word]

    calc = Vasp(xc=parser_object.parameters['xc'],isym=parser_object.parameters['isym'],
    ediff=parser_object.parameters['ediff'],ediffg=parser_object.parameters['ediffg'],ismear=parser_object['ismear'],sigma=parser_object.parameters['sigma'],
    prec=parser_object.parameters['prec'],lreal=parser_object.parameters['lreal'],ispin=parser_object.parameters['ispin'],ibrion=parser_object.parameters['ibrion'],
    isif=parser_object.parameters['isif'],nsw=parser_object.parameters['nsw'],encut=parser_object.parameters['encut'],ncore=parser_object.parameters['ncore'],
    nelm=parser_object.parameters['nelm'],lcharg=parser_object.parameters['lcharg'],lwave=parser_object.parameters['lwave'],kpts=parser_object.parameters['kpts'],restart=parser_object.parameters['restart'])

    atom_object = Atoms()
    calc.write_incar(atom_object)
    calc.write_kpoints(atom_object)
    return
##############################################################
