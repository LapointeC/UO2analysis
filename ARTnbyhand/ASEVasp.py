from ase.calculators.vasp import Vasp
from ase import Atoms
import ase.dft.kpoints

"""set initial inputs for INCAR"""
xc='pbe'
algo='Fast'
isym=0
ediff=1e-6
ediffg=-1e-2
ismear=2
sigma=0.1
prec = 'Normal'
lreal = 'Auto'
ispin=2
ibrion=2
isif=2
nsw=0
encut=300.0
ncore=20
nelm=70
lcharg=False
lwave=False
kpoints_grid = [1,1,1]
restart = False
"""set initial inputs for INCAR"""

dic_correspondance = {'xc':xc,'algo':algo,'isym':isym,'ediff':ediff,'ediffg':ediffg,
    'ismear':ismear,'sigma':sigma,'prec':prec,'lreal':lreal,'ispin':ispin,'ibrion':ibrion,
    'isif':isif,'nsw':nsw,'encut':encut,'ncore':ncore,'nelm':nelm,'lcharg':lcharg,'lwave':lwave,'kpts':ase.dft.kpoints.monkhorst_pack(kpoints_grid),'restart':restart}


def set_up_VaspCalculator(**kwargs) -> Vasp:
    """"""

    if 'new_setup' in kwargs.keys():
        new_setup=kwargs['new_setup']
        for key_word in new_setup.keys() :
            dic_correspondance[key_word] = new_setup[key_word]

        calc = Vasp(xc=dic_correspondance['xc'],isym=dic_correspondance['isym'],
        ediff=dic_correspondance['ediff'],ediffg=dic_correspondance['ediffg'],ismear=dic_correspondance['ismear'],sigma=dic_correspondance['sigma'],
        prec=dic_correspondance['prec'],lreal=dic_correspondance['lreal'],ispin=dic_correspondance['ispin'],ibrion=dic_correspondance['ibrion'],
        isif=dic_correspondance['isif'],nsw=dic_correspondance['nsw'],encut=dic_correspondance['encut'],ncore=dic_correspondance['ncore'],nelm=dic_correspondance['nelm'],lcharg=dic_correspondance['lcharg'],lwave=dic_correspondance['lwave'],kpts=dic_correspondance['kpts'],restart=dic_correspondance['restart'])

    else :
        calc = Vasp(xc=xc, algo=algo, isym = isym, ediffg=ediffg , ediff=ediff, ismear=ismear, sigma=sigma, prec = prec, lreal = lreal, ispin=ispin,
                ibrion=ibrion, isif=isif,  nsw=nsw, encut=encut, ncore=ncore, nelm=nelm ,lcharg=lcharg,lwave=lwave,kpts=ase.dft.kpoints.monkhorst_pack(kpoints_grid),restart=restart)

    calc.__dict__['spinpol'] = False
    return calc
