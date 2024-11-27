import sys, argparse, os

sys.path.insert(0,'../')
from Src import NearestMode, SurfaceParser, AseVaspParser, \
                 CompositionFromBulk, BuilderSurfaceOriented, \
                 SetupVaspASE, WritingSlurm, SetupVaspSlabASE, \
                 RecursiveChecker, CheckProblems, SetRelaunch, \
                 DataSurface, ReadConvergenceFile, ExtractPathSlab, ComputeandWriteParametricSurfaceEnergy, \
                 ReadFileNormalVector, RelativeStabilityOnGrid, \
                 PlotProjectionHyperplaneInto2D, PlotAllHyperplanes3D
                 

# ase time + pymatgen
from ase import Atoms
import ase.io

dic_cluster = {'occigen':'sbatch','irene':'ccc_msub','jean_zay':'sbatch --account=aih@cpu','explor':'sbatch','topaze':'ccc_msub'}

"""
Here is just the launching part of the script ...
"""
parser = argparse.ArgumentParser('SurfaceBuilder')
parser.add_argument('-p','--parameters',default="SurfaceBuilder.xml")
parser.add_argument('-vasp', '--vasp_parameters',default='AseVasp.xml')
parser.add_argument('-m','--mode',default='slab_builder')
args = parser.parse_args()

xml_file = args.parameters if os.path.exists(args.parameters) else 'SurfaceBuilder.xml'
xml_vasp_file = args.vasp_parameters if os.path.exists(args.vasp_parameters) else 'VaspAse.xml'
mode = args.mode

list_mode = ['slab_builder','slab_energy_launcher','just_check','check_and_relaunch','extract_surface_energy','plot_and_stability']
if mode not in list_mode :
    print('There is something wrong with mode')
    possible_mode = NearestMode(list_mode,mode,3)
    str = ''
    for string in possible_mode : 
        str += '%s, '%(string)
    print('Maybe you mean : %s...'%(str[:-2]))
    exit(0)

"""This routine build all the adsorption structure for a given slab and a given molecule by using : build operation
    next you can use launch option to launch all vasp calculations"""
Parser_xml = SurfaceParser(xml_file,mode)
Parser_vasp_xml = AseVaspParser(xml_vasp_file)

###################################################################
### here is slab and formation enthalpy launching part
###################################################################
if mode == 'slab_builder' :
    """generate slab configuration from the bulk (based on Florian code)"""
    composition = CompositionFromBulk(Parser_xml.parameters['PathBulk'])
    if not os.path.exists(Parser_xml.parameters['WritingSlabs']) :
        os.mkdir(Parser_xml.parameters['WritingSlabs'])

    for orientation in Parser_xml.parameters['hklVectors'] :
        print('Slab will be generated for orientation (%s,%s,%s)'%(orientation[0],orientation[1],orientation[2]))
        BuilderSurfaceOriented(Parser_xml.parameters['PathBulk'],
                                orientation,
                                Parser_xml.parameters['HeighPymatgen'],
                                Parser_xml.parameters['Vaccum'],
                                composition,
                                Parser_xml.parameters['Levels'],
                                Parser_xml.parameters['HeighConstraint'],
                                Parser_xml.parameters['ToleranceHeigh'],
                                Parser_xml.parameters['WritingSlabs'],
                                tolerance_surface=Parser_xml.parameters['ToleranceAtomSurface'],
                                check_sym=Parser_xml.parameters['CheckSym'],
                                alpha_list=Parser_xml.parameters['AlphaParameters'],
                                tolerance_descripteur=Parser_xml.parameters['ToleranceDescriptors'],
                                dense_check=Parser_xml.parameters['CheckDense'],
                                remesher=Parser_xml.parameters['RemeshSurface'])
        print()

    print('Enjoy :)')

###################################################################
### here is slab and formation enthalpy launching part
###################################################################
if mode == 'slab_energy_launcher' :
    list_path_slab = [ '%s/%s'%(Parser_xml.parameters['PathSlabs'],el) for el in os.listdir(Parser_xml.parameters['PathSlabs']) ]
    read_slab=ase.io.read(list_path_slab[0], format='vasp')
    single_atoms=read_slab.get_chemical_symbols()

    """generate the list of single elements"""
    real_single_atoms = []
    for atom in single_atoms :
        if atom not in real_single_atoms :
            real_single_atoms.append(atom)

    """create the computation directory"""
    if not os.path.exists(Parser_xml.parameters['PathComput']) :
        os.mkdir(Parser_xml.parameters['PathComput'])
        os.mkdir('%s/slab'%(Parser_xml.parameters['PathComput']))
        os.mkdir('%s/elements'%(Parser_xml.parameters['PathComput']))
        os.mkdir('%s/bulk_full'%(Parser_xml.parameters['PathComput']))

    for atom in real_single_atoms :
        """computation for each el"""
        bulk_atom=ase.io.read('%s/%s.cif'%(Parser_xml.parameters['PathCif'],atom),format='cif')
        single_atom=Atoms(atom, positions=[(0,0,0)],cell=(12,12,12),pbc=(1,1,1))

        if not os.path.exists('%s/elements/%s'%(Parser_xml.parameters['PathComput'],atom)) :
            os.mkdir('%s/elements/%s'%(Parser_xml.parameters['PathComput'],atom))
        path = '%s/elements/%s'%(Parser_xml.parameters['PathComput'],atom)

        """prepare vasp calculation
        ==> first single atom """
        if not os.path.exists('%s/single'%(path)) :
            os.mkdir('%s/single'%(path))

        ase.io.write('%s/single/POSCAR'%(path),single_atom,sort=True,direct=True, vasp5=True,format='vasp')
        kpar = SetupVaspASE(Parser_vasp_xml,Parser_xml.parameters['InputsVASP'],'%s/single'%(path),Parser_xml.parameters['ReciprocalDensity'],single_atom.cell[:],"fast",atom,new_setup=dic_setup)
        WritingSlurm('%s/single'%(path),Parser_xml.parameters['SlurmCommand'],'single_%s'%(atom),Parser_xml.parameters['NProcs'],kpar)
        os.chdir('%s/single'%(path))
        os.system('%s surface.slurm'%(Parser_xml.parameters['SlurmCommand']))

        """
        ===> bulk time"""
        if not os.path.exists('%s/bulk'%(path)) :
            os.mkdir('%s/bulk'%(path))

        ase.io.write('%s/bulk/POSCAR'%(path),bulk_atom,sort=True,direct=True, vasp5=True,format='vasp')      
        """isif have to be set at 3 for bulk element to relaxation volume"""
        dic_setup = {'isif':3}
        kpar = SetupVaspASE(Parser_vasp_xml,Parser_xml.parameters['InputsVASP'],'%s/bulk'%(path),Parser_xml.parameters['ReciprocalDensity'],bulk_atom.cell[:],"normal",atom,new_setup=dic_setup)
        WritingSlurm('%s/bulk'%(path),Parser_xml.parameters['SlurmCommand'],'bulk_%s'%(atom),Parser_xml.parameters['NProcs'],kpar)
        os.chdir('%s/bulk'%(path))
        os.system('%s surface.slurm'%(Parser_xml.parameters['SlurmCommand']))

    """time for bulk calculation"""
    path_work_bulk = '%s/bulk_full/%s'%(Parser_xml.parameters['PathComput'],os.path.basename(Parser_xml.parameters['PathBulk']).split('.')[0])
    if not os.path.exists(path_work_bulk) :
        os.mkdir(path_work_bulk)
    os.system('cp %s %s/POSCAR'%(Parser_xml.parameters['PathBulk'],path_work_bulk))
    bulk_system = ase.io.read(Parser_xml.parameters['PathBulk'], format='vasp')
    """isif have to be set at 3 for bulk element to relaxation volume"""
    dic_setup = {'isif':3}
    kpar = SetupVaspSlabASE(Parser_vasp_xml,Parser_xml.parameters['InputsVASP'],path_work_bulk,Parser_xml.parameters['ReciprocalDensity'],bulk_system.cell[:],"normal",[0.0,0.0,0.0],new_setup=dic_setup)
    WritingSlurm(path_work_bulk,Parser_xml.parameters['SlurmCommand'],os.path.basename(Parser_xml.parameters['PathBulk']).split('.')[0],Parser_xml.parameters['NProcs'],kpar)
    os.chdir(path_work_bulk)
    os.system('%s surface.slurm'%(Parser_xml.parameters['SlurmCommand']))

    """time for slab calculation"""
    for path_slab in list_path_slab :
        read_slab=ase.io.read(path_slab, format='vasp')
        cell_slab = read_slab.cell[:]

        if not os.path.exists('%s/slab'%(Parser_xml.parameters['PathComput'])) :
            os.mkdir('%s/slab'%(Parser_xml.parameters['PathComput']))

        path_work = '%s/slab/%s'%(Parser_xml.parameters['PathComput'],os.path.basename(path_slab).split('.')[0])
        if not os.path.exists(path_work) :
            os.mkdir(path_work)
        os.system('cp %s %s/POSCAR'%(path_slab,path_work))
    
        """isif have to be set at 2, no volume relaxation for the slab"""
        dic_setup = {'isif':2}
        kpar = SetupVaspSlabASE(Parser_vasp_xml,Parser_xml.parameters['InputsVASP'],path_work,Parser_xml.parameters['ReciprocalDensity'],bulk_system.cell[:],"normal",[0.0,0.0,1.0],new_setup=dic_setup)
        
        WritingSlurm(path_work,Parser_xml.parameters['SlurmCommand'],os.path.basename(path_slab).split('.')[0],Parser_xml.parameters['NProcs'],kpar)
        os.chdir(path_work)
        os.system('%s surface.slurm'%(Parser_xml.parameters['SlurmCommand']))

    print('Enjoy :)')

####################################################################
#### here is the check mode
####################################################################
if mode == 'just_check' :
    list_unconverged  = RecursiveChecker(Parser_xml.parameters['PathComput'],'%s/check_convergence.log'%(Parser_xml.parameters['PathComput']))
    if len(list_unconverged) == 0 :
        print('All the calculations have converged :)')

    else :
        print('Some calculation not converged :(')
        for path in list_unconverged :
            print('==> %s'%(path))
        print('')
        CheckProblems(list_unconverged)

####################################################################

####################################################################
#### here is the check mode alternative
####################################################################
if mode == 'check_and_relaunch' :
    list_unconverged  = RecursiveChecker(Parser_xml.parameters['PathComput'],'%s/check_convergence.log'%(Parser_xml.parameters['PathComput']))
    if len(list_unconverged) == 0 :
        print('All the calculations have converged :)')

    else :
        print('Some calculations not converged :(')
        for path in list_unconverged :
            print('Relaunching ==> %s'%(path))
            try :
                SetRelaunch(path)
                os.chdir(path)
                os.system('%s surface.slurm'%(Parser_xml.parameters['SlurmCommand']))
            except :
                print('Error with ==> %s'%(path))

####################################################################


####################################################################
#### here is the extract mode 
####################################################################
if mode == 'extract_surface_energy' :
    if os.path.exists(Parser_xml.parameters['HyperplaneData']) :
        os.system('rm %s'%(Parser_xml.parameters['HyperplaneData']))
        os.system('rm %s/hp.data'%(os.path.dirname(Parser_xml.parameters['HyperplaneData'])))

    data_surface_object = DataSurface()
    filled_data_surface_object = ReadConvergenceFile(data_surface_object,'%s/check_convergence.log'%(Parser_xml.parameters['PathComput']))
    name_bulk = os.path.basename(Parser_xml.parameters['PathBulk']).split('.')[0]
    list_name_slab = [el.split('.')[0] for el in  os.listdir(Parser_xml.parameters['PathSlabs'])]
    list_path_slab = ExtractPathSlab(list_name_slab,'%s/check_convergence.log'%(Parser_xml.parameters['PathComput']))
    list_name_slab = [el.split('/')[-2] for el in list_path_slab]        
    ComputeandWriteParametricSurfaceEnergy(filled_data_surface_object,
                                            f'{os.path.basename(Parser_xml.parameters['HyperplaneData'])}/hp_full.data',
                                            list_name_slab,
                                            name_bulk,
                                            list_path_slab,
                                            Parser_xml.parameters['ElToRemove'])

    print('Enjoy ! :)')


####################################################################
#### here is the plotting mode 
####################################################################
if mode == 'plot_and_stability' : 
    path_to_hp_file = '%s/hp.data'%(os.path.dirname(Parser_xml.parameters['HyperplaneData']))
    dic_normal_vect, dic_mu, dic_bounds, dic_constraint = ReadFileNormalVector(path_to_hp_file)
    RelativeStabilityOnGrid(dic_bounds,Parser_xml.parameters['SlabsToPlot'],dic_normal_vect,dic_constraint,discr_per_mu=Parser_xml.parameters['MuDiscretisation'])

    """3D plot case"""
    all_key_mu = list(dic_mu.keys())
    if len(dic_mu[all_key_mu[0]]) == 4 :
        PlotProjectionHyperplaneInto2D(path_to_hp_file,Parser_xml.parameters['SlabsToPlot'],100,level_line=Parser_xml.parameters['LevelLines'])
        print('! Take care 3D plot not taking into account contraint due to removed element !')
        PlotAllHyperplanes3D(path_to_hp_file,Parser_xml.parameters['SlabsToPlot'])
    
    print('Enjoy ! :)')
####################################################################
