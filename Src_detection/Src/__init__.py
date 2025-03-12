from .analysis import DfctMultiAnalysisObject, NormDescriptorHistogram, MetricAnalysisObject, ReferenceBuilder
from .clusters import Cluster, ClusterDislo, DislocationObject
from .mld import Milady, DBDictionnaryBuilder, GenerateMiladyInput, DBManager, Optimiser, Regressor, Descriptor, DescriptorsHybridation, write_milady_poscar
from .tools import FrameOvito, NaiveOvitoModifier, MCDModifier, LogisticModifier, my_cfg_reader, timeit, DataPhondy, RecursiveCheck, RecursiveBuilder, nearest_mode, get_N_neighbour, build_extended_neigh_ , ComputeInteractiveModifier
from .thermic import HarmonicThermicGenerator, ThermicSampling, ThermicFiting, FastEquivariantDescriptor
from .metrics import PCA_, PCAModel, MCD, MCDModel, Logistic, LogisticRegressor, GMM, GMMModel, MetaModel
from .parser import UNSEENConfigParser
from .structure import SolidAse, DislocationsBuilder, C15Builder, A15Builder, InputsDictDislocationBuilder, InputsCompactPhases, NeighboursCoordinates
try :
    from .surface import NearestMode, SurfaceParser, AseVaspParser, CompositionFromBulk, BuilderSurfaceOriented, SetupVaspASE, WritingSlurm, \
                     SetupVaspSlabASE, RecursiveChecker, CheckProblems, SetRelaunch, DataSurface, ReadConvergenceFile, ExtractPathSlab, ComputeandWriteParametricSurfaceEnergy, \
                     ReadFileNormalVector, RelativeStabilityOnGrid, PlotProjectionHyperplaneInto2D, PlotAllHyperplanes3D
except ImportError : 
    print('I will not load Surface Builder Package... (No pymatgen)')

def print_logo() -> None :
    """
        A friendly welcome message :)
    """
    welcome = f"""
            ┳┳           ┏┳┓  ┓     
            ┃┃┏┓┏┏┓┏┓┏┓   ┃ ┏┓┃┏┏┓┏┓
            ┗┛┛┗┛┗ ┗ ┛┗   ┻ ┗┛┛┗┗ ┛┗     
    
    copyright CEA by ...                                        
    ... C. Lapointe, A.-G. Goryaeva, M.-C. Marinica                               
    email: clovis.lapointe@cea.fr, mihai-cosmin.marinica@cea.fr"""
    print(welcome)
    print()
    return 
                        
print_logo()
