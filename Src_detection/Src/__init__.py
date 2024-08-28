from .analysis import DfctAnalysisObject, NormDescriptorHistogram, MCDAnalysisObject
from .clusters import Cluster, ClusterDislo, DislocationObject
from .mld import Milady, DBDictionnaryBuilder, GenerateMiladyInput, DBManager, Optimiser, Regressor, Descriptor, DescriptorsHybridation, write_milady_poscar
from .tools import FrameOvito, NaiveOvitoModifier, MCDModifier, LogisticModifier, my_cfg_reader, timeit, DataPhondy, RecursiveCheck, RecursiveBuilder, nearest_mode
from .thermic import HarmonicThermicGenerator, ThermicSampling, ThermicFiting, SolidAse, FastEquivariantDescriptor
from .metrics import PCA_, PCAModel, MCD, MCDModel, Logistic, LogisticRegressor, GMM, GMMModel

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
    return 
                        
print_logo()