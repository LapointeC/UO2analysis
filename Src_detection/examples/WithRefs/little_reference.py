import os, sys
import pickle
from typing import List, Dict, Any

sys.path.insert(0,'/home/marinica/GitHub/UO2analysis.git/Src_detection')
os.system('pwd') 
#from Src import ReferenceBuilder
from Src.parser.BaseParser import UNSEENConfigParser
from Src.analysis.reference import ReferenceBuilder
from Src.mld.milady import ComputeDescriptor


def split_path(path: str) -> tuple[str, str]:
    # If the path normalizes to '.', return './' for both directory and base name.
    if os.path.normpath(path) == '.':
        return './', './'
    
    # For non-root paths that end with a separator, remove the trailing separator.
    # (Note: We keep the separator for root '/', which should remain as is.)
    if len(path) > 1 and path.endswith(os.sep):
        path = path.rstrip(os.sep)
    
    # Split the path into directory and base name.
    dir_name, base_name = os.path.split(path)
    
    # If the directory part is not empty, ensure it ends with a separator.
    if dir_name and not dir_name.endswith(os.sep):
        dir_name += os.sep
        
    return dir_name, base_name

def print_fancy_header(message: str) -> None:
    """Prints a fancy header for better readability"""
    print("=" * 50)
    print(f"{message:^50}")
    print("=" * 50)


def print_config(title: str, config: Dict[str, Any]) -> None:
    """Prints a configuration dictionary in a structured way"""
    print(f"\n{title}:")
    for key, value in config.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for subkey, subvalue in value.items():
                print(f"    {subkey}: {subvalue}")
        else:
            print(f"  {key}: {value}")

if __name__ == "__main__":
    VERSION = '0.90'
    print(f"UNSEEN TOKEN VERSION: {VERSION}\n")
    
    # Configuration printing functions (same as before)
    # ...

    # Initialize configurations
    auto_config = {}
    custom_config = []

    # Load Auto config
    auto_file = "auto_config.xml"
    if os.path.exists(auto_file):
        print_fancy_header(f"Reading Auto Config from {auto_file}")
        auto_parser = UNSEENConfigParser(auto_file)
        auto_config = auto_parser.auto_config
        #print_config("Auto Configuration", auto_config)
    else:
        print_fancy_header(f"{auto_file} not found. Using Default Auto Config")
        auto_parser = UNSEENConfigParser(auto_file)  # Will create defaults
        auto_config = auto_parser.auto_config

    # Load Custom config
    custom_file = "custom_config.xml"
    if os.path.exists(custom_file):
        print_fancy_header(f"Reading Custom Config from {custom_file}")
        custom_parser = UNSEENConfigParser(custom_file)
        custom_config = custom_parser.custom_config
        #print_config("Custom Configuration", {"References": custom_config})
    else:
        print_fancy_header(f"{custom_file} not found. No Custom References")
        
        
   # compute descriptors if needed ...    
   
   
    # ---------------------------------------------------------
    # Run descriptor computation for each configuration.
    # Descriptor time ... 
    # ---------------------------------------------------------

    # For the Auto configuration:
    #put current directory into some variable: 
    cdir = os.getcwd()
    
    if auto_config:
        # Get the MD files directory, MD file format, and a name for the pickle file
        auto_directory = auto_config.get('directory', './')
        dir_where, dir_name = split_path(auto_directory)
        auto_md_format = auto_config.get('md_format', 'cfg')
        # Use the 'name' field to build the pickle filename. For example, if name is "bulk_BCC":
        auto_pickle_file = f"{auto_config.get('name', 'auto')}_auto_data.pickle"
        auto_config['pickle_data'] = auto_pickle_file
        
        print_fancy_header(f"Running descriptor computation for Auto configuration")
        print(f"Directory: {auto_directory}")
        print(f"MD file format: {auto_md_format}")
        print(f"Output pickle file: {auto_pickle_file}\n")
        
        # Create an instance of ComputeDescriptor with the XML options
        os.chdir(dir_where)
        cd_auto = ComputeDescriptor(path_bulk=dir_name,
                                    pickle_data_file=auto_pickle_file,
                                    md_format=auto_md_format)
        cd_auto.compute()
        os.chdir(cdir)
    ## For each Custom reference configuration:
    os.system('pwd')
    print_config("Auto Configuration", auto_config)
    
    
    if custom_config:
        for ref in custom_config:
            # For each custom reference, extract the directory, md_format, and name.
            ref_directory = ref.get('directory', './')
            dir_where, dir_name = split_path(ref_directory)
            ref_md_format = ref.get('md_format', 'cfg')
            ref_pickle_file = f"{ref.get('name', 'ref')}_ref_data.pickle"
            ref['pickle_data'] = ref_pickle_file   
            print_fancy_header(f"Running descriptor computation for Custom Reference: {ref.get('name', 'ref')}")
            print(f"Directory: {ref_directory}")
            print(f"MD file format: {ref_md_format}")
            print(f"Output pickle file: {ref_pickle_file}\n")
            
            os.chdir(dir_where)         
            cd_custom = ComputeDescriptor(path_bulk=dir_name,
                                          pickle_data_file=ref_pickle_file,
                                          md_format=ref_md_format)
            cd_custom.compute()  
            os.chdir(cdir)
    print_config("Custom Configuration", {"References": custom_config})
    # ---------------------------------------------------------
    # Build models get the references ... 
    #----------------------------------------------------------
    os.system('pwd')
    
    try:
        builder = ReferenceBuilder(species='Fe', auto_config=auto_config, custom_config=custom_config)
        
        if auto_config:
            builder.process_auto_config()
        
        # if custom_config:
        #     builder.process_custom_references()
            
        print("\n" + "="*50)
        print("Model Building Complete".center(50))
        print("="*50)
        
    except Exception as e:
        print(f"\nError during model building: {str(e)}")
        sys.exit(1)