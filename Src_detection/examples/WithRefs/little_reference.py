import os, sys
import pickle
from typing import List, Dict, Any

sys.path.insert(0,'/home/marinica/GitHub/UO2analysis.git/Src_detection')
os.system('pwd') 
#from Src import ReferenceBuilder
from Src.parser.BaseParser import UNSEENConfigParser
from Src.analysis.reference import ReferenceBuilder


#if __name__ == "__main__":
#    VERSION='0.90'
#    print(f"UNSEEN TOKEN VERSION: {VERSION}\n")
#    
#    def print_fancy_header(message: str) -> None:
#        """Prints a fancy header for better readability"""
#        print("=" * 50)
#        print(f"{message:^50}")
#        print("=" * 50)
#
#    def print_config(title: str, config: Dict[str, Any]) -> None:
#        """Prints a configuration dictionary in a structured way"""
#        print(f"\n{title}:")
#        for key, value in config.items():
#            if isinstance(value, dict):
#                print(f"  {key}:")
#                for subkey, subvalue in value.items():
#                    print(f"    {subkey}: {subvalue}")
#            else:
#                print(f"  {key}: {value}")
#
#    # Check and process auto_config.xml
#    auto_file = "auto_config.xml"
#    if os.path.exists(auto_file):
#        print_fancy_header(f"Reading Auto Config from {auto_file}")
#        auto_parser = UNSEENConfigParser(auto_file)
#        print_config("Auto Configuration", auto_parser.auto_config)
#    else:
#        print_fancy_header(f"{auto_file} not found. Loading Default Auto Config")
#        auto_parser = UNSEENConfigParser(auto_file)
#        print_config("Default Auto Configuration", auto_parser.auto_config)
#
#    # Check and process custom_config.xml
#    custom_file = "custom_config.xml"
#    if os.path.exists(custom_file):
#        print_fancy_header(f"Reading Custom Config from {custom_file}")
#        custom_parser = UNSEENConfigParser(custom_file)
#        if custom_parser.custom_config:
#            print("\nCustom Config References:")
#            for i, ref in enumerate(custom_parser.custom_config, 1):
#                print(f"\nReference {i}:")
#                print_config("Reference Configuration", ref)
#        else:
#            print("\nNo custom references found in the file.")
#    else:
#        print_fancy_header(f"{custom_file} not found. No Custom References Loaded")
#        custom_parser = UNSEENConfigParser(custom_file)
#        print("No custom references were loaded as the file does not exist.")


#path=""
#a = ReferenceBuilder(path)

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
        print_config("Auto Configuration", auto_config)
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
        print_config("Custom Configuration", {"References": custom_config})
    else:
        print_fancy_header(f"{custom_file} not found. No Custom References")
        
        
   # compute descriptors if needed ...      

   # Build models
    try:
        builder = ReferenceBuilder(auto_config=auto_config, custom_config=custom_config)
        
        if auto_config:
            builder.process_auto_config()
        
        if custom_config:
            builder.process_custom_references()
            
        print("\n" + "="*50)
        print("Model Building Complete".center(50))
        print("="*50)
        
    except Exception as e:
        print(f"\nError during model building: {str(e)}")
        sys.exit(1)