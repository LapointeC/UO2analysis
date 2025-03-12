import os
import sys
from ase import io
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]

def convert_and_copy(directory, output_directory, each_of, elements):
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    files = sorted([f for f in os.listdir(directory) if f.endswith(".dump")], key=natural_sort_key)
    
    element_list = elements.split()
    
    for i, filename in enumerate(files):
        if i % each_of == 0:
            dump_path = os.path.join(directory, filename)
            base_name = os.path.splitext(filename)[0]  # Get the prefix of the file
            poscar_path = os.path.join(output_directory, f"{base_name}.poscar")
            
            # Convert dump to POSCAR
            atoms = io.read(dump_path, format="lammps-dump-text")
            
            if len(element_list) == 1:
                atoms.set_chemical_symbols([element_list[0]] * len(atoms))
            elif len(element_list) == len(set(atoms.get_atomic_numbers())):
                atoms.set_chemical_symbols(element_list)
            else:
                print(f"Warning: Number of elements provided does not match unique atomic types in {filename}.")
                continue
            
            io.write(poscar_path, atoms, format="vasp")
            print(f"Converted: {filename} -> {poscar_path} in {output_directory}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python script.py IN=<input_directory> OUT=<output_directory> EACHOF=<integer> ELEM=<elements>")
        sys.exit(1)
    
    args = {arg.split("=")[0]: arg.split("=")[1] for arg in sys.argv[1:]}
    input_directory = args.get("IN")
    output_directory = args.get("OUT")
    each_of = int(args.get("EACHOF", 1))
    elements = args.get("ELEM", "")
    
    convert_and_copy(input_directory, output_directory, each_of, elements)




