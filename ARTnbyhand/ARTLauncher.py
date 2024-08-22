import os, sys
from ARTManager import ARTManager
from ARTParser import ARTParser

argv = sys.argv
if len(argv) == 1 : 
    xml_file = [f for f in os.listdir(os.getcwd()) if 'xml' in f][0]
elif len(argv) == 2 :
    xml_file = sys.argv[1]
else : 
    print('Number of arguments is wrong to launch ARTn python ...')
    print('Exiting program ...')
    exit(0)

ART_parser = ARTParser(xml_file)
ART_manager = ARTManager(ART_parser.dic_molecule,
                        ART_parser.parameters,
                        ART_parser.pathways,
                        ART_parser.dumps,
                        ART_parser.calculator,
                        restart_file=ART_parser.dumps['PickleFiles'])

if not os.path.exists(ART_parser.pathways['DumpPath']) :
    os.mkdir(ART_parser.pathways['DumpPath'])

if not os.path.exists(ART_parser.pathways['CalcPath']) : 
    os.mkdir(ART_parser.pathways['CalcPath'])
    os.chdir(ART_parser.pathways['CalcPath'])

ART_manager.run_ART()