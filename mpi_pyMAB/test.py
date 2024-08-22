import xml.etree.ElementTree as ET
import numpy as np

def read_parameters(xml_parameters:ET.Element) -> None:
    """Read in simulation parameters defined in the XML file 
    Parameters
    ----------
    xml_parameters : xml.etree.ElementTreeElement
        The <Parameters> branch of the configuration file,
        represented as an ElementTree Element
    
    """
    parameters = {"Temperature":10.0,
                  "LambdaGrid":{"Min":0.0,"Max":1.0,"Number":100}}

    def boolean_converter(str_xml : str) -> bool :
        """Convert string chain into boolean 
        
        Parameters 
        ----------
        str_xml : str
            string to convert 
        Returns 
        -------
        
        bool
        """
        if str_xml in ['true' ,'True', '.TRUE.','.True.', 'Yes'] : 
            return True
        if str_xml in ['false', 'False', '.FALSE.', '.False.' ,'No'] :
            return False

    for var in xml_parameters:
        tag = var.tag.strip()
        if not tag in parameters:
            print(f"Undefined parameter {tag}!!, skipping")
            continue
        else:
            o = parameters[tag]
            n = var.text
            if isinstance(o,int):
                parameters[tag] = int(n)
            elif isinstance(o,float):
                parameters[tag] = float(n)
            elif isinstance(o,bool):
                parameters[tag] = boolean_converter(n)
            elif isinstance(o,str):
                parameters[tag] = n
            elif isinstance(o,list):
                parameters[tag] = n
            elif isinstance(o,dict):
                for child in var : 
                    tag_child = child.tag
                    co = parameters[tag][tag_child]
                    cn = child.text
                    if isinstance(co,int) :
                        parameters[tag][tag_child] = int(cn)
                    elif isinstance(co,float) :
                        parameters[tag][tag_child] = float(cn)
                    elif isinstance(co,str) :
                        parameters[tag][tag_child] = cn            
                    elif isinstance(co,bool) : 
                        parameters[tag][tag_child] = boolean_converter(cn)

#xml_path = 'mab.xml'

#xml_tree = ET.parse(xml_path)
#for branch in xml_tree.getroot():
#    if branch.tag=="Parameters":
#        read_parameters(branch)

def convert_lambda_dict(lambda_dict) -> np.ndarray : 
    """Convert the lambda dictionnary into np array grid"""
    buffer_array_left = np.linspace(lambda_dict['Min']-lambda_dict['Rbuffer'],lambda_dict['Min'], num=lambda_dict['NumberBuffer'], endpoint=False)
    buffer_array_right = np.linspace(lambda_dict['Max'], lambda_dict['Max']+lambda_dict['Rbuffer'], num=lambda_dict['NumberBuffer']+1)
    main_array_lambda = np.linspace(lambda_dict['Min'], lambda_dict['Max'], num=lambda_dict['Number'], endpoint=False)
    
    if len(buffer_array_left) > 0 :
        main_array_lambda = np.concatenate((buffer_array_left,main_array_lambda),axis=0)
    
    return  np.concatenate((main_array_lambda,buffer_array_right), axis=0) #np.linspace(lambda_dict['Min'], lambda_dict['Max'], num=lambda_dict['Number'])

dic_lamb = {"Min":0.0,"Max":1.0,"Number":100,"Rbuffer":0.1,"NumberBuffer":5} 
print(convert_lambda_dict(dic_lamb))