import os
import sys

from input_output.read_msms import read_msms
# from triangulation.xyzrn import output_pdb_as_xyzrn
from RNA_make_surface_order.triangulation.xyzrn_new import output_pdb_as_xyzrn
# from default_config.global_vars import msms_bin
import random

# Calls MSMS and returns the vertices.
# Special atoms are atoms with a reduced radius.

def computeMSMS(pdb_file,  protonate=True, Mg = None, pdb_id=None):
    randnum = random.randint(1, 10000000)
    #file_base = rna_opts['temp_mid']+"/msms_"+ pdb_id #str(randnum)
    file_base = "/media/wangkagn/757283ec-3725-4b25-b458-8286d61ecf6a/spring_river/xiao_fen_zi/1uudtest/temp_mid"+"/msms_"+ pdb_id
    # str(randnum)
    out_xyzrn = file_base+".xyzrn"

    if protonate:
        #Mg_coords = output_pdb_as_xyzrn(pdb_file, out_xyzrn, Mg)
        Mg_coords = output_pdb_as_xyzrn(pdb_file, out_xyzrn, Mg = None)
    else:
        print("Error - pdb2xyzrn is deprecated")
        sys.exit(1)

    # now run MSMS on xyzrn file
    cmd = "msms -density 3.0 -hdensity 3.0 -probe 1.5 -if " + out_xyzrn + " -of " + file_base + " -af " + file_base
    os.system(cmd)

    vertics, faces, normals, names = read_msms(file_base)
    areas = {}
    ses_file = open(file_base+".area")
    next(ses_file)  # ignore header line
    for line in ses_file:
        fields = line.split()
        areas[fields[3]] = fields[1]

    # Remove temporary files.
    #os.remove(file_base+'.area')
    #os.remove(file_base + '.xyzrn')
    #os.remove(file_base + '.vert')
    #os.remove(file_base + '.face')
    return vertics, faces, normals, names, areas, Mg_coords

#pdb_file = '/media/wangkagn/757283ec-3725-4b25-b458-8286d61ecf6a/spring_river/xiao_fen_zi/1uudtest/1uudqt/1uud.pdbqt'
#id = "1uud"
#computeMSMS(pdb_file,protonate=True, Mg=None,pdb_id=id)

