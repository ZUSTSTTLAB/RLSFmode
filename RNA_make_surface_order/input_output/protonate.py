"""
protonate.py: Wrapper method for the MGLTools program: protonate (i.e., add hydrogens) a pdb using prepare_receptor4.py
                and save to an output file.
"""
import os


def protonate(in_pdb_file, out_pdb_file):
    # protonate (i.e., add hydrogens) a pdb using reduce and save to an output file.
    # in_pdb_file: file to protonate.
    # out_pdb_file: output file where to save the protonated pdbqt file.
    cmd = "mgltools_x86_64Linux2_1.5.6/bin/pythonsh " \
          "mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py -r " \
    + in_pdb_file +" -o " + out_pdb_file + " -A checkhydrogens"

    os.system(cmd)

#+ " -A checkhydrogens"