"""
pdbqt_to_pdb.py: Wrapper method for the oenbabel program: pdbqt_to_pdb (i.e., add hydrogens) a pdbqt file
                and save to an output file.
"""


import os


def pdbqt_to_pdb(in_pdbqt_file, out_pdb_file):
    cmd = "obabel -ipdbqt " + in_pdbqt_file + " -O " + out_pdb_file
    print(cmd)
    os.system(cmd)