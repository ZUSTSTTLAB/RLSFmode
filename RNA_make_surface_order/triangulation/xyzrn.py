from Bio.PDB import *
from default_config.chemistry import radii, polarHydrogens

"""
xyzrn.py: Read a pdb file and output it is in xyzrn for use in MSMS
This file is part of RNA_NET
"""

def output_pdb_as_xyzrn(pdbfilename, xyzrnfilename, Mg = None):
    """
        pdbfilename: input pdb filename
        xyzrnfilename: output in xyzrn format.
    """
    parser = PDBParser()
    struct = parser.get_structure(pdbfilename, pdbfilename)
    outfile = open(xyzrnfilename, "w")
    for atom in struct.get_atoms():
        name = atom.get_name()
        residue = atom.get_parent()



        resname = residue.get_resname()
        chain = residue.get_parent().get_id()
        atomtype = name[0]

        color = "Green"
        coords = None
        if Mg == True:
            if atomtype in radii and resname[0] in polarHydrogens:
                if atomtype == "M":
                    color = "Purple"
                    coords = "{:.06f} {:.06f} {:.06f}".format(
                        atom.get_coord()[0], atom.get_coord()[1], atom.get_coord()[2]
                )

                insertion = "x"
                if residue.get_id()[2] != " ":
                    insertion = residue.get_id()[2]
                full_id = "{}_{:d}_{}_{}_{}_{}".format(
                    chain, residue.get_id()[1], insertion, resname, name, color
                )
        else:
            # ignore Mg
            if residue.get_id()[0] == "H_MG":
                continue
            if atomtype in radii and resname[0] in polarHydrogens:
                if atomtype == "O":
                    color = "Red"
                if atomtype == "N":
                    color = "Blue"
                if atomtype == "H":  # this code is useless
                    if name[0] in polarHydrogens[resname[0]]:
                        color = "Blue"  # Polar hydrogens
                coords = "{:.06f} {:.06f} {:.06f}".format(
                    atom.get_coord()[0], atom.get_coord()[1], atom.get_coord()[2]
                )

                insertion = "x"
                if residue.get_id()[2] != " ":
                    insertion = residue.get_id()[2]
                full_id = "{}_{:d}_{}_{}_{}_{}".format(
                    chain, residue.get_id()[1], insertion, resname, name, color
                )
        if coords is not None:
            outfile.write(coords + " " + radii[atomtype] + " 1 " + full_id + "\n")









