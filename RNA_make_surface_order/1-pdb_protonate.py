import os
from input_output.protonate import protonate
import numpy as np

tmp_dir = ''
rna_pdb_dir = ''

pdb_ids = []
pdb_filename = []

RNA_pdb_names = os.listdir(rna_pdb_dir)

for i, d in enumerate(RNA_pdb_names):
    d = d.split(".")
    d = d[0]+66
    pdb_ids.append(d)
error_list = []
number = 0
for pdb_id in pdb_ids:
    pdb_filename = rna_pdb_dir + pdb_id + ".pdb"
    number = number + 1
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    protonated_file = tmp_dir+"/"+pdb_id+".pdbqt"

    try:
        protonate(pdb_filename, protonated_file)  # Unable Mg atom.
    except:
        error_list.append(pdb_id)
        continue


np.save("error_pdbs.npy", error_list)






