import os
import sys
from default_config.rna_opts import rna_opts
from default_config.replace_rna_chain import replace_rna_chain

def pdb_read_file():
    path = rna_opts["MGdecoy"]
    RNA_pdb_names = os.listdir("../" + path)
    pdb_ids = []
    pdb_filename = []
    DiSion = {}

    for i, d in enumerate(RNA_pdb_names):
        pdb_filename.append(rna_opts['MGdecoy'] + d)
        d = d.split(".")
        d = d[0]
        pdb_ids.append(d)
        in_pdb = "../" + pdb_filename[i]
        out_pdb = "../" + rna_opts["out_replace_decoypdb_dir"] + pdb_ids[i] + ".pdb"
        DiSion[pdb_ids[i]] = replace_rna_chain(in_pdb, out_pdb)

    return pdb_ids, DiSion


