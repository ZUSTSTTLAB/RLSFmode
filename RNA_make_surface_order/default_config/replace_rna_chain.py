
def replace_rna_chain(in_pdb, out_pdb):
  iPDB = in_pdb
  fid = open(iPDB, "r")
  lines = fid.readlines()
  fid.close()

  oPDB = out_pdb
  fid = open(oPDB, "w")
  for i in range(len(lines)):
    line = lines[i]
    if i == 0:
      DISion = line[16:20]
    if len(line) < 4:
      fid.write(line)
      continue
    if line[0:6] not in ("ATOM  ", "HETATM"):
      fid.write(line)
      continue
    lst=list(line)
    lst[21]="R"
    line="".join(lst)
    fid.write(line)


  fid.close()