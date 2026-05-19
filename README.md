# Create the environment:
conda env create -f environment.yml
conda activate rlsfmode

# Quick Start:
1. Prepare the RNA receptor (PDB format) and ligand (SDF format)
2. Run docking to obtain score and RNA-ligand pdb
3. make RLSF：
   1. python run /RNA_make_surface_order/1-pdb_protonate.py
   2. python run /RNA_make_surface_order2-pdb_triangulate.py
   3. python run /RNA_make_surface_order3_rna_precompute.py
   4. python run /RNA_make_surface_order4_rna_site.py
4. get RNA sequence embedding
   python run /sequenve_embedding/sequence.py
5. Train
   Input file:
   RLSF(.npy 112*112*10), RNA sequence embedding(.txt 1*768), rdock_score(.txt, 1*4)
   python run train.py --task_type pose --data_dir #binding pose
   python run train.py --task_type affinity --data_dir #binding affinity
7. Test
   you can prepare all data needed or use the example (data and .pth) we provided to start a quick test
   python run test_pose.py #binding pose
   python run test_affinity.py #binding affinity

   
   
   
