#!/bin/bash
#SBATCH --account=ppml
#SBATCH --partition=pbatch
#SBATCH --job-name=Pickle_Data
#SBATCH --ntasks=1
#SBATCH --time 0:10:00
#SBATCH --output=/p/lustre2/glenrich/Mock_Charge_Module_Analysis/cryo-asic-analysis/submission_scripts/output/Convert_to_Pickle.log

python3 /p/lustre2/glenrich/Mock_Charge_Module_Analysis/cryo-asic-analysis/submission_scripts/python/Convert_to_Pickle.py "/p/lustre2/nexouser/data/StanfordData/ChargeModule/LXe_Run1/HV_Noise_Test_After_New_Filter/" "/p/lustre2/nexouser/data/StanfordData/ChargeModule/LXe_Run1_Processed_Data/HV_Noise_Test_After_New_Filter/"