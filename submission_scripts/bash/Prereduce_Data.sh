#!/bin/bash
#SBATCH --account=ppml
#SBATCH --partition=pbatch
#SBATCH --job-name=Gamma_Data_Pickle
#SBATCH --ntasks=1
#SBATCH --time 5:00:00
#SBATCH --output=/p/lustre2/glenrich/Mock_Charge_Module_Analysis/cryo-asic-analysis/submission_scripts/output/Gamma_Data_Pickle.log

python3 /p/lustre2/glenrich/Mock_Charge_Module_Analysis/cryo-asic-analysis/submission_scripts/python/Convert_to_Pickle.py "/p/lustre1/nexouser/data/StanfordData/ChargeModule/LXe_Run1/Gamma_Data_Post_Surgery_7_15_24/" "/p/lustre2/nexouser/data/StanfordData/ChargeModule/LXe_Run1_Processed_Data/Gamma_Data_Post_Surgery_7_15_24/"