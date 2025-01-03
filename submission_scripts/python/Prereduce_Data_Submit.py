import os


activate_venv = 'source $HOME/my_personal_env/bin/activate'
logfile_path = "/p/lustre1/nexouser/data/StanfordData/ChargeModule/LXe_Run1/Gamma_Data_7_16_24/"

jobname = "prered-6g24pt-bkg"
#change path! For this, put a glob string WITHOUT the filetag at the end
input_globstring = "/p/lustre1/nexouser/data/StanfordData/ChargeModule/LXe_Run1/Gamma_Data_7_16_24/Source_Retracted_B*2.4*"
#change path!
output_path = "/p/lustre1/nexouser/data/StanfordData/ChargeModule/LXe_Run1/Gamma_Data_7_16_24/prereduced/6g24pt_bkg/"

config_path = "$HOME/cryo-asic-analysis/config/gamma-post-surg-24.yml"


cmd_options = '--export=ALL -p pbatch -t 1:00:00 -n 1 -J {} -o {}{}.out'.format(jobname, logfile_path, jobname)
exe = "python $HOME/cryo-asic-analysis/submission_scripts/python/Prereduce_Data.py '{}' {} {}".format(input_globstring, output_path, config_path)
cmd_full = '{} && sbatch {} --wrap=\"{}\"'.format(activate_venv,cmd_options,exe)

print(cmd_full)
os.system(cmd_full)
print('job {} sumbitted'.format(jobname))
