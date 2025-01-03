import os


activate_venv = 'source $HOME/my_personal_env/bin/activate'
logfile_path = "/p/lustre1/nexouser/data/StanfordData/ChargeModule/LXe_Run1/"

jobname = "red-6g24pt-sig"
#change path!
input_path = "/p/lustre1/nexouser/data/StanfordData/ChargeModule/LXe_Run1/Gamma_Data_7_16_24/prereduced/6g24pt_sig/"
#change path!
output_path = "/p/lustre1/nexouser/data/StanfordData/ChargeModule/LXe_Run1/Gamma_Data_7_16_24/reduced/6g24pt_sig/"

config_path = "$HOME/cryo-asic-analysis/config/gamma-post-surg-24.yml"


cmd_options = '--export=ALL -p pbatch -t 1:00:00 -n 1 -J {} -o {}{}.out'.format(jobname, logfile_path, jobname)
exe = 'python $HOME/cryo-asic-analysis/submission_scripts/python/Reduce_Data.py {} {} {}'.format(input_path, output_path, config_path)
cmd_full = '{} && sbatch {} --wrap=\'{}\''.format(activate_venv,cmd_options,exe)

print(cmd_full)
os.system(cmd_full)
print('job {} sumbitted'.format(jobname))