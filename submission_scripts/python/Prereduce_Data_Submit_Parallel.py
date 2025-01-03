import os
import glob 
#looks at all input files and creates parallel jobs for each. 


def Get_File_Name(file_path):
  
  file_name = file_path.split('/')[-1]
  if file_name.split('.') != 2: file_name = file_name.split('.')[0] + file_name.split('.')[1]
  else: file_name = file_name.split('.')[0]
  return file_name


def submit():

    activate_venv = 'source $HOME/my_personal_env/bin/activate'
    logfile_path = "/p/lustre1/nexouser/data/StanfordData/ChargeModule/LXe_Run1/Gamma_Data_7_16_24/prereduced/logfiles/"

    jobname = "prered-6g24pt-bkg"
    #change path! For this, put a glob string WITHOUT the filetag at the end
    input_globstring = "/p/lustre1/nexouser/data/StanfordData/ChargeModule/LXe_Run1/Gamma_Data_7_16_24/Source_Retracted_B*2.4*"
    #change path!
    output_path = "/p/lustre1/nexouser/data/StanfordData/ChargeModule/LXe_Run1/Gamma_Data_7_16_24/prereduced/6g24pt_bkg/"

    config_path = "$HOME/cryo-asic-analysis/config/gamma-post-surg-24.yml"


    input_files = glob.glob(input_globstring+"*.dat")
    pickle_files = glob.glob(output_path + "*.p")
    pickle_filenames = [Get_File_Name(p) for p in pickle_files]

    for infile in input_files:
        infile_name = Get_File_Name(infile)
        if infile_name in pickle_filenames:
            print("File {} already exists in output directory. Skipping".format(infile_name))
            continue

        #file tag
        tagnumber = infile.split('_')[-1].split('.')[0]
        #create a job for this particular file
        this_jobname = jobname + tagnumber
        cmd_options = '--export=ALL -p pbatch -t 0:15:00 -n 1 -J {} -o {}{}.out'.format(jobname, logfile_path, this_jobname)
        exe = "python $HOME/cryo-asic-analysis/submission_scripts/python/Prereduce_Data.py '{}' {} {}".format(infile, output_path, config_path)
        cmd_full = '{} && sbatch {} --wrap=\"{}\"'.format(activate_venv,cmd_options,exe)

        print(cmd_full)
        os.system(cmd_full)
        print('job {} sumbitted'.format(jobname))

if __name__ == "__main__":
   submit()