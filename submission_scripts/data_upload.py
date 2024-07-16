import os
import sys
import time
import shutil
from datetime import datetime
import glob


def Transfer_Files(source, destination):
    os.system('rsync -Pav {0} {1}'.format(source, destination))
    print("sync complete")


def Pardon_Files(pardon_location, file):
    
    sub_run_name = file.split('/')[-2]
    file_name = file.split('/')[-1]

    if not os.path.exists(pardon_location + '/' + sub_run_name):
        os.makedirs(pardon_location + '/' + sub_run_name)

    shutil.copy2(file, pardon_location + '/' + sub_run_name +'/' + file_name)

    print('Moving file for checks:')
    print('Moved file: ' + file)



def Delete_Files(buffer_directory, sanity_check_dir, kill_time, counter, pardon_count = 10):

    file_list = glob.glob(buffer_directory+'/*')
    kill_list = []
    
    for file in file_list:
        last_modified_time = os.path.getmtime(file)
        if last_modified_time < (time.time() - kill_time):
            kill_list.append(file)
            counter += 1
            if counter%pardon_count == 0: # Move every [tenth] file to a seperate directory before deletion for sanity checks during the day
                #Pardon_Files(sanity_check_dir, file)
                counter = 0 # Resetting counter to avoid large numbers during runtime
        elif "TEST" in file: # Move test files to the main saved directory and then delete them if they haven't been modified in the last minute
            if last_modified_time < (time.time() - 60):
                shutil.copy2(file, sanity_check_dir+"/"+file)
                os.remove(file)
            
    
    for death_row in kill_list: # Delete files that have lasted longer than the kill time
         os.remove(death_row)
        
    return kill_list, counter


def main(upload_rate, message_rate, kill_time):

    source_directory = '/media/asicdaq/02a27d49-74c8-4298-88e1-d96ac453d3dd/cryoasic/LXe_Run1/Gamma_Data_Post_Surgery_7_15_24' # Don't include the trailing / here so that rsync copies the full directory and not just the files inside
    sanity_directory = '/media/asicdaq/02a27d49-74c8-4298-88e1-d96ac453d3dd/cryoasic/LXe_Run1/Saved_Data' # Pardon function assumes there's no trailing / for consistancy
    
    target_borax = 'glenrich@borax.llnl.gov:/p/lustre1/nexouser/data/StanfordData/ChargeModule/LXe_Run1/' # Use trailing / here to create the directoy the first transfer
    target_slac = 'glenn96@s3dflogin.slac.stanford.edu:/sdf/group/exo/ChargeModule2024/LXe_Run1/'

    # Checking if pardon counter has been created yet - only excepts on first pass
    try: counter
    except: counter = 0

    print('\n'+'*'*40)
    print('*'*40)
    print("Uploading data to S3DF - Please enter S3DF password")
    print('*'*40)
    print('*'*40 + '\n')
    Transfer_Files(source_directory, target_slac)

    print('\n'+'*'*40)
    print('*'*40)
    print("Uploading data to Borax - Please enter Borax password and OPT")
    print('*'*40)
    print('*'*40 + '\n')
    Transfer_Files(source_directory, target_borax)

    # Counting time until next upload - only reason to not do it continously is because of password input
    start_time = time.time()
    while time.time() < (start_time+upload_rate):
        print('\n Status: Waiting - Current Time is {0} - {1} min until next upload'.format(datetime.now().strftime('%H:%M'), int((start_time+upload_rate-time.time())/60)))
        line_time = time.time()
        while time.time() < (line_time + message_rate): # Printing to screen to check program is still runing at a glance
            sys.stdout.write('.')
            sys.stdout.flush()
            time.sleep(message_rate/40)
        print('\n Deleting old files ... ')
        kill_list, counter = Delete_Files(source_directory, sanity_directory, kill_time, counter) # Deleting 
        print('Files Deleted: ' + str([file for file in kill_list]))



if __name__=="__main__":
    while True: # Run uplaod script until ctrl-C kill order from user
        main(upload_rate=5*60, message_rate=1*60, kill_time = 20*60) # Upload data every 30 min while running and putting time remaining every 5 min, and deleting files over 2 hour old