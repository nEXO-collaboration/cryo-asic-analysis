import sys
import glob
sys.path.append("../../")
import CryoAsicFile


def Get_File_Name(file_path):
  
  file_name = file_path.split('/')
  if file_name.split('.') != 2: file_name = file_name.split('.')[0] + file_name.split('.')[1]
  else: file_name = file_name.split('.')[0]
  return file_name


def Convert_Files(data_path, pickle_path, reload_all = False):

  dat_files = glob.glob(data_path + ".dat")
  pickle_files = glob.glob(pickle_path + ".p")
  pickle_names = [Get_File_Name(p) for p in pickle_files]

  print("Picke list is {0}">format(pickle_names))
  print("Starting .dat file name loop")
  for dat in dat_files:
    
    dat_name = Get_File_Name(dat)

    print(dat_name)

    if (dat_name not in pickle_names) or (reload_all==True):
      cf = CryoAsicFile.CryoAsicFile(dat, config_filepath)
      cf.load_raw_data()
      cf.group_into_pandas()
      outfile_name = pickle_path + dat_name + '.p'
      cf.pickle_dump_waveform_df(outfile_name)


if __name__=="__main__":
  
  print("Begining Conversion")
  data_path = sys.argv[1]
  pickle_path = sys.argv[2]
  print("Data path is {0}".format(data_path))
  print("Pickle path is {0}".format(pickle_path))
  config_filepath = "../../config/analysisconfig.yml"

  Convert_Files(data_path, pickle_path, True)
  print("Conversion Complete")