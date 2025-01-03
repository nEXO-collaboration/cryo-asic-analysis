import sys
import os
import glob
sys.path.append("../../")
import CryoAsicFile


def Get_File_Name(file_path):
  
  file_name = file_path.split('/')[-1]
  if file_name.split('.') != 2: file_name = file_name.split('.')[0] + file_name.split('.')[1]
  else: file_name = file_name.split('.')[0]
  return file_name


def Convert_Files(input_globstring, output_path, config_filepath, reload_all=True):


  dat_files = glob.glob(input_globstring +"*.dat")
  pickle_files = glob.glob(output_path + "*.p")
  pickle_names = [Get_File_Name(p) for p in pickle_files]

  #check if output directory exists
  if not os.path.isdir(output_path): os.mkdir(output_path)

  for dat in dat_files:
    dat_name = Get_File_Name(dat)
    if (dat_name not in pickle_names) or (reload_all==True):
      cf = CryoAsicFile.CryoAsicFile(dat, config_filepath)
      cf.load_raw_data()
      cf.group_into_pandas()
      outfile_name = output_path + dat_name + '.p'
      cf.pickle_dump_waveform_df(outfile_name)


if __name__=="__main__":
  if(len(sys.argv) != 4):
    print("Usage: python Prereduce_Data.py input_globstring output_path config_filepath")
    print("Input globstring should have a full path to data files and NOT have a file extension. ")
    sys.exit()

  input_globstring = sys.argv[1]
  output_path = sys.argv[2]
  config_filepath = sys.argv[3]

  Convert_Files(input_globstring, output_path, config_filepath)