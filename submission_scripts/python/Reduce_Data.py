import sys
import glob
import pickle
import os
import pandas as pd

sys.path.append("../../")
import DataReduction 


def reduce(input_path, output_path, config_path):

    #output_path = "../../../data/MockTileRun1/Gamma_Data_Post_Surgery_7_15_24/reduced/"
    input_files = glob.glob(input_path+"*.p")
    try:
        print("We found {:d} files, one of which is path {}".format(len(input_files), input_files[0]))
    except:
        print("No files found in input directory")
        print("Does directory exist?: ", os.path.isdir(input_path))
        sys.exit()
        

    #initialize the DataReduction class
    dr = DataReduction.DataReduction(config_path)
    #load input data of many files, which combines the dataframes into one
    for i, infile in enumerate(input_files):
        print("Reducing file {}".format(infile))
        dr.load_prereduced_data(infile)
        dr.reduce_to_pulses() #does basic initial waveform processing and creates Pulse objects
        dr.process_pulses() #analyzes the pulses in detail to populate pulse reduced quantities
        dr.process_clusters() #clusters pulses into events and measures properties
        dr.process_globals() #measures global properties of the event from the clusters
        dr.dictify_objects() #deletes Pulse and Cluster objects, turning them into dictionaries in the reduced df
        dr.save_reduced_df(output_path, infile.split("/")[-1]) #saves the reduced df to a pickle file


    #combine the reduced files
    print("Combining reduced files")
    combined_df = pd.DataFrame()
    for i, infile in enumerate(glob.glob(output_path+"*.p")):
        if("combined" in infile):
            continue
        df = pickle.load(open(infile, "rb"))[0]
        if(i == 0):
            combined_df = df
        else:
            combined_df = pd.concat([combined_df, df], ignore_index=True)

    pickle.dump([combined_df], open(output_path+"combined.p", "wb"))

if __name__ == "__main__":
    if(len(sys.argv) != 4):
        print("Usage: python Reduce_Data.py input_path output_path config_path. Absolute paths please")
        sys.exit()
        
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    config_path = sys.argv[3]
    reduce(input_path, output_path, config_path)