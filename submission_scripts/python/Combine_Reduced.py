import os
import glob
import pickle
import pandas as pd
import sys


def combine(output_path):

    #combine the reduced files
    print("Combining reduced files")
    combined_df = pd.DataFrame()
    infiles = glob.glob(output_path+"*.p")
    for i, infile in enumerate(infiles):
        if(i %100 == 0):
            print("Combining file {:d} of {:d}".format(i, len(infiles)))
        if("combined" in infile):
            continue
        df = pickle.load(open(infile, "rb"))[0]
        if(i == 0):
            combined_df = df
        else:
            combined_df = pd.concat([combined_df, df], ignore_index=True)

    pickle.dump([combined_df], open(output_path+"combined.p", "wb"))

if __name__ == "__main__":
    if(len(sys.argv) != 2):
        print("Usage: python Combine_Reduced.py output_path")
        sys.exit()
        
    output_path = sys.argv[1]
    combine(output_path)