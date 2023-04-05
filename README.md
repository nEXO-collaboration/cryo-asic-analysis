This is a set of python classes that allow for quick viewing and "script" level analysis of single files of CRYO ASIC data. These classes were originally written with the intention of using them within a broader framework where there may be charge-collection tiles attached to the CRYO ASIC, there may be half of the CRYO ASIC channels attached to dummy capacitors, and there may be multiple tiles. 

For this reason, there are two types of configuration files: channel_map and tile_map. The channel_map indexes each CRYO ASIC channel with a tile ID, strip-type associated with the channel, and relative position of that strip within the tile. The tile_map associates each tile ID with an absolute position within a detector. For analyzing CRYO ASIC data without a tile, you can set these to whatever - they are only interpreted by a broader analysis framework. For CRYO ASICs with half their channels attached to dummy capacitors, I have associated those channels with their own tile. 

## Basic usage
In your own script or python notebook, the beginning of basic usage would look like: 

```
import CryoAsicAnalysis
import CryoAsicEventViewer
import CryoAsicFile

infile = "path/to/data_timestamp.dat" #output binary file from CRYO ASIC software
cf = CryoAsicFile.CryoAsicFile(infile, "channel_map_template.txt", "tile_map_template.txt")
cf.load_raw_data() #loads the binary data into an event list
cf.group_into_pandas() #packages the waveform data alongside tile/channel map metadata

outfile_name = "path/to/data_timestamp" #any filename, without a suffix
cf.save_to_hdf5(outfile_name) #saves the pandas dataframe into an hdf5 file, for future use by analysis/viewer classes

#a set of configuration parameters for your analysis.
#key_channel : this channel is disabled, very specific purpose to look for misalignment issues
#dead_channels : marked for analysis purposes
#baseline : used for calculating baseline calibrations, noise calculations, etc. 
config = {
    "baseline":[0, 30000], #us \ 
    "sampling_rate":2, \
    "pulse_threshold": 100, \
    "mv_per_adc": 1800./4096, \
    "key_channel": 32, \
    "dead_channels": [1, 32, 33, 37, 38, 39]
}

ca = CryoAsicAnalysis.CryoAsicAnalysis(outfile_name+".h5", config) #an analysis object that loads the hdf5 pandas dataframe
#perform analysis tasks in that class, for example, a power spectral density avg over all events
ca.calculated_avg_psds()
```

