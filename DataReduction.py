import yaml 
import os
import pandas as pd 
import numpy as np
from scipy.signal import find_peaks
import pickle
import Utilities as Util
import Pulse

class DataReduction:
	#Config is the "analysis config" file in configs, or a dictionary
	#that has been edited in the notebook (either filepath or dictionary of the yaml file)
	#The input_files is a list of filenames of what you want to reduce. For example, a list
	#from glob that selects all files with gain 6 and 1.2 pt from some directory. Full path expected.
	
	def __init__(self, input_files, config):

		self.configfile_or_dict = config
		self.config = None #has global analysis config dictionary contents
		self.chmap = None #has the channel and tile mappings. 
		self.load_config(config) #loads both of the above dictionaries

		#in the config file, there is a key for a file path that identifies
		#all of the reduced quantities. 
		self.rq_dict = None
		self.load_rq_dict() #Populates that dictionary with the entirity of the reducedquantities yaml file

		#The reduced_df is a dictionary with many keys associated with the rqs. Pulses and Clusters
		#are stored as lists of Pulse and Cluster objects. At the end of reduction, the dict is turned
		#into a pandas df so that analyses can be performed with slicing. 
		self.reduced_df = {} 

		#a temporary waveform_df object for the waveforms that are being analyzed
		#in the present file. Gets repopulated as one loads the next prereduced file. 
		self.waveform_df = None #this is the waveform df imported by the input files. 

		#list of prereduced filepaths in .p form at the moment. 
		self.input_files = input_files
		

	def load_config(self, config):
		#the config input is either a path to a yaml file or a dictionary.
		#load the yaml file if it is a path
		if(type(config) == str):
			with open(config, 'r') as stream:
				try:
					self.config = yaml.safe_load(stream)
				except yaml.YAMLError as exc:
					print(exc)
		else:
			self.config = config


		#now that the config is loaded, load the channel map file that
		#is referenced in the config. Check if it exists
		if(os.path.isfile(self.config["chmap"]) == False):
			print("Cant find the channel map file: " + str(self.config["channel_map"]))
			self.chmap = None
			return 
		
		with open(self.config["chmap"], 'r') as stream:
				try:
					self.chmap = yaml.safe_load(stream)
				except yaml.YAMLError as exc:
					print(exc)
		#done 


	def load_rq_dict(self):
		#load the reduced quantities dictionary
		if(os.path.isfile(self.config["reduced_quantities"]) == False):
			print("Cant find the reduced quantities file: " + str(self.config["reduced_quantities"]))
			self.rq_dict = None
			return 
		
		with open(self.config["reduced_quantities"], 'r') as stream:
				try:
					self.rq_dict = yaml.safe_load(stream)
				except yaml.YAMLError as exc:
					print(exc)
		#done


	#returns an empty, initialized event where each key's element
	#can be appended to the reduced_df keys of the same name. 
	def get_empty_event(self):
		
		event = {}

		for key in self.rq_dict["global"]:
			event[key] = self.rq_dict["global"][key] #initialize to the default value specified in the yaml file. 

		# Note we implictly skip dummy channels by only looping over the channel map
		for key in self.rq_dict["channel_rqs"]:
			for asic in self.chmap:
				for xch in self.chmap[asic]["xstrips"]:
					chid = Util.get_unique_id(asic, xch)
					event["ch{:d} {}".format(chid, key)] = []
			
				for ych in self.chmap[asic]["ystrips"]:
					chid = Util.get_unique_id(asic, ych)
					event["ch{:d} {}".format(chid, key)] = []

		
		#clusters and pulses are stored as lists of Pulse and Cluster
		#objects inside the event dictionary.
		event["clusters"] = []
		event["pulses"] = []


		return event


	#The path is the full path of output 
	#The filename is the name of the file you want to save with no extensions. 
	#It checks if the path exists and creates it if possible. 
	def save_reduced_df(self, path, filename):
		if(path[-1] != '/'):
			path += '/'
		
		if(os.path.exists(path) == False):
			os.makedirs(path)
		
		#check the filetag of the filename and correct it to .p if it is not already.
		if(filename.split('.')[-1] != 'p'):
			tag_removed = filename.split('.')[:-1]
			filename += tag_removed+'.p'

		#first convert to dataframe
		if(isinstance(self.reduced_df, dict)):
			df = pd.DataFrame.from_dict(self.reduced_df)
			pickle.dump([df], open(path+filename, 'wb'))
		else:
			print("Somehow the self.reduced_df became something other than a dict.")
			print("Write some handling code in save_reduced_df to handle this")
			pickle.dump([self.reduced_df], open(path+filename, 'wb'))

	

	def reduce_data(self, Skip_Baseline=False):

		#There may be an infinite amount of data files input to this reduction code. 
		#Instead of loading all of them and combining into a big waveform_df, we will
		#load each one, reduce each one, build up a big reduced_df that is a culmination
		#of all of the waveform_df files. Two key elements of the reduced_df are the
		#filename and evidx within that filename, used to re-index events to their origin. 

		for infile in self.input_files:
			print("Reducing file {}".format(infile))

			#Evan removed a part here that allowed the user to give raw data to this function,
			#doing the pre-reduction step. I do not want the user to have the flexibility to do this.
			#it creates too much file handling and organizational issues that we don't want to be responsible for. 
			#The user must (1) pre-reduce the raw data, then (2) reduce it with this class, and organize accordingly. 

			if(infile.split('.')[-1] == 'p'):
				self.waveform_df = pickle.load(open(infile, 'rb'))[0]

			else:
				print('Unrecognized file type .{0} given to data reducer. Please check file paths and try again.'.format(infile.split('.')[-1]))
				return
			
			#create a reduced dictionary that has reduced quantities for all events. 
			#this will be concatenated at the end of this file iteration to the self.reduced_df.
			red_df = {} 

			#some operations, like baseline subtraction, are much better
			#to perform on a numpy array as a vectorized operation. For that,
			#we unpack this dataframe into a numpy array of shape 
			#wavs[events][channels][samples].shape = (n_events, n_channels, n_samples)
			wavs = np.array(self.waveform_df["Data"].to_list())
			chidx_map = np.array(self.waveform_df["Channels"].to_list())[0]

			#convert all sample values from ADC to ENC
			wavs = Util.ADC_to_ENC(wavs, self.config["gain"], self.config["pt"])

			#baseline subtract the waveforms. This function
			#will also extracts information related to baselines,
			#like the std and means.
			wavs, extracted = self.analyze_and_subtract_baselines(wavs)

			#its also convenient here to get the full waveform stds 
			full_stds = np.std(wavs, axis=2) #for all events and all channels
			#add the extracted info to our red_df
			for chidx in range(len(wavs[0])):
				#get the unique, ASIC-number agnostic channel ID
				ch = chidx_map[chidx]
				red_df["ch{:d} baseline".format(ch)] = extracted["baselines"][:, chidx]
				red_df["ch{:d} baseline_std".format(ch)] = extracted["stds"][:, chidx]
				red_df["ch{:d} full_std".format(ch)] = full_stds[:, chidx]

			#the min and max value of all channels can also be vectorized, and would
			#be simple if not for the glitch pulses that we have to ignore/max certain
			#regions for. So in the future, you can replace this with one line like np.max(wavs, axis=2)
			#but for now, we have a special function that calls a utility. 
			extracted = self.analyze_min_max(wavs)
			#add the extracted info to our red_df
			for chidx in range(len(wavs[0])):
				#get the unique, ASIC-number agnostic channel ID
				ch = chidx_map[chidx]
				red_df["ch{:d} min".format(ch)] = extracted["min"][:, chidx]
				red_df["ch{:d} max".format(ch)] = extracted["max"][:, chidx]



			#add some global reduced quantities that are simple at this stage
			#first, Glenn likes to use a filenum at the end of filenames, so save that as a quick variable
			#in addition to saving the full filename
			file_num = (((infile.split('/')[-1]).split('_')[-1]).split('.')[0])[4:]
			red_df["filenum"] = [file_num]*len(self.waveform_df.index)
			red_df["filename"] = [infile]*len(self.waveform_df.index)
			red_df["evidx"] = list(range(len(self.waveform_df.index)))


			#Begin reduction tasks that involve looping event by event. 
			for evno, row in self.waveform_df.iterrows():
				if(evno % 500 == 0): print("On event {:d} of {:d}".format(evno, len(self.waveform_df.index)))
				
				#get an empty event to fill in
				red_ev = self.get_empty_event()

				#

				
				self.reduced_df["filenum"].append(file_num)
				self.reduced_df["evidx"].append(i)
				self.reduced_df["timestamp"].append(row["Timestamp"])

				# To fill in the cluster and pulse reduced quantities we start from the 
				# bottom and work our way up - identify pulses in the window and then
				# fill in pulses and once that's done we group them together to fill
				# in cluster information and then finally complete the relevent
				# global information

				# Temporarily writing to our temp column of pulses while we work on clustering algorithm
				self.reduced_df["pulse"].append(self.find_pulses(i,row))

				# As of this build, baseline data is by far the slowest, so we give the option to skip it for speed if desired
				if not Skip_Baseline:
					self.fill_in_baselines(row)

	#takes in a numpy array of all waves in a file. 
	def analyze_and_subtract_baselines(self, wavs):
		#get the baseline window in samples. 
		bl_window = [int(self.config["baseline"][0]*self.config["sampling_rate"]), int(self.config["baseline"][1]*self.config["sampling_rate"])]

		#in a vectorized way, get the baselines and std values for all channels
		baselines = np.apply_along_axis(Util.find_baseline_windowed, 2, wavs, bl_window)
		stds = np.apply_along_axis(Util.find_baseline_stds_windowed, 2, wavs, bl_window)
		sub_wavs = wavs - baselines[:, :, np.newaxis]
		return sub_wavs, {"baselines": baselines, "stds": stds}

	def analyze_min_max(self, wavs):
		ignore_regions = self.config["ignore_regions"]
		#turn into units of samples
		ignore_regions = [[int(region[0]*self.config["sampling_rate"]), int(region[1]*self.config["sampling_rate"])] for region in ignore_regions]
		#initialize a mask 
		samples = wavs.shape[2]
		mask = np.ones(samples, dtype=bool)
		for region in ignore_regions:
			if(region[1] >= samples):
				region[1] = samples-1
			if(region[0] < 0):
				region[0] = 0
			mask[region[0]:region[1]] = False

		masked_waves_min = np.where(mask, wavs, np.inf) #np inf will replace masked values, so a minimum function ignores them
		masked_waves_max = np.where(mask, wavs, -1*np.inf) #np -inf will replace masked values, so a minimum function ignores them

		return {"min": np.min(masked_waves_min, axis=2), "max": np.max(masked_waves_max, axis=2)}

	def find_pulses(self, event, row, n_sigma=4, width=30):


		pulse_df = {}
		
		# Checking to see if user has given a custom threshold value
		if n_sigma is None:
			n_sigma = self.config["pulse_threshold"]

		for ch in row["Channels"]:

			# Note we skip dummy channels for the actual event anlaysis too
			if not is_channel_strip(self.chmap, ch):
				continue

			wvfm = row["Data"][ch]
			temp_pulses, params = find_peaks(wvfm, width=self.config["pt"], height = n_sigma*np.std(wvfm)+np.mean(wvfm), wlen=width)
			pulses = []
			
			for p in temp_pulses:
				
				#this is because at the time of writing, we did not know how to remove the glitch pulses
				#that are injected as leakage currents by the internal calibration pulser. 
				if not (any(window[0] <= p <= window[1] for window in self.config["ignore_regions"])):
					pulses.append(p)

			for i, peak in enumerate(pulses):

				## Taking a convention where each pulse will be listed as "Pulse {event}-{pulse number within the event}"
				pulse_df["Pulse {0}-{1}".format(event, i)] = Pulse.Pulse(self.config, self.rq_dict["pulse"], wvfm, i, params).d
				(pulse_df["Pulse {0}-{1}".format(event, i)])["channel"] = ch

		return pulse_df





			
	
		
