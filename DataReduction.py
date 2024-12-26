import yaml 
import os
import pandas as pd 
import numpy as np
from scipy.signal import find_peaks
import pickle
import Utilities as Util
import Pulse
import matplotlib.pyplot as plt
import sys

class DataReduction:
	#Config is the "analysis config" file in configs, or a dictionary
	#that has been edited in the notebook (either filepath or dictionary of the yaml file)
	#The input_files is a list of filenames of what you want to reduce. For example, a list
	#from glob that selects all files with gain 6 and 1.2 pt from some directory. Full path expected.
	
	def __init__(self, config):

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
		self.reduced_df = {} #accumulates over many files. 

		#a temporary waveform_df object for the waveforms that are being analyzed
		#in the present file. Gets repopulated as one loads the next prereduced file. 
		self.waveform_df = None #this is the waveform df imported by the input files. 
		self.infile = None #current infile path. 
		self.red_df = {} #A temporary, one-file-only reduced dictionary that is concatenated to the self.reduced_df at the end of the file iteration.

		

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

	
	def load_prereduced_data(self, infile):
		#Evan removed a part here that allowed the user to give raw data to this function,
		#doing the pre-reduction step. I do not want the user to have the flexibility to do this.
		#it creates too much file handling and organizational issues that we don't want to be responsible for. 
		#The user must (1) pre-reduce the raw data, then (2) reduce it with this class, and organize accordingly. 
		
		if(infile.split('.')[-1] == 'p'):
			print("loading file {}".format(infile))
			self.waveform_df = pickle.load(open(infile, 'rb'))[0]
			self.infile = infile
			print("Done")

		else:
			print('Unrecognized file type .{0} given to data reducer. Please check file paths and try again.'.format(infile.split('.')[-1]))
			return


	def reduce_to_pulses(self):

		
		#create a reduced dictionary that has reduced quantities for all events. 
		#this will be concatenated at the end of this file iteration to the self.reduced_df.
		red_df = {} 

		#some operations, like baseline subtraction, are much better
		#to perform on a numpy array as a vectorized operation. For that,
		#we unpack this dataframe into a numpy array of shape 
		#wavs[events][channels][samples].shape = (n_events, n_channels, n_samples)
		wavs = np.array(self.waveform_df["Data"].to_list())
		chidx_map = np.array(self.waveform_df["Channels"].to_list())[0]

		#NOTE: tried to convert all wavs from ADC to ENC here so that all analysis operations
		#are in ENC units from this point on. It took way way way too long for some reason... 
		#possibly not the right vectorized syntax or something. This is why you see a bunch of 
		#calls to that function down below. 

		#baseline subtract the waveforms. This function
		#will also extracts information related to baselines,
		#like the std and means.
		print("Subtracting baselines")
		wavs, extracted = self.analyze_and_subtract_baselines(wavs)

		#its also convenient here to get the full waveform stds 
		print("Getting full STDs")
		full_stds = np.std(wavs, axis=2) #for all events and all channels
		#add the extracted info to our red_df
		for chidx in range(len(wavs[0])):
			#get the unique, ASIC-number agnostic channel ID
			ch = chidx_map[chidx]
			red_df["ch{:d} baseline".format(ch)] = Util.ADC_to_ENC(extracted["baselines"][:, chidx], self.config["gain"], self.config["pt"])
			red_df["ch{:d} baseline_std".format(ch)] = Util.ADC_to_ENC(extracted["stds"][:, chidx], self.config["gain"], self.config["pt"])
			red_df["ch{:d} full_std".format(ch)] = Util.ADC_to_ENC(full_stds[:, chidx], self.config["gain"], self.config["pt"])

		#the min and max value of all channels can also be vectorized, and would
		#be simple if not for the glitch pulses that we have to ignore/max certain
		#regions for. So in the future, you can replace this with one line like np.max(wavs, axis=2)
		#but for now, we have a special function that calls a utility. 
		print("Analyzing minimums and maximums")
		extracted = self.analyze_min_max(wavs)
		#add the extracted info to our red_df
		for chidx in range(len(wavs[0])):
			#get the unique, ASIC-number agnostic channel ID
			ch = chidx_map[chidx]
			red_df["ch{:d} min".format(ch)] = Util.ADC_to_ENC(extracted["min"][:, chidx], self.config["gain"], self.config["pt"])
			red_df["ch{:d} max".format(ch)] = Util.ADC_to_ENC(extracted["max"][:, chidx], self.config["gain"], self.config["pt"])



		#add some global reduced quantities that are simple at this stage
		red_df["filename"] = [self.infile]*len(self.waveform_df.index)
		red_df["evidx"] = list(range(len(self.waveform_df.index)))
		red_df["timestamp"] = self.waveform_df["Timestamp"].to_list()


		red_df["pulses"] = [[] for i in range(len(self.waveform_df.index))]
		red_df["n_pulses"] = []
		print("Initializing pulses for events with any sample above positive threshold of {:d} sigma".format(self.config["coarse_threshold"]))
		#do a np.where to find where any channel number is above threshold
		mask = None
		for chidx in range(len(wavs[0])):
			ch = chidx_map[chidx]
			if(not Util.is_channel_strip(self.chmap, ch)):
				continue
			maxs = red_df["ch{:d} max".format(ch)]
			threshs = self.config["coarse_threshold"]*red_df["ch{:d} baseline_std".format(ch)]
			if(mask is None):
				mask = np.where(maxs > threshs, 1, 0)
			#if a mask already exists, I want to OR it with the new mask
			else:
				mask = np.where(maxs > threshs, 1, 0) | mask
		

		#get red_ev indices that passed mask
		red_ev_idxs = np.where(mask == 1)[0]
		for ev_idx in red_ev_idxs:
			#loop through all strip channels and initialize pulse objects
			#for each channel that has a pulse above coarse threshold
			for chidx in range(len(wavs[0])):
				ch = chidx_map[chidx]
				if(not Util.is_channel_strip(self.chmap, ch)):
					continue
				
				if(red_df["ch{:d} max".format(ch)][ev_idx] > self.config["coarse_threshold"]*red_df["ch{:d} baseline_std".format(ch)][ev_idx]):
					p = Pulse.Pulse(self.rq_dict, self.config, wavs[ev_idx][chidx], ch)
					red_df["pulses"][ev_idx].append(p)

			#on a coarse level, we want to initialize pulse objects
			#for channels spatially adjacent 
			adjacent_pulses = []
			for p in red_df["pulses"][ev_idx]:
				#initialize the spatially adjacent pulses
				ch = p.ch 
				adj_chs = Util.get_adjacent_channels(self.chmap, ch, self.config["adjacency"])
				for adj in adj_chs:
					chidx = np.where(chidx_map == adj)[0][0]
					adjacent_pulses.append(Pulse.Pulse(self.rq_dict, self.config, wavs[ev_idx][chidx], adj))

			red_df["pulses"][ev_idx] += adjacent_pulses
			red_df["n_pulses"].append(len(red_df["pulses"][ev_idx]))

		self.red_df = red_df #store the reduced dictionary for this file.
		#later will be concatenated manually to the self.reduced_df.


	#does waveform level analysis on pulse objects, 
	#vetting them to either remove if they are meaningless
	#or calculate reduced quantities like energy and such. 
	def process_pulses(self):

		#get events that have non-zero length of pulse objects
		red_ev_idxs = np.where(np.array(self.red_df["n_pulses"]) > 0)[0]
		print("Got {:d} events with pulses".format(len(red_ev_idxs)))
		#loop through all events that have pulses
		for ev_idx in red_ev_idxs:
			#Create time series of a coarse rolling
			#integral of the pulses. 
			pulses = self.red_df["pulses"][ev_idx]
			new_pulses = []
			for i, p in enumerate(pulses):
				#populates an integral self attribute in the pulses. 
				p.rolling_integral(window=self.config["coarse_integral_window"])
				#find all peaks in the integral that pass thresholds
				potential_pulses = []
				pass_integ_thresh = np.where(np.array(p.integ) > self.config["integral_threshold"])[0]
				pass_integ_thresh_idxs = np.array(p.integ_idx)[pass_integ_thresh] + p.idx_start
				pass_integ_thresh_times = pass_integ_thresh_idxs/self.config["sampling_rate"]
				#mask out glitch regions
				for k, _t in enumerate(pass_integ_thresh_times):
					keep = True
					for ign in self.config["ignore_regions"]:
						if(ign[0] <= _t <= ign[1]):
							keep = False
					if(keep):
						potential_pulses.append(pass_integ_thresh[k])

				#if no samples are outside of the masked regions,
				#just continue. 
				if(len(potential_pulses) == 0):
					continue

				#otherwise, cluster the 1D timeseries to find possibility
				#of multiple peaks that pass threshold. Require at least
				#2 half window of the coarse_integral_window between pulses
				clusters = Util.simple_1d_clustering(potential_pulses, 2)

				for clust in clusters:
					#add this full pulse to the new_pulses list, containing
					#only the waveform data that is relevant to this pulse.
					
					#add half a window to the start and end of the pulse as determined
					#by the indices that pass threshold for the integral series. 
					buffer = self.config["coarse_integral_window"]*self.config["sampling_rate"]
					#get indexes of the waveform that correspond to the start and
					#end of this region of the integral window. 
					start = int(p.integ_idx[clust[0]] + p.idx_start - buffer)
					end = int(p.integ_idx[clust[-1]] + p.idx_start + buffer)
					new_pulses.append(Pulse.Pulse(self.rq_dict, self.config, p.wav[start:end], p.ch, idx_start=start))
					fig, ax = plt.subplots()
					ax.plot(range(p.idx_start, p.idx_start + len(p.wav)), p.wav)
					ax.plot(range(start, end), new_pulses[-1].wav)
					plt.show()

					


			





	#takes in a numpy array of all waves in a file. 
	def analyze_and_subtract_baselines(self, wavs):
		#get the baseline window in samples. 
		bl_window = [int(self.config["baseline"][0]*self.config["sampling_rate"]), int(self.config["baseline"][1]*self.config["sampling_rate"])]

		#in a vectorized way, get the baselines and std values for all channels
		baselines = np.median(wavs[:,:,bl_window[0]:bl_window[1]], axis=2)
		stds = np.std(wavs[:,:,bl_window[0]:bl_window[1]], axis=2)
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





			
	
		
