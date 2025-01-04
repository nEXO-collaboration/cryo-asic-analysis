import yaml 
import os
import pandas as pd 
import numpy as np
from scipy.signal import find_peaks
import pickle
import Utilities as Util
import Pulse
import Cluster
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
			print("Cant find the channel map file: " + str(self.config["chmap"]))
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
		if(isinstance(self.red_df, dict)):
			df = pd.DataFrame.from_dict(self.red_df)
			pickle.dump([df], open(path+filename, 'wb'))
		else:
			print("Somehow the self.red_df became something other than a dict.")
			print("Write some handling code in save_reduced_df to handle this")
			pickle.dump([self.red_df], open(path+filename, 'wb'))

	
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
		red_df["n_pulses"] = np.zeros(len(self.waveform_df.index))
		print("Initializing pulses for events with any sample above positive threshold of {:0.2f} sigma".format(self.config["low_threshold"]))
		#do a np.where to find where any channel number is above threshold
		mask = None
		for chidx in range(len(wavs[0])):
			ch = chidx_map[chidx]
			if(not Util.is_channel_strip(self.chmap, ch)):
				continue
			if(ch in self.config["dead_channels"]):
				continue 
			maxs = red_df["ch{:d} max".format(ch)]
			threshs = self.config["low_threshold"]*red_df["ch{:d} baseline_std".format(ch)]
			if(mask is None):
				mask = np.where(maxs > threshs, 1, 0)
			#if a mask already exists, I want to OR it with the new mask
			else:
				mask = np.where(maxs > threshs, 1, 0) | mask
		

		#get red_ev indices that passed mask
		red_ev_idxs = np.where(mask == 1)[0]
		for ev_idx in red_ev_idxs:
			temp_pulse_channels = [] #list of pulse objects that will be edited 
			#loop through all strip channels and initialize pulse objects
			#for each channel that has a pulse above a wide acceptance threshold
			for chidx in range(len(wavs[ev_idx])):
				ch = chidx_map[chidx]
				if(not Util.is_channel_strip(self.chmap, ch)):
					continue
				if(ch in self.config["dead_channels"]):
					continue 
				
				if(red_df["ch{:d} max".format(ch)][ev_idx] > self.config["low_threshold"]*red_df["ch{:d} baseline_std".format(ch)][ev_idx]):
					temp_pulse_channels.append(ch)

			#we want to initialize pulse objects
			#for channels spatially adjacent 
			set_adjacent_chs = [] #to avoid double counting
			for ch in temp_pulse_channels:
				#initialize the spatially adjacent pulses
				adj_chs = Util.get_adjacent_channels(self.chmap, ch, self.config["adjacency"])
				for adj in adj_chs:
					if(adj in self.config["dead_channels"]):
						continue
					set_adjacent_chs.append(adj)
			
			#make sets to remove duplicate channels and duplicate pulses
			adj_chs = list(set(set_adjacent_chs))
			all_chs = temp_pulse_channels + adj_chs
			all_chs = list(set(all_chs))
			for adj in all_chs:
				chidx = np.where(chidx_map == adj)[0][0]
				red_df["pulses"][ev_idx].append(Pulse.Pulse(self.rq_dict["pulse"], self.config, wavs[ev_idx][chidx], adj))

			red_df["n_pulses"][ev_idx] = len(red_df["pulses"][ev_idx])

		self.red_df = red_df #store the reduced dictionary for this file.
		#later will be concatenated manually to the self.reduced_df.


	#does waveform level analysis on pulse objects, 
	#vetting them to either remove if they are meaningless
	#or calculate reduced quantities like energy and such. 
	def process_pulses(self):
		#get events that have non-zero length of pulse objects
		red_ev_idxs = np.where(np.array(self.red_df["n_pulses"]) > 0)[0]
		print("Got {:d} events with pulses".format(len(red_ev_idxs)))

		#First, re-buffer pulses into small chunks that
		#(1) separate multiple pulses from one channel in one event waveform
		#(2) reject pulses that single data point glitches
		print("Rejecting single data point glitches and buffering pulses")
		for ev_idx in red_ev_idxs:
			pulses = self.red_df["pulses"][ev_idx]

			#first, find any pulses where a peak above threshold can be found,
			#which may not be many of the channels, as we added pulses that are spatially
			#adjacent in the previous step. They may not have a peak above threshold, but may
			#have measurable charge through integration. 
			peakfound_pulses = []

			for i, p in enumerate(pulses):
				ch = p.ch
				#get std of the baseline for this channel
				std = self.red_df["ch{:d} baseline_std".format(ch)][ev_idx]
				thresh = self.config["low_threshold"]*std
				#the waveform is in ADC and this thresh is in ENC, put the thresh in ADC
				thresh = Util.ENC_to_ADC(thresh, self.config["gain"], self.config["pt"])
				#ignore-region pulses are rejected in the p.find_peaks function. 
				temp_pulses, properties = p.find_peaks(width=self.config["pt"], thresh=thresh)
				if(len(temp_pulses) == 0):
					continue
				
				for j, tp in enumerate(temp_pulses):
					#reject single-data point glitches due to data corruption
					if(properties["widths"][j] < 1.0/self.config["sampling_rate"]):
						continue

					#isolate this pulse from the waveform and store it in a new pulse object. 
					#this is so that we can analyze the pulse in isolation.
					window = [tp - int(self.config["pulse_window"]*self.config["sampling_rate"]/2), tp + int(self.config["pulse_window"]*self.config["sampling_rate"]/2)]
					newP = Pulse.Pulse(self.rq_dict["pulse"], self.config, p.wav[window[0]:window[1]], ch, idx_start=window[0])
					peakfound_pulses.append(newP)
				
			self.red_df["pulses"][ev_idx] = peakfound_pulses
			self.red_df["n_pulses"][ev_idx] = len(peakfound_pulses)

		#Calculate all reduced quantities for the pulses
		red_ev_idxs = np.where(np.array(self.red_df["n_pulses"]) > 0)[0]
		print("Calculating reduced quantities for pulses from {:d} remaining events".format(len(red_ev_idxs)))
		for ev_idx in red_ev_idxs:
			for p in self.red_df["pulses"][ev_idx]:
				p.calculate_reduced_quantities()

		



	def process_clusters(self):
		#initialize an empty list of clusters for each event
		self.red_df["clusters"] = [[] for i in range(len(self.red_df["evidx"]))]
		#and the rest of the reduced quantities for the clusters
		self.red_df["n_clusters"] = [0]*len(self.red_df["evidx"])
		self.red_df["total_charge"] = [None]*len(self.red_df["evidx"])
		self.red_df["x"] = [None]*len(self.red_df["evidx"])
		self.red_df["y"] = [None]*len(self.red_df["evidx"])
		self.red_df["z"] = [None]*len(self.red_df["evidx"])
		self.red_df["t"] = [None]*len(self.red_df["evidx"])

		#only process events with pulses
		red_ev_idxs = np.where(np.array(self.red_df["n_pulses"]) > 0)[0]
		print("Processing clusters for {:d} events which have pulses".format(len(red_ev_idxs)))
		for ev_idx in red_ev_idxs:
			#cluster time first
			ts = [p.d["t_arrival"] for p in self.red_df["pulses"][ev_idx]]
			t_clust, t_clust_idx = Util.simple_1d_clustering(ts, self.config["clust_time_sep"])
			#within each time cluster, cluster in x and y (to see if two clusters arrive at the same time)
			for tcidx, tc in enumerate(t_clust_idx):
				#pulses in the cluster
				clust_ps = [self.red_df["pulses"][ev_idx][i] for i in tc]
				xs = []
				ys = []
				for p in clust_ps:
					if(Util.get_channel_type(self.chmap, p.ch) == 'x'):
						ys.append(Util.get_channel_pos(self.chmap, p.ch)[1])
					else:
						xs.append(Util.get_channel_pos(self.chmap, p.ch)[0])

				x_clust, x_clust_idx = Util.simple_1d_clustering(xs, self.config["clust_space_sep"])
				y_clust, y_clust_idx = Util.simple_1d_clustering(ys, self.config["clust_space_sep"])
				#handle the simple case where there is only one cluster in both x and y
				#or if there is just a single x cluster with no y cluster or vice versa
				if((len(x_clust) == 1 and len(y_clust) == 1) or (len(x_clust) + len(y_clust) == 1)):
					#initialize the cluster object
					temp_clust = Cluster.Cluster(self.rq_dict["cluster"], self.config)
					for xc in x_clust_idx:
						for i in xc:
							temp_clust.pulses.append(clust_ps[i])
					for yc in y_clust_idx:
						for i in yc:
							temp_clust.pulses.append(clust_ps[i])

					self.red_df["clusters"][ev_idx].append(temp_clust)
					self.red_df["n_clusters"][ev_idx] += 1
				else:
					#I don't quite know what to do if we get two events
					#landing on the tile at the same time. It's a degeneracy
					#built into the tile geometry. One needs to make assumptions
					#about charge sharing to be able to associate the x-y
					#values of the charge depositions. SO, for now, we will
					#just add all of the pulses to the cluster and flag it for 
					#degeneracy. 
					temp_clust = Cluster.Cluster(self.rq_dict["cluster"], self.config)
					temp_clust.pulses = clust_ps
					temp_clust.d["degeneracy"] = True
					self.red_df["clusters"][ev_idx].append(temp_clust)
					self.red_df["n_clusters"][ev_idx] += 1

		#calculate reduced quantities for the clusters
		red_ev_idxs = np.where(np.array(self.red_df["n_clusters"]) > 0)[0]
		print("Calculating reduced quantities of clusters for {:d} events".format(len(red_ev_idxs)))
		for ev_idx in red_ev_idxs:
			for c in self.red_df["clusters"][ev_idx]:
				c.calculate_reduced_quantities()
			

	#at this stage, clusters and pulses should have been populated
	#into the reduced df. Now we will calculate the remaining global quantities
	#that are associated with the event as a whole.
	def process_globals(self):
		#process events with clusters
		red_ev_idxs = np.where(np.array(self.red_df["n_clusters"]) > 0)[0]
		print("Processing global quantities for {:d} events which have clusters".format(len(red_ev_idxs)))
		for ev_idx in red_ev_idxs:
			#total charge is the sum of all positive integrals of pulses
			#in the event. 
			total_charge = 0
			max_q_cluster = None #get cluster with the max Q
			max_q = 0
			for c in self.red_df["clusters"][ev_idx]:
				total_charge += c.d["q"]
				if(c.d["q"] > max_q):
					max_q = c.d["q"]
					max_q_cluster = c

			self.red_df["total_charge"][ev_idx] = total_charge

			#get the position and time of the max q cluster
			if(max_q_cluster is not None):
				self.red_df["x"][ev_idx] = max_q_cluster.d["x"]
				self.red_df["y"][ev_idx] = max_q_cluster.d["y"]
				self.red_df["t"][ev_idx] = max_q_cluster.d["t_arrival"]
			else:
				self.red_df["x"][ev_idx] = None
				self.red_df["y"][ev_idx] = None
				self.red_df["t"][ev_idx] = None


	#during processing, the Pulse and Cluster objects
	#are stored in lists within the reduced dictionary. They
	#also contain waveform data that may be large (but is often ~100 samples). 
	#This will remove those objects in prep for storage and transfer of the reduced
	#data into a format that does not rely on the Pulse and Cluster classes. So,
	#it just extracts their reduced quantity dictionaries. 
	def dictify_objects(self):
		#process events with clusters
		print("Dictifying the Pulse and Cluster objects in prep for data transfer")
		for ev_idx in range(len(self.red_df["evidx"])):
			#process pulses
			pulses = [] #list of dictionaries
			for p in self.red_df["pulses"][ev_idx]:
				pulses.append(p.d)
			self.red_df["pulses"][ev_idx] = pulses

			#process clusters
			clusters = [] #list of dictionaries
			for c in self.red_df["clusters"][ev_idx]:
				pulses = []
				for p in c.pulses:
					pulses.append(p.d)
				c.d["pulses"] = pulses
				clusters.append(c.d)
			self.red_df["clusters"][ev_idx] = clusters
		





			
