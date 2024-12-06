import yaml 
import os
import pandas as pd 
import numpy as np
from scipy.signal import find_peaks
import pickle
from Utilities import get_asic_and_ch, get_unique_id, get_channel_type, get_channel_pos, is_channel_strip, ADC_to_ENC
import CryoAsicFile
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

		#The output starts as a dictionary, can later be saved as a pandas dataframe. 
		#The keys are the reduced quantities, and the values are lists of the reduced quantities
		#where each element of the list is a "row" or event. For example, 
		#self.reduced_df["x"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10...] which can then be
		#converted using pd.DataFrame.from_dict(self.reduced_df) which are often easier
		#for analysis notebooks as you can mask events using boolean masks. But it is computationally
		#more expensive to append to pandas dataframes than it is to append to lists. 
		self.reduced_df = {} 
		self.initialize_reduced_df() #populates the reduced_df with the keys from the reduced quantities dictionary

		self.waveform_df = None #this is the waveform df imported by the input files. 

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

	def initialize_reduced_df(self):
		self.reduced_df = {} 
		#initialize the reduced_df with the keys from the reduced quantities dictionary
		if(self.rq_dict is None):
			print("Reduced quantities dictionary is empty")
			print("Trying to populate it now")
			self.load_rq_dict()
			if(self.rq_dict is None):
				print("Failed to populate the reduced quantities dictionary")
			return

		#get an empty event, which is a dictionary with the keys of the reduced quantities
		empty_event = self.get_empty_event()
		#populate the reduced_df with the keys from the reduced quantities dictionary
		for key in empty_event:

			# Organizing the sub channel_rqs sub structre
			if key in self.rq_dict["channel_rqs"]:
				self.reduced_df[key] = {}
				for key2 in empty_event[key]:
					self.reduced_df[key][key2] = []
			else:
				self.reduced_df[key] = []

		#done

	#returns an empty, initialized event where each key's element
	#can be appended to the reduced_df keys of the same name. 
	def get_empty_event(self):
		
		event = {}

		event["pulse"] = [] # Adding in a temporary pulse key to store out infomration while we work on making the pulse class

		for key in self.rq_dict["global"]:
			event[key] = self.rq_dict["global"][key] #initialize to the default value specified in the yaml file. 

		# Note we implictly skip dummy channels by only looping over the channel map

		for key in self.rq_dict["channel_rqs"]:
			event[key] = {}
			for asic in self.chmap:
				for xch in self.chmap[asic]["xstrips"]:
					chid = get_unique_id(asic, xch)
					event[key]["ch{:d}".format(chid)] = []
			
				for ych in self.chmap[asic]["ystrips"]:
					chid = get_unique_id(asic, ych)
					event[key]["ch{:d}".format(chid)] = []
		return event


	#The path is the full path of output 
	#The filename is the name of the file you want to save with no extensions. 
	#It checks if the path exists and creates it if possible. 
	def save_reduced_df(self, path, filename):
		if(path[-1] != '/'):
			path += '/'
		
		if(os.path.exists(path) == False):
			os.makedirs(path)
		
		#first convert to dataframe
		if(isinstance(self.reduced_df, dict)):
			df = pd.DataFrame.from_dict(self.reduced_df)
			pickle.dump([df], open(path+filename+".p", 'wb'))
		else:
			print("Somehow the self.reduced_df became something other than a dict.")
			print("Write some handling code in save_reduced_df to handle this")
			pickle.dump([self.reduced_df], open(path+filename+".p", 'wb'))

	

	def reduce_data(self, Skip_Baseline=False):

		#There may be an infinite amount of data files input to this reduction code. 
		#Instead of loading all of them and combining into a big waveform_df, we will
		#load each one, reduce each one, build up a big reduced_df that is a culmination
		#of all of the waveform_df files. Two key elements of the reduced_df are the
		#filename and evidx within that filename, used to re-index events to their origin. 

		# Glenn's Note: I disagree slightly here. I think there can be a theoretically infinite
		# number of files handed here, but as long as we maintain a good naming scheme with data
		# files, then we don't need to save the full file name, which I think will be clunky to 
		# read, and hard to mask on as well. Instead, we can maintain out current data naming
		# scheme which ends each file with "file_##.dat" and reference the number of that file.
		# The name of the file will be savd in the name of the reduced df file so all information
		# is preserved in minimal and easily parsable way

		for infile in self.input_files:
			print("Reducing file {}".format(infile))

			if infile.split('.')[-1] == "dat":

				# If binary file is given it will automatically reduce it to necessary pickle file
				print('Data Reduction was given a binary file - Converting to unreduced df')
				cf = CryoAsicFile.CryoAsicFile(infile, self.configfile_or_dict)
				cf.load_raw_data()
				cf.group_into_pandas()
				outfilename = infile.split('.')[0] + infile.split('.')[1] + '.p'
				cf.pickle_dump_waveform_df(outfilename)

				self.waveform_df = pickle.load(outfilename, 'rb')[0]

			elif infile.split('.')[-1] == 'p':
				self.waveform_df = pickle.load(open(infile, 'rb'))[0]

			else:
				print('Unrecognized file type .{0} given to data reducer. Please check file paths and try again.'.format(infile.split('.')[-1]))
				return

			file_num = (((infile.split('/')[-1]).split('_')[-1]).split('.')[0])[4:]

			self.initialize_reduced_df()

			## Looping over each event in the file to add paramters to 
			for i, row in self.waveform_df.iterrows():
				if(i % 500 == 0): print("On event {:d} of {:d}".format(i, len(self.waveform_df.index)))
				
				#do all of your analysis on the event ("row")
				
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


	# Put this in DataReduction file because I thought it'd be short and sweet, but might be worth moving to its own file for consistancy later
	def fill_in_baselines(self, row):

		for ch in row["Channels"]:

			# Note we continue to skip dummy channels
			if not is_channel_strip(self.chmap, ch): continue
			
			wvfm = row["Data"][ch]
			bl_window = wvfm[self.config["baseline"][0]*self.config["sampling_rate"]: self.config["baseline"][1]*self.config["sampling_rate"]]
			self.reduced_df["baseline_noise"]["ch{:d}".format(ch)].append(ADC_to_ENC(np.std(bl_window)))
			# Ignoring full_window_noise for now since will require cutting out pulses - so need to get pulse finder workng first
			#self.reduced_df["full_window_noise"]["ch{:d}".format(ch)].append(ADC_to_ENC(np.std(wvfm))) 
			self.reduced_df["baseline"]["ch{:d}".format(ch)].append(np.asarray(ADC_to_ENC(bl_window)))
			self.reduced_df["baseline_shift"]["ch{:d}".format(ch)].append(ADC_to_ENC(np.mean(bl_window)))


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
				
				if not (any(window[0] <= p <= window[1] for window in self.config["ignore_regions"])):
					pulses.append(p)

			for i, peak in enumerate(pulses):

				## Taking a convention where each pulse will be listed as "Pulse {event}-{pulse number within the event}"
				pulse_df["Pulse {0}-{1}".format(event, i)] = Pulse.Pulse(self.config, self.rq_dict["pulse"], wvfm, i, params).d
				(pulse_df["Pulse {0}-{1}".format(event, i)])["channel"] = ch

		return pulse_df





			
	
		
