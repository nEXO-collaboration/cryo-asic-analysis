import yaml 
import os
import pandas as pd 
import pickle
from Utilities import get_asic_and_ch, get_unique_id, get_channel_type, get_channel_pos, is_channel_strip


class DataReduction:
	#Config is the "analysis config" file in configs, or a dictionary
	#that has been edited in the notebook (either filepath or dictionary of the yaml file)
	#The input_files is a list of filenames of what you want to reduce. For example, a list
	#from glob that selects all files with gain 6 and 1.2 pt from some directory. Full path expected.
	
	def __init__(self, config, input_files):

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
			self.reduced_df[key] = []

		#done

	#returns an empty, initialized event where each key's element
	#can be appended to the reduced_df keys of the same name. 
	def get_empty_event(self):
		event = {}
		for key in self.rq_dict["global"]:
			event[key] = self.rq_dict["global"][key] #initialize to the default value specified in the yaml file. 
		
		#Here is a place where you 
		#may want to decide to ignore all dummy channels, as you may
		#never want to really analyze the dummy channels within the data
		#reduction code framework. 
		for asic in self.chmap:
			for ch in self.chmap:
				chid = get_unique_id(asic, ch)
				for key in self.rq_dict["channel_rqs"]:
					event["ch{:d}_".format(chid) + key] = self.rq_dict["channel_rqs"][key] #initialize to the default value specified in the yaml file.

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

	

	def reduce_data(self):

		#There may be an infinite amount of data files input to this reduction code. 
		#Instead of loading all of them and combining into a big waveform_df, we will
		#load each one, reduce each one, build up a big reduced_df that is a culmination
		#of all of the waveform_df files. Two key elements of the reduced_df are the
		#filename and evidx within that filename, used to re-index events to their origin. 
		for infile in self.input_files:
			print("Reducing file {}".format(infile))
			self.waveform_df = pickle.load(open(infile, 'rb'))[0]

			for i, row in self.waveform_df.iterrows():
				if(i % 500 == 0): print("On event {:d} of {:d}".format(i, len(self.waveform_df.index)))
				event_output = self.get_empty_event()
				
				#do all of your analysis on the event ("row")
				event_output["filename"] = infile
				event_output["evidx"] = i

				#for now I am leaving all analysis steps empty and going to save
				#all default values to the reduced_df. This is the final step. 

				#append the event to the reduced_df
				for key in event_output:
					self.reduced_df[key].append(event_output[key])



			
	
		
