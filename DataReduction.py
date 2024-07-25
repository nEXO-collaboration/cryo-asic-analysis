import yaml 
import os
import pandas as pd 
import pickle


class DataReduction:
	#Config is the "analysis config" file in configs, or a dictionary
	#that has been edited in the notebook (either filepath or dictionary of the yaml file)
	#The input_files are actualy filenames of what you want to reduce. For example, a list
	#from glob that selects all files with gain 6 and 1.2 pt from some directory. Full path expected.
	
	#The input_files can be a list with a single file, in case
	#you have already combined your input files into one dataframed pickle file. 
	#Otherwise, it will automatically attempt to combine the input files into a single file.
	#This could be a place you want to change... say put that combining routine in the utilities.py 
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

		#start with globals. 
		for key in self.rq_dict["global"]:
			self.reduced_df[key] = self.rq_dict["global"][key] #values in the dict are the initialization state. 

		#now do channel-level quantities for each channel. 
		#get all of the channel IDs
		for ch in self.chmap:
			for key in self.rq_dict["channel_rqs"]:
				self.reduced_df["ch{:d}_".format(ch) + key] = self.rq_dict["channel_rqs"][key]

		#done

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

	
	def load_input_data(self):
		#load the input data
		#If the input_files is a single file, just load it. 
		if(len(self.input_files) == 1):
			self.waveform_df = pickle.load(open(self.input_files[0], 'rb'))[0]
		
		#otherwise, combine the files. 
		else:
			#combine the dataframes from many files. 
			full_df = None
			for f in self.input_files:
				temp_df = pickle.load(open(f, "rb"))[0]
				if(full_df is None):
					full_df = temp_df
				else:
					full_df = pd.concat([full_df, temp_df], ignore_index=True)
			self.waveform_df = full_df
		