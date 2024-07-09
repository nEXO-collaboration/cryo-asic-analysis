import sys
import os
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import yaml
import pickle



class CryoAsicFile:
	#filename - path/to/datafile.dat, an output datafile from CRYOASIC software
	#the analysis config file contains path to the channel map file which has
	#all positions of all strips, identifies strips vs dummies, and has local
	#tile coordinates, as well as which ASIC identifiers are active. 
	#You can also pass in a dictionary on its own so that at notebook level,
	#you may parse many files in a loop with different configurations. So config
	#is either a filename.yml or a dictionary object. 
	def __init__(self, filename, config):
		self.filename = filename #path to file
		self.nevents = None #number of events in the file
		self.waveform_df = None
		#initialize waveform df with a consistent column headers
		self.initialized_waveform_df()
		

		self.config = None #has global analysis config dictionary contents
		self.chmap = None #has the channel and tile mappings. 
		self.load_config(config) #loads both of the above dictionaries

		self.events = [] #events[event_no][channel] = [volts, volts, volts...]

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

	def initialized_waveform_df(self):
		self.waveform_df = None
		#Channels: a list of channel numbers for this ASIC, which from the map can reconstruct
		#its position and type. 
		#Timestamp: timestamp of the event (not yet implemented)
		#Data: list of waveforms for each channel. 
		self.waveform_df = pd.DataFrame(columns=['Channels','Timestamp','Data'])

	#a lot of this function is a bit esoteric, so comments are sparse.
	#i've taken this from Dionisio, on binary
	#format saved by the rogue software controlling the asic. 
	#nevents limits the number of events loaded. 
	def load_raw_data(self, nevents=None, nskip=0, AsicID = 0):
		#this block reads in raw binary file into
		#some interesting format?
		f = open(self.filename, mode = 'rb')
		file_header = [0]
		numberOfFrames = 0
		previousSize = 0
		while (len(file_header)>0):
			try:
				# reads file header [the number of bytes to read, EVIO]
				file_header = np.fromfile(f, dtype='uint32', count=2)
				payloadSize = int(file_header[0]/4)-1 #-1 is need because size info includes the second word from the header
				newPayload = np.fromfile(f, dtype='uint32', count=payloadSize) #(frame size splited by four to read 32 bit 
				#save only serial data frames
				if (((file_header[1] & 0xff000000) >> 24) == 1): #image packet only, 2 mean scope data
					if (numberOfFrames == 0): 
						allFrames = [newPayload.copy()]
					else:
						newFrame  = [newPayload.copy()]
						if(numberOfFrames >= nskip):
							allFrames = np.append(allFrames, newFrame, axis = 0)
					numberOfFrames = numberOfFrames + 1 
					previousSize = file_header

				if ((numberOfFrames % 2) == 0):
					print("Read %d events from CRYO ASIC file" % numberOfFrames)
				if(nevents is not None):
					if (numberOfFrames > nevents):
						break

			except Exception: 
				e = sys.exc_info()[0]
				print ("Message\n", e)
				print ('size', file_header, 'previous size', previousSize)
				print("numberOfFrames read: " ,numberOfFrames)


		print("Finished reading raw binary, now descrambling the data")
		#this block descrambles the format to a structured list of events
		#like event[number][ch] = [...,...,..., waveform]
		currentRawData = [] 
		imgDesc = [] #the output of this function, descrambled raw data
		numberOfFrames = len(allFrames)
		if(numberOfFrames == 0):
			print("Something went wrong in reading binary file, got no events")
			return None
		elif(numberOfFrames == 1):
			imgDesc = np.array([self.descramble_cryo_image(bytearray(allFrames[0].tobytes()))], dtype=float) #Just descramble the 0th element of the length 1 array. 

		else:
			if(nevents is None):
				looper = range(numberOfFrames)
			else:
				if(nevents >= numberOfFrames):
					looper = range(numberOfFrames)
				else:
					looper = range(int(nevents))

			for i in looper:
				currentRawData = allFrames[i, :]
				if int(( currentRawData[0] & 0x10)>>4) != AsicID:
					continue
				if(len(imgDesc) == 0):
					imgDesc = np.array([self.descramble_cryo_image(bytearray(currentRawData.tobytes()))], dtype=float)
				else:
					temp = self.descramble_cryo_image(bytearray(currentRawData.tobytes()))
					temp = temp.astype(float, copy=False)
					imgDesc = np.concatenate((imgDesc, np.array([temp])), 0)

		self.events = imgDesc 
		print("Done loading " + str(len(self.events)) + " CRYO ASIC events")


	#this is distinct from load_raw_data in that it references
	#the channel map that has been configured in this class, and could
	#throw errors if that channel map doesn't match the raw data format. 
	#it also then stores into pandas dataframe structure, which takes some computing time.
	def group_into_pandas(self):
		#clear the present waveform df
		self.initialized_waveform_df()

		#check that data has been loaded in from binary
		if(len(self.events) == 0):
			print("You have not yet run load_raw_data() to load data from the binary file.")
			print("Please do that before grouping into pandas dataframe")
			return 

		#start parsing events.
		columns = self.waveform_df.columns
		#it is way faster to work with a dictionary and then
		#convert to dataframe in one fell swoop. 
		output_dict = {} 
		for c in columns:
			output_dict[c] = [] #event indexed

		for ev in self.events:
			channels = []
			waves = []
			for ch, wave in enumerate(ev):
				channels.append(ch + int(self.config["asic"])*64) #here we create unique Channel IDs for each asic in the system. 
				waves.append(wave)

			output_dict["Timestamp"].append(None) #trying to figure out where this lives in the raw data at the moment...
			output_dict["Channels"].append(channels)
			output_dict["Data"].append(waves)

		self.waveform_df = pd.DataFrame(output_dict) #convert to dataframe


	def pickle_dump_waveform_df(self, outfile=None):
		if(outfile is None):
			#just use the input filename but with a tag at the end (presently no tag), and h5 suffix
			#infile suffix will always be ".dat", so assume that and modify the end of filepath
			outfile = self.filename[:-4] + ".p"

		print("Saving dataframe to file: " + outfile)
		pickle.dump([self.waveform_df], open(outfile, 'wb'))


	def get_number_of_events(self):
		return len(self.events)

	#a utility function for parsing the binary format from the CRYO asic
	def descramble_cryo_image(self, rawData):
		header_length = 6 #words of header for our format
		num_ch_per_asic = 64

		if (type(rawData != 'numpy.ndarray')):
			img = np.frombuffer(rawData,dtype='uint16')

		#calculate the number of samples
		samples = int((img.shape[0] - header_length)/num_ch_per_asic)

		#these lines are directly copied and thus are not well described (not commented in original code)
		#(found in Cameras.py::_descrambleCRYO64XNImage)
		img2 = img[header_length:].reshape(samples, num_ch_per_asic) #separate header from data
		#descramble image

		imgDesc0 = np.append(img2[:,0:num_ch_per_asic:16].transpose(), img2[:,2:num_ch_per_asic:16].transpose())
		imgDesc4 = np.append(img2[:,4:num_ch_per_asic:16].transpose(), img2[:,6:num_ch_per_asic:16].transpose())
		imgDesc8 = np.append(img2[:,8:num_ch_per_asic:16].transpose(), img2[:,10:num_ch_per_asic:16].transpose())
		imgDesc12 = np.append(img2[:,12:num_ch_per_asic:16].transpose(), img2[:,14:num_ch_per_asic:16].transpose())
		imgDesc1 = np.append(img2[:,1:num_ch_per_asic:16].transpose(), img2[:,3:num_ch_per_asic:16].transpose())
		imgDesc5 = np.append(img2[:,5:num_ch_per_asic:16].transpose(), img2[:,7:num_ch_per_asic:16].transpose())
		imgDesc9 = np.append(img2[:,9:num_ch_per_asic:16].transpose(), img2[:,11:num_ch_per_asic:16].transpose())
		imgDesc13 = np.append(img2[:,13:num_ch_per_asic:16].transpose(), img2[:,15:num_ch_per_asic:16].transpose())
		imgDesc = np.append(imgDesc0,imgDesc4)
		imgDesc = np.append(imgDesc,imgDesc8)
		imgDesc = np.append(imgDesc,imgDesc12)
		imgDesc = np.append(imgDesc,imgDesc1)
		imgDesc = np.append(imgDesc,imgDesc5)
		imgDesc = np.append(imgDesc,imgDesc9)
		imgDesc = np.append(imgDesc,imgDesc13).reshape(num_ch_per_asic,samples)
		
		#descrambled image is now in uint32 numbers that are huge. 
		#there is then a bitmask applied that ANDS out a lot of those bits,
		#converting to ADC counts. 
		imgDesc = np.bitwise_and(imgDesc, 0xFFFF)
		return imgDesc

