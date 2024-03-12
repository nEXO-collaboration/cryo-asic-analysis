import sys
import os
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd


#for progress bars, it depends whether one is working in
#a jupyter notebook or on a script. if you are getting missing
#library errors, please do the following lines to install. 
#pip install tqdm ipywidgets; 
#jupyter nbextension enable --py widgetsnbextension;
from IPython import get_ipython
def isnotebook():
	try:
		shell = get_ipython().__class__.__name__
		if shell == 'ZMQInteractiveShell':
			return True   # Jupyter notebook or qtconsole
		elif shell == 'TerminalInteractiveShell':
			return False  # Terminal running IPython
		else:
			return False  # Other type (?)
	except NameError:
		return False      # Probably standard Python interpreter
if(isnotebook()):
	from tqdm.notebook import tqdm
else:
	from tqdm import tqdm


class CryoAsicFile:
	#filename - path/to/datafile.dat, an output datafile from CRYOASIC software
	#channel_map_fn - path/to/channelmap.txt same format as the channel maps from Struck parser,
	#except that "slot" is replaced by "tile" (indexing the tile number)
	def __init__(self, filename, channel_map_fn, tile_map_fn):
		self.filename = filename #path to file
		self.nevents = None #number of events in the file
		self.waveform_df = None 
		self.channel_map_fn = channel_map_fn #filename for channel map 
		self.tile_map_fn = tile_map_fn #filename for tile position map
		self.events = [] #events[event_no][channel] = [volts, volts, volts...]
		self.scopes = [] #scopes[event_no]

		#parse the channel map file right away, flag any errors before trying to load data
		#structure: pandas dataframe with columns "Tile", "Channel", "Type", "LocalPosition"
		self.channel_map = self.parse_channel_map()
		#structure: pandas dataframe with columns "Tile", "X", "Y" (positions of center of tile in mm)
		self.tile_map = self.parse_tile_map()
		self.mapping_errors = self.check_mapping_errors() #checks to make sure values in the channel and tile map make sense and are 1:1


		self.p = 6 #mm pitch between strips

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
				if ((file_header[1]&0xff000000)>>24)==1: #image packet only, 2 mean scope data
					if (numberOfFrames == 0): 
						allFrames = [newPayload.copy()]
					else:
						newFrame  = [newPayload.copy()]
						if numberOfFrames >= nskip:
							allFrames = np.append(allFrames, newFrame, axis = 0)
					numberOfFrames = numberOfFrames + 1 
					previousSize = file_header

				if (numberOfFrames%1000==0):
					print("Read %d events from CRYO ASIC file" % numberOfFrames)
				if (numberOfFrames>nevents):
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
				looper = tqdm(range(numberOfFrames), desc='descrambling event data')
			else:
				if(nevents >= numberOfFrames):
					looper = tqdm(range(numberOfFrames), desc='descrambling event data')
				else:
					looper = tqdm(range(int(nevents)), desc='descrambling event data')

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

	def load_scope_data(self, nevents=None):
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
				newPayload = np.fromfile(f, dtype='uint16', count=payloadSize*2) #(frame size splited by four to read 32 bit 
				#save only serial data frames
				if ((file_header[1]&0xff000000)>>24)==2: #image packet only, 2 mean scope data
					if (numberOfFrames == 0):
						allFrames = [newPayload.copy()]
					else:
						newFrame  = [newPayload.copy()]
						allFrames = np.append(allFrames, newFrame, axis = 0)
					numberOfFrames = numberOfFrames + 1 
					previousSize = file_header

				if (numberOfFrames%1000==0):
					print("Read %d scope events from CRYO ASIC file" % numberOfFrames)

			except Exception: 
				e = sys.exc_info()[0]
				print ("Message\n", e)
				print ('size', file_header, 'previous size', previousSize)
				print("numberOfFrames read: " ,numberOfFrames)
		


		print("Finished reading raw binary scope data")
		self.scopes = allFrames
		#this block descrambles the format to a structured list of events
		#like event[number][ch] = [...,...,..., waveform]


	#this is distinct from load_raw_data in that it references
	#the channel map that has been configured in this class, and could
	#throw errors if that channel map doesn't match the raw data format. 
	#it also then stores into pandas dataframe structure, which takes some computing time.
	def group_into_pandas(self):
		#clear the present waveform df
		self.waveform_df = None
		self.waveform_df = pd.DataFrame(columns=['Channels','Timestamp','Data','ChannelTypes','ChannelPositions', 'Scope'])

		#check that data has been loaded in from binary
		if(len(self.events) == 0):
			print("You have not yet run load_raw_data() to load data from the binary file.")
			print("Please do that before grouping into pandas dataframe")
			return 

		#check that maps have been loaded in correctly
		if(self.channel_map is None):
			print("Did not find channel map, please re-load channel map or check the config file")
		if(self.tile_map is None):
			print("Did not find tile map, please re-load tile map or check the config file")


		#the match between Tile map and channel map is checked prior to data loading
		#(to save time in case of large data files). Check the line in the __init__ of this class.

		#start parsing events.
		looper = tqdm(self.events, desc='Adding waveforms to pandas dataframe') 
		for ev in looper:
			ev_ser = pd.Series()
			channels = []
			channel_types = []
			channel_positions = []
			waves = []
			for ch, wave in enumerate(ev):
				if(ch not in list(self.channel_map["Channel"])):
					print("Could not find channel " + str(ch) + " in channel map, but did find it in event data")
					continue
				#presently, there are two tiles worth of channels per asic. 
				#The software and electronics are not anywhere near having
				#multi-ASIC readout, so for the moment I will leave the channel ID
				#structure for a later time. Channel number will just be 1-63
				channels.append(ch) 
				typ = list(self.channel_map[self.channel_map["Channel"] == ch]["Type"])[0]
				if('x' in typ.lower()):
					channel_types.append('x')
					typ = 'x'
				else:
					channel_types.append('y')
					typ = 'y'
				waves.append(wave)

				#Here, we want to add the global position of the strip. Find the definitions
				#of this, and how its parsed from the maps, in the following function
				tile_id = int(self.channel_map[self.channel_map["Channel"] == ch]["Tile"])
				channel_positions.append(self.get_ch_position(tile_id, ch, typ))

			ev_ser["Timestamp"] = None #trying to figure out where this lives in the raw data at the moment...
			ev_ser["Channels"] = channels
			ev_ser["ChannelTypes"] = channel_types
			ev_ser["ChannelPositions"] = channel_positions
			ev_ser["Data"] = waves 
			self.waveform_df = self.waveform_df.append(ev_ser, ignore_index=True)
		if(len(self.scopes)) > 0:
			self.waveform_df['Scope'] = self.scopes.tolist()



	def save_to_hdf5(self, outfile=None):
		if(outfile is None):
			#just use the input filename but with a tag at the end (presently no tag), and h5 suffix
			#infile suffix will always be ".dat", so assume that and modify the end of filepath
			outfile = self.filename[:-4] + ".h5"

		print("Saving dataframe to file: " + outfile)
		self.waveform_df.to_hdf(outfile, key='raw')



	def get_ch_position(self, tile_id, ch, typ):
		tile_coord_series = self.tile_map[self.tile_map["Tile"] == tile_id]
		tile_center = np.array([float(tile_coord_series['X']), float(tile_coord_series['Y'])])

		strip_local_pos = float(self.channel_map[self.channel_map["Channel"] == ch]["LocalPosition"])
		#this local position is an integer multiple number of strips from center. For 32 channels, 16
		#strips per side, this goes -8,-7,-6,-5,-4,-3,-2,-1,1,2,3,... note the lack of 0. The -1 strip
		#is 0.5*pitch to the left of center (in x direction). Hence, the 0.5 below. 
		direc = np.sign(strip_local_pos) 
		if(typ == 'x'):
			strip_center = np.array([direc*(np.abs(strip_local_pos) - 0.5)*self.p, 0])
		else:
			strip_center = np.array([0, direc*(np.abs(strip_local_pos) - 0.5)*self.p])

		strip_global_center = strip_center + tile_center
		return strip_global_center

		

	def get_number_of_events(self):
		return len(self.events)


	#NOTE: there are both local and global positions
	#found in the channel map configuration file. 
	def parse_channel_map(self):

		#check if channel map file exists
		if(os.path.isfile(self.channel_map_fn) == False):
			print("Cant find the channel map file: " + str(self.channel_map_fn))
			return None

		#the header=9 sets the 9th line in the file to be the column names
		#for the dataframe
		chmap = pd.read_csv(self.channel_map_fn, sep=',', header=11)
		correct_column_names = ["Tile", "Channel", "Type", "LocalPosition"]
		#check that there aren't typos in the file's column names
		error_found = False
		for k in correct_column_names:
			if(k not in chmap.columns):
				print("Typo in channel map, didn't find column : " + k)
				error_found = True

		if(error_found):
			print("Please have the columns match these values:")
			print(correct_column_names)
			print("instead, we read line: ", end='')
			print(chmap.columns)
			return None


		#convert columns from string to their expected types
		chmap = chmap.astype({"Tile": "int32", "Channel": "int32", "Type": "string", "LocalPosition": "float"})
		return chmap

	#NOTE: there are both local and global positions
	#found in the channel map configuration file. 
	def parse_tile_map(self):

		#check if channel map file exists
		if(os.path.isfile(self.tile_map_fn) == False):
			print("Cant find the tile map file: " + str(self.tile_map_fn))
			return None

		#the header=9 sets the 9th line in the file to be the column names
		#for the dataframe
		chmap = pd.read_csv(self.tile_map_fn, sep=',', header=8)
		correct_column_names = ["Tile", "X", "Y"]
		#check that there aren't typos in the file's column names
		error_found = False
		for k in correct_column_names:
			if(k not in chmap.columns):
				print("Typo in tile map, didn't find column : " + k)
				error_found = True

		if(error_found):
			print("Please have the columns match these values:")
			print(correct_column_names)
			print("instead, we read line: ", end='')
			print(chmap.columns)
			return None


		#convert columns from string to their expected types
		chmap = chmap.astype({"Tile": "int32", "X":"float", "Y":"float"})
		return chmap

	#returns a list of errors in the map files. 
	#returns empty list if all good. 
	#Values:
	# 1 if Tile IDs aren't 1:1 between channel and tile map
	def check_mapping_errors(self):
		errors = []
		tiles_in_chmap = sorted(set(list(self.channel_map["Tile"])))
		tiles_in_tilemap = sorted(set(list(self.tile_map["Tile"])))
		if(tiles_in_chmap != tiles_in_tilemap):
			print("Tile IDs in tile map do not match those of channel map exactly:")
			print("Chmap tile IDs:" , end = ' ')
			print(tiles_in_chmap)
			print("Tile map tile IDs:" , end = ' ')
			print(tiles_in_tilemap)
			errors.append(1)

		return errors





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

