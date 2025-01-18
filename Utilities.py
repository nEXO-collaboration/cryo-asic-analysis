import math 
import numpy as np 


#this is a set of importable utilities that is common to many of the classes in this project. 

#return the unique channel ID given an asic number 
#and a channel number local to that asic
def get_unique_id(asic, ch):
	return asic*64 + ch

#inverse of the above, return the asic and channel number
#given a unique channel ID
def get_asic_and_ch(ch):
	asic = math.floor(ch/64)
	ch = ch % 64
	return asic, ch


#get the channel type from the channel number
def get_channel_type(chmap, ch):
	if(chmap is None):
		print("Channel map didn't properly load")
		return None
	local_ch = ch % 64 #the channel number on the asic level. 
	asic = math.floor(ch/64) # the asic ID that this ch corresponds to. 
	
	
	if(asic in chmap):
		if(local_ch in chmap[asic]["xstrips"]):
			return 'x'
		elif(local_ch in chmap[asic]["ystrips"]):
			return 'y'
		else:
			return 'dummy'
		
	else:
		print("Asic {:d} not found in the configuration file channel map".format(asic))
		return None
	
#returns the global position of the channel in the TPC
#using knowledge of the tile position. If it is a dummy, 
#return 0, 0 
def get_channel_pos(chmap, ch):
	if(chmap is None):
		print("Channel map didn't properly load")
		return None

	local_ch = ch % 64 #the channel number on the asic level. 
	asic = math.floor(ch/64) # the asic ID that this ch corresponds to. 
	tile_pos = chmap[asic]["tile_pos"]
	pitch = chmap[asic]["strip_pitch"] #in mm

	if(asic in chmap):
		if(local_ch in chmap[asic]["xstrips"]):
			local_pos = float(chmap[asic]["xstrips"][local_ch])
			return (tile_pos[0], tile_pos[1] + np.sign(local_pos)*(np.abs(local_pos) - 0.5)*pitch)
		elif(local_ch in chmap[asic]["ystrips"]):
			local_pos = float(chmap[asic]["ystrips"][local_ch])
			return (tile_pos[0] + np.sign(local_pos)*(np.abs(local_pos) - 0.5)*pitch, tile_pos[1])
		else:
			return tile_pos #this is a dummy capacitor
	else:
		print("Asic {:d} not found in the configuration file channel map".format(asic))
		return None
	

def is_channel_strip(chmap, ch):
	result = get_channel_type(chmap, ch)
	if(result == "dummy"):
		return False
	else:
		return True
	
#does the spatial analysis to find the N adjacent
#channels on EACH side of ch and returns a list of ch numbers
def get_adjacent_channels(chmap, ch, N):
	#needs to be a strip to have position
	if(is_channel_strip(chmap, ch) == False):
		return [] 
	
	local_ch = ch % 64 #the channel number on the asic level. 
	asic = math.floor(ch/64) # the asic ID that this ch corresponds to. 
	ch_type = get_channel_type(chmap, ch)
	
	#get all channel IDs for that strip type
	if(ch_type == 'x'):
		all_chs = chmap[asic]["xstrips"]
	elif(ch_type == 'y'):
		all_chs = chmap[asic]["ystrips"]

	#get the local position of the channel
	local_pos = float(all_chs[local_ch])
	#get a list of the differences between the local position
	#and the local positions of all the other channels
	distances = [(ch, loc, loc - local_pos) for ch, loc in all_chs.items() if ch != local_ch]
	#get the indices of the N closest channels in the positive
	#and negative direction
	above = [(ch, loc) for ch, loc, diff in distances if diff > 0]
	below = [(ch, loc) for ch, loc, diff in distances if diff < 0]
	# Sort by absolute distance (ascending)
	above.sort(key=lambda x: x[1] - local_pos)
	below.sort(key=lambda x: local_pos - x[1])
	# Get the N closest above and below
	closest_above = above[:N]
	closest_below = below[:N]
	#get the channel numbers of the N closest channels
	adj = [get_unique_id(asic, chtup[0]) for chtup in closest_above + closest_below]
	return adj


def simple_1d_clustering(data, cluster_spacing):
	"""
	Find clusters in a 1D list of integers based on a specified cluster spacing.

	Parameters:
	- data: list of int or floats
		The 1D list of values to find clusters
	- cluster_spacing: float
		The maximum spacing allowed between consecutive numbers in a cluster.

	Returns:
	- clusters: list of lists
		A list of clusters, where each cluster is a list of the values. 
	- indices: list of lists
		A list of clusters, where each cluster is a list of the indices of the values in the original data.
	"""
	if(data is None or len(data) == 0):
		return []  # Handle empty input

	# Sort the data along with their original indices
	indexed_data = sorted((value, idx) for idx, value in enumerate(data))

	# Initialize clusters
	clusters = [[indexed_data[0]]]  # Start with the first element in a cluster

	for value, idx in indexed_data[1:]:
		# Check if the current value is within the cluster_spacing of the last cluster
		if value - clusters[-1][-1][0] <= cluster_spacing:
			clusters[-1].append((value, idx))  # Add to the current cluster
		else:
			clusters.append([(value, idx)])  # Start a new cluster

	return clusters

#FYI - for other calculations. 
#(x1, x1.5, x3, x6) is {1: 9.6, 1.5: 14.3, 3:28.6, 6:57.2} mV/fC
@np.vectorize
def ADC_to_ENC(ADC, Gain=6, pt=1.2):

	Gain = str(Gain)
	pt = str(pt)

	V_Max = {}
	V_Max["1"] = {"0.6": 1.5798, "1.2": 1.5751, "2.4": 1.5785, "3.6": 1.5769} # ASIC Voltage Saturation According to Aldo
	V_Max["1.5"] = {"0.6": 1.5773, "1.2": 1.5731, "2.4": 1.5783, "3.6": 1.5763}	
	V_Max["3"] = {"0.6": 1.5706, "1.2": 1.5679, "2.4": 1.5735, "3.6": 1.5741}
	V_Max["6"] = {"0.6": 1.5604, "1.2": 1.5609, "2.4": 1.5659, "3.6": 1.5703}
	
	Q_Max = {"1": 150e-15, "1.5": 100e-15, "3": 50e-15, "6": 25e-15} #Maximum charge range of the ASIC in Coloumbs

	ENC = ADC*(1.2/2**12)*(Q_Max[Gain]/V_Max[Gain][pt])/1.6e-19
	return ENC

@np.vectorize
def ENC_to_ADC(ENC, Gain=6, pt=1.2):
	Gain = str(Gain)
	pt = str(pt)

	V_Max = {}
	V_Max["1"] = {"0.6": 1.5798, "1.2": 1.5751, "2.4": 1.5785, "3.6": 1.5769} # ASIC Voltage Saturation According to Aldo
	V_Max["1.5"] = {"0.6": 1.5773, "1.2": 1.5731, "2.4": 1.5783, "3.6": 1.5763}	
	V_Max["3"] = {"0.6": 1.5706, "1.2": 1.5679, "2.4": 1.5735, "3.6": 1.5741}
	V_Max["6"] = {"0.6": 1.5604, "1.2": 1.5609, "2.4": 1.5659, "3.6": 1.5703}
	
	Q_Max = {"1": 150e-15, "1.5": 100e-15, "3": 50e-15, "6": 25e-15} #Maximum charge range of the ASIC in Coloumbs

	ADC = ENC*1.6e-19*(V_Max[Gain][pt])/(1.2/2**12)/(Q_Max[Gain])
	return ADC
