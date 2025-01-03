import numpy as np 
import matplotlib.pyplot as plt
import Utilities as Util
import os
import yaml

class Cluster:
	#initialize the cluster with its reduced quantities which
	#should be parsed externally (by the instantiator) from a yaml file
	def __init__(self, rqs, config):
		self.rqs = rqs
		self.d = {}
		#initialize the cluster dictionary, with
		#initialze values specified in the yaml file that 
		#defines RQs. 
		for key in self.rqs:
			self.d[key] = self.rqs[key]

		self.config = config #already a dict
		self.pulses = []

		#if the channel map is needed
		self.chmap = None
		

	def load_channel_map(self):
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


	def calculate_reduced_quantities(self):
		self.d["n_pulses"] = len(self.pulses)
		self.d["pulses"] = self.pulses

		#arrival time is the average of arrival times
		arrival_times = [p.d["t_arrival"] for p in self.pulses]
		self.d["t_arrival"] = np.mean(arrival_times)
		self.d["dt"] = np.std(arrival_times)

		#total charge will presently be the sum of positive integrals
		#of pulses, represented collection charge. 
		total_charge = 0
		for pulse in self.pulses:
			total_charge += pulse.d["pos_integral"]
		#this has units of ENC*us, so we divide by the integration
		#window used in the analysis. 
		t_integration = self.config["integ_window"][1] - self.config["integ_window"][0] #us
		self.d["q"] = total_charge / t_integration

		#special case where the total charge is 0 so we can't calculate anything else
		if(self.d["q"] == 0):
			self.d["x"] = None
			self.d["y"] = None
			self.d["dx"] = None
			self.d["dy"] = None
			self.d["n_x"] = None
			self.d["n_y"] = None
			return

		#the position of the cluster will be charge centroid in 1D. 
		#for that we need to separate out the pulses into x and y
		x_positions = []
		y_positions = []
		x_qs = []
		y_qs = []
		self.load_channel_map()
		for p in self.pulses:
			if(Util.get_channel_type(self.chmap, p.ch) == "y"):
				x_positions.append(Util.get_channel_pos(self.chmap, p.ch)[0])
				x_qs.append(p.d["pos_integral"]/t_integration)
			else:
				y_positions.append(Util.get_channel_pos(self.chmap, p.ch)[1])
				y_qs.append(p.d["pos_integral"]/t_integration)

		self.d["n_x"] = len(x_positions)
		self.d["n_y"] = len(y_positions)
		if(len(x_positions) == 0):
			self.d["x"] = None
		else:
			self.d["x"] = np.average(x_positions, weights=x_qs)

		if(len(y_positions) == 0):
			self.d["y"] = None
		else:
			self.d["y"] = np.average(y_positions, weights=y_qs)

		#Eventually, we can find the uncertainty in the x/y or the size of the charge cloud
		#by using either a lookup table of channel-distributions, or fitting the spatial
		#distribution of the charge cloud and taking the width, or other methods. See
		#paper here for more details on how this may be done:
		#10.1016/j.nima.2020.164702
		#For now, we'll leave it as nones.
		self.d["dx"] = None
		self.d["dy"] = None


	def plot_cluster(self):
		fig, ax = plt.subplots()
		for pulse in self.pulses:
			wav = pulse.wav 
			ts = range(pulse.idx_start, pulse.idx_start + len(wav))
			ax.plot(ts, wav)
		plt.show()


	