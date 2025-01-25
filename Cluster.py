import numpy as np 
import matplotlib.pyplot as plt
import Utilities as Util
import os
import yaml
import time

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
		self.load_channel_map()

		#form 1D distributions of charge verses distance,
		#where charge is defined by the regular (not negative or positive)
		#integral of the channel. Separate X and Y channels. 
		qxs = []
		qys = []
		xs = []
		ys = []
		for pulse in self.pulses:
			if(Util.get_channel_type(self.chmap, pulse.ch) == "y"):
				qxs.append(pulse.d["integral"])
				xs.append(Util.get_channel_pos(self.chmap, pulse.ch)[0])
			else:
				qys.append(pulse.d["integral"])
				ys.append(Util.get_channel_pos(self.chmap, pulse.ch)[1])

		#calculate the charge-weighted average of the x and y positions
		qw_x = None
		qw_y = None
		if(len(qxs) > 0):
			qw_x = np.sum(np.array(xs)*np.array(qxs))/np.sum(qxs)
		if(len(qys) > 0):
			qw_y = np.sum(np.array(ys)*np.array(qys))/np.sum(qys)
		

		if(max(qxs + qys) > 5000):
			fig, ax = plt.subplots(ncols = 2)
			ax[0].scatter(xs, qxs, label="X", s=300)
			ax[1].scatter(ys, qys, label="Y", s=300)
			if(qw_x != None):
				ax[0].axvline(x=qw_x, color='r', linestyle='--', label="Charge Weighted X")
			if(qw_y != None):
				ax[1].axvline(x=qw_y, color='b', linestyle='--', label="Charge Weighted Y")
			plt.show()


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
		if(np.sum(x_qs) == 0):
			self.d["x"] = None
		else:
			self.d["x"] = np.average(x_positions, weights=x_qs)

		if(np.sum(y_qs) == 0):
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


	
