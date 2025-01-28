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

		qxs = np.array(qxs)
		qys = np.array(qys)
		xs = np.array(xs)
		ys = np.array(ys)

		#Find the mean and STD of the integrals, and find any
		#Points in both distributions that lie outside of 1-sigma. 
		qxmean = np.mean(qxs)
		qxstd = np.std(qxs)
		qymean = np.mean(qys)
		qystd = np.std(qys)
		#find the pulses that are outside of 1.5-sigma
		qxpass = qxs[np.abs(qxs - qxmean) > qxstd]
		xxpass = xs[np.abs(qxs - qxmean) > qxstd]
		qypass = qys[np.abs(qys - qymean) > qystd]
		yypass = ys[np.abs(qys - qymean) > qystd]

		#find a new mean value for the integrals
		#of all points that are within the 1-sigma range. 
		#We will use that to re-calculate the "baseline" 
		#of the integral distribution. 
		qx_baseline = np.mean([qxs[i] for i in range(len(qxs)) if xs[i] not in xxpass])
		qy_baseline = np.mean([qys[i] for i in range(len(qys)) if ys[i] not in yypass])

		#baseline subtract the integrals that pass that 1-sigma threshold
		qxpass = np.array(qxpass) - qx_baseline
		qypass = np.array(qypass) - qy_baseline
		qxs = qxs - qx_baseline
		qys = qys - qy_baseline

		#combine into a single data structure
		qs = []
		for i in range(len(qxpass)):
			qs.append([xxpass[i], qxpass[i], "x"])
		for i in range(len(qypass)):
			qs.append([yypass[i], qypass[i], "y"])

		#if the list of passing pulses is empty, 
		#then we should not consider this cluster.
		if(len(qs) == 0):
			return None #this signals to the DataReduction class that it should be removed. 
		
		#find the maximum positive charge in the cluster
		max_charge = sorted(qs, key=lambda x: x[1], reverse=True)[0]
		#if there is no positive charge in the cluster
		#then we should essentially no longer consider this 
		#a cluster... 
		if(max_charge[1] < 0):
			return None #this signals to the DataReduction class that it should be removed. 

		#find the absolute max for each strip type. This will
		#return negative values if the absmax is negative. 
		absmax_x = None
		absmax_y = None
		for q in qs:
			if(q[2] == "x"):
				if(absmax_x == None or np.abs(q[1]) > np.abs(absmax_x[1])):
					absmax_x = q
			else:
				if(absmax_y == None or np.abs(q[1]) > np.abs(absmax_y[1])):
					absmax_y = q

		#we'll use this to determine the location of the cluster
		#by finding any positive (collection) integrals within a
		#neighborhood of that absmax location. 
		pitch = self.chmap[int(self.config["asic"])]["strip_pitch"] #mm
		neigh = self.config["max_charge_adjacency"]*pitch
		x_neighbors = []
		y_neighbors = []
		for i in range(len(qxs)):
			if(np.abs(xs[i] - absmax_x[0]) <= neigh):
				x_neighbors.append([xs[i], qxs[i]])
		for i in range(len(qys)):
			if(np.abs(ys[i] - absmax_y[0]) <= neigh):
				y_neighbors.append([ys[i], qys[i]])

		x_neighbors_pos = [_ for _ in x_neighbors if _[1] > 0]
		y_neighbors_pos = [_ for _ in y_neighbors if _[1] > 0]


		#if there are no positives in the neighbors, then 
		#the cluster position in that dimension should be
		#the weighted avg of the pulses in that dimension. 
		summed_collection_integrals = 0 #will be a prototype of the total reconstructed charge
		if(len(x_neighbors) == 0):
			#or if the whole list is empty, then we have no x position
			xpos = None
		elif(len(x_neighbors_pos) == 0):
			xpos = np.average([_[0] for _ in x_neighbors], weights=[_[1] for _ in x_neighbors])
		else:
			#if any of the neighbors are passing the 1-sigma threshold, i.e. 
			#are "passing" and are positive, use that as the position. 
			passing_positives_temp = [_ for _ in x_neighbors_pos if _[0] in xxpass]
			passing_negatives_temp = [_ for _ in x_neighbors if _[0] in xxpass and _[1] < 0]
			if(len(passing_positives_temp) > 0):
				xpos = np.average([_[0] for _ in passing_positives_temp], weights=[_[1] for _ in passing_positives_temp])
				summed_collection_integrals += np.sum([_[1] for _ in passing_positives_temp])
			elif(len(passing_negatives_temp) > 0):
				#otherwise, do a weighted average of only the negative 
				#passing pulses in the neighborhood, and don't add to the collected charge
				xpos = np.average([_[0] for _ in passing_negatives_temp], weights=[_[1] for _ in passing_negatives_temp])
			else:
				#theyre both empty lists. 
				xpos = None
				

		if(len(y_neighbors) == 0):
			#or if the whole list is empty, then we have no x position
			ypos = None
		elif(len(y_neighbors_pos) == 0):
			ypos = np.average([_[0] for _ in y_neighbors], weights=[_[1] for _ in y_neighbors])
		else:
			#if any of the neighbors are passing the 1-sigma threshold, i.e. 
			#are "passing" and are positive, use that as the position. 
			passing_positives_temp = [_ for _ in y_neighbors_pos if _[0] in yypass]
			passing_negatives_temp = [_ for _ in y_neighbors if _[0] in yypass and _[1] < 0]
			if(len(passing_positives_temp) > 0):
				ypos = np.average([_[0] for _ in passing_positives_temp], weights=[_[1] for _ in passing_positives_temp])
				summed_collection_integrals += np.sum([_[1] for _ in passing_positives_temp])
			elif(len(passing_negatives_temp) > 0):
				#otherwise, do a weighted average of only the negative 
				#passing pulses in the neighborhood, and don't add to the collected charge
				ypos = np.average([_[0] for _ in passing_negatives_temp], weights=[_[1] for _ in passing_negatives_temp])
			else:
				#theyre both empty lists. 
				ypos = None

		#for debugging
		if(max(qxs) + max(qys) < 5000):
			fig, ax = plt.subplots(ncols = 2)
			ax[0].scatter(xs, qxs, label="X", s=300)
			ax[1].scatter(ys, qys, label="Y", s=300)
			ax[0].scatter(xxpass, qxpass, color='r', label="X > 1-sigma", s=300)
			ax[1].scatter(yypass, qypass, color='r', label="Y > 1-sigma", s=300)
			ax[0].axhspan(-np.std(qxs), np.std(qxs), color='gray', alpha=0.5)
			ax[1].axhspan(-np.std(qys), np.std(qys), color='gray', alpha=0.5)
			ax[0].scatter([_[0] for _ in x_neighbors], [_[1] for _ in x_neighbors], s=300, marker='x', color='g', label="X neighbors")
			ax[1].scatter([_[0] for _ in y_neighbors], [_[1] for _ in y_neighbors], s=300, marker='x', color='g', label="Y neighbors")
			ax[0].scatter([_[0] for _ in x_neighbors_pos], [_[1] for _ in x_neighbors_pos], s=300, marker='x', color='b', label="X neighbors > 0")
			ax[1].scatter([_[0] for _ in y_neighbors_pos], [_[1] for _ in y_neighbors_pos], s=300, marker='x', color='b', label="Y neighbors > 0")
			ax[0].axvline(xpos, color='k', linestyle='--', label="X centroid")
			ax[1].axvline(ypos, color='k', linestyle='--', label="Y centroid")
			ax[0].set_xlabel("X positions of Y strips [mm]")
			ax[1].set_xlabel("Y positions of X strips [mm]")
			ax[0].set_ylabel("Integral about time of largest pulse [ENC*us]")
			ax[1].set_ylabel("Integral about time of largest pulse [ENC*us]")
			ax[0].set_title("Total integral sum: {:d}".format(int(summed_collection_integrals)))
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


	
