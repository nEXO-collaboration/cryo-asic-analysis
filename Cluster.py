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



	#This reconstruction is both simple but also verbose...
	#This is Evan's attempt to illustrate what it presently does:
	#The assumed input is a set of pulses that correspond in 
	#proximity with one another, or every channel in both x and y strips. 
	#The integrals of each channel have already been performed about the
	#small region in time around a primary pulse that has been detected
	#by a threshold discriminator. 
	#1. Distinguish the integrals of x and y strips separately. 
	#2. Find the mean and standard deviation of the integrals of each strip type.
	#3. Find the pulses that are outside of 1-sigma of the mean of the integrals
	#4. Find the mean of the integrals of the pulses that are within 1-sigma of the mean
	#5. "Baseline" subtract the mean from the integrals of the pulses 

	#At this stage, we have a baseline corrected set of integrals, and we have
	#identified those channels that lie outside of a 1-sigma range on both positive
	#and negative side of the mean. 

	#6. Find the maximum positive charge in the cluster. If there is no positive charge
	#in the cluster, then we should not consider this cluster.
	#7. Find the absolute max for each strip type. This will return negative values if the
	#absmax is negative.
	#8. Find the neighbors of the absmax locations within a certain distance, defined by the
	#config file.
	#9. If there are no positive charges in the neighbors, then the cluster position in that
	#dimension should be the weighted average of the negative pulses in the neighborhood.
	#10. If there are positive charges in the neighbors, then the cluster position in that
	#dimension should be the weighted average of the positive pulses in the neighborhood. The
	#total charge is also determined from this neighborhood and those channels with positive collection. 


	#The pulses which ultimately end up contributing to the total charge calculation,
	#as well as a separate list of those pulses that went into determining to position,
	#are stored in the reduced quantities for future reference. 
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
		charge_channels = [] #[pos, 'x'] list of channels used in charge calculation
		position_channels = [] #[pos, 'x'] for channels used in the position calculation
		if(len(x_neighbors) == 0):
			#or if the whole list is empty, then we have no x position
			xpos = None
		elif(len(x_neighbors_pos) == 0):
			xpos = np.average([_[0] for _ in x_neighbors], weights=[_[1] for _ in x_neighbors])
			position_channels += [[_[0], 'x'] for _ in x_neighbors]
		else:
			#if any of the neighbors are passing the 1-sigma threshold, i.e. 
			#are "passing" and are positive, use that as the position. 
			passing_positives_temp = [_ for _ in x_neighbors_pos if _[0] in xxpass]
			passing_negatives_temp = [_ for _ in x_neighbors if _[0] in xxpass and _[1] < 0]
			if(len(passing_positives_temp) > 0):
				xpos = np.average([_[0] for _ in passing_positives_temp], weights=[_[1] for _ in passing_positives_temp])
				summed_collection_integrals += np.sum([_[1] for _ in passing_positives_temp])
				charge_channels += [[_[0], 'x'] for _ in passing_positives_temp]
				position_channels += [[_[0], 'x'] for _ in passing_positives_temp]
			elif(len(passing_negatives_temp) > 0):
				#otherwise, do a weighted average of only the negative 
				#passing pulses in the neighborhood, and don't add to the collected charge
				xpos = np.average([_[0] for _ in passing_negatives_temp], weights=[_[1] for _ in passing_negatives_temp])
				position_channels += [[_[0], 'x'] for _ in passing_negatives_temp]
			else:
				#theyre both empty lists. 
				xpos = None


		if(len(y_neighbors) == 0):
			#or if the whole list is empty, then we have no x position
			ypos = None
		elif(len(y_neighbors_pos) == 0):
			ypos = np.average([_[0] for _ in y_neighbors], weights=[_[1] for _ in y_neighbors])
			position_channels += [[_[0], 'y'] for _ in y_neighbors]
		else:
			#if any of the neighbors are passing the 1-sigma threshold, i.e. 
			#are "passing" and are positive, use that as the position. 
			passing_positives_temp = [_ for _ in y_neighbors_pos if _[0] in yypass]
			passing_negatives_temp = [_ for _ in y_neighbors if _[0] in yypass and _[1] < 0]
			if(len(passing_positives_temp) > 0):
				ypos = np.average([_[0] for _ in passing_positives_temp], weights=[_[1] for _ in passing_positives_temp])
				summed_collection_integrals += np.sum([_[1] for _ in passing_positives_temp])
				charge_channels += [[_[0], 'y'] for _ in passing_positives_temp]
				position_channels += [[_[0], 'y'] for _ in passing_positives_temp]

			elif(len(passing_negatives_temp) > 0):
				#otherwise, do a weighted average of only the negative 
				#passing pulses in the neighborhood, and don't add to the collected charge
				ypos = np.average([_[0] for _ in passing_negatives_temp], weights=[_[1] for _ in passing_negatives_temp])
				position_channels += [[_[0], 'y'] for _ in passing_negatives_temp]
			else:
				#theyre both empty lists. 
				ypos = None


		#if there is no charge reconstructed, then we should
		#not sore this cluster any longer
		if(summed_collection_integrals == 0):
			return None

		#get a new list of pulse objects that went into the
		#charge and position calculation 
		charge_pulses = []
		position_pulses = []
		for pulse in self.pulses:
			typ = Util.get_channel_type(self.chmap, pulse.ch)
			pos = Util.get_channel_pos(self.chmap, pulse.ch)
			if([pos, typ] in charge_channels):
				charge_pulses.append(pulse)
			if([pos, typ] in position_channels):
				position_pulses.append(pulse)
		
		#store these pulses in reduced quantities
		self.d["charge_pulses"] = charge_pulses
		self.d["position_pulses"] = position_pulses
		self.d["n_pulses"] = len(charge_pulses)

		#the total charge reconstructed needs to be divided
		#by the integration range, as it is presently in units
		#of ENC*us.
		integ_t = np.abs(self.config["integ_window"][1] - self.config["integ_window"][0]) #us
		self.d["q"] = summed_collection_integrals/integ_t

		#store the position of the cluster
		self.d["x"] = xpos
		self.d["y"] = ypos #can be None's if no position is found.

		#For the time of arrival, use the time of the
		#largest positive charge pulse in the charge_pulses
		best_t = None
		max_q = None
		all_ts = []
		for pulse in charge_pulses:
			if(pulse.d["integral"] > 0):
				all_ts.append(pulse.d["t_arrival"])
				if(max_q == None or pulse.d["integral"] > max_q):
					max_q = pulse.d["integral"]
					best_t = pulse.d["t_arrival"]

		self.d["t_arrival"] = best_t
		#calculate a spread in the times if there are more than one
		if(len(all_ts) > 1):
			self.d["dt"] = np.std(all_ts)
		else:
			self.d["dt"] = 0


		#for debugging
		"""
		if(self.d["q"] > 1500):
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
		"""

		#this return format, where we return None if 
		#the cluster should be rejected and self if the
		#cluster is good, will eventually (hopefully)
		#evolve to distinguish possible multiple clusters
		#within this one time frame. There have been instances
		#where it seems like there may be multiple sites arriving 
		#at the same time, which is rare and strange to disambiguate. 
		#For now, we just return the cluster if we ant to keep it. 
		return self #return the cluster object





	def plot_cluster(self):
		fig, ax = plt.subplots()
		for pulse in self.pulses:
			wav = pulse.wav 
			ts = range(pulse.idx_start, pulse.idx_start + len(wav))
			ax.plot(ts, wav)
		plt.show()


	
