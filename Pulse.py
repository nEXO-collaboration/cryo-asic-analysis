import numpy as np 
import Utilities as Util
import matplotlib.pyplot as plt 
from scipy.signal import find_peaks
import sys
import os
import yaml
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from sklearn.cluster import DBSCAN


class Pulse:
	#initialize the pulse with its reduced quantities which
	#should be parsed externally (by the instantiator) from a yaml file
	#config is the config dict, no file parsing needed, as this will always
	#come from something that has already parsed the config. 
	#Wav: waveform data to be used during processing. dealocated afterwards.
	#ch: channel of the pulse 
	#idx_start: the pulses may not have the same number of samples, as
	#we will find multiple pulses in one window, we will separated them into
	#different pulse objects. But the time of the start will be preserved
	#by storing the index within the original buffered waveform at which the wav
	#data starts. 
	def __init__(self, rqs, config, wav, ch, idx_start = 0):

		self.rqs = rqs
		self.d = {}
		#initialize the cluster dictionary, with
		#initialze values specified in the yaml file that 
		#defines RQs. 
		for key in self.rqs:
			self.d[key] = self.rqs[key]

		self.config = config #already a dict


		self.ch = ch
		self.wav = wav
		self.idx_start = idx_start
		self.integ = [] #rolling integral values
		self.integ_idx = [] #rolling integral sample numbers

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


	def get_gaussian_smoothed_waveform(self):
		sig_us = self.config["peak_detect_smoothing"]
		sig_n = int(sig_us*self.config["sampling_rate"])
		return gaussian_filter(self.wav, sigma=sig_n)


	#a simple peak finder that ignores masked regions. 
	#It finds peaks of both polarities. 
	def find_peaks(self, thresh=None):
		if(thresh == None):
			#get a quick baseline noise estimate
			bl_window = [int(self.config["baseline"][0]*self.config["sampling_rate"]), int(self.config["baseline"][1]*self.config["sampling_rate"])]
			std = np.std(self.wav[bl_window[0]:bl_window[1]])
			thresh = self.config["high_threshold"]*std

		thresh = float(thresh) #comes in as ndarray secretly 

		
		#collective for both polarities
		all_pulses = []

		#POSITIVE POLARITY
		#get all indices that are above threshold
		wav_smoothed = np.array(self.get_gaussian_smoothed_waveform())
		pass_thresh = np.where(wav_smoothed >= thresh)[0]
		#cluster the indices with a 1 sample separation maximum
		peak_clusters = Util.simple_1d_clustering(pass_thresh, 1)
		#remove all clusters with 1 or fewer samples. 
		peak_clusters = [_ for _ in peak_clusters if len(_) > 1]
		#for each cluster, get the mean value of the cluster indices. 
		temp_pulses = [int(np.mean([_[0] for _ in c])) for c in peak_clusters]

		
		#remove any peaks that are in the masked region
		igs_us = self.config["ignore_regions"]
		igs_s = [[int(ig[0]*self.config["sampling_rate"]), int(ig[1]*self.config["sampling_rate"])] for ig in igs_us]
		masked_pulses = []
		
		for i, tp in enumerate(temp_pulses):
			mask = False
			for ig in igs_s:
				if(tp > ig[0] and tp < ig[1]):
					mask = True
			if(mask == False):
				masked_pulses.append(tp)

		#update the collection before moving to negative polarity
		all_pulses = masked_pulses

		#NEGATIVE POLARITY
		#get all indices that are above threshold
		wav_smoothed = np.array(self.get_gaussian_smoothed_waveform())
		pass_thresh = np.where(wav_smoothed <= -1*thresh)[0]
		#cluster the indices with a 1 sample separation maximum
		peak_clusters = Util.simple_1d_clustering(pass_thresh, 1)
		#remove all clusters with 1 or fewer samples. 
		peak_clusters = [_ for _ in peak_clusters if len(_) > 1]
		#for each cluster, get the mean value of the cluster indices. 
		temp_pulses = [int(np.mean([_[0] for _ in c])) for c in peak_clusters]

		#remove any peaks that are in the masked region
		igs_us = self.config["ignore_regions"]
		igs_s = [[int(ig[0]*self.config["sampling_rate"]), int(ig[1]*self.config["sampling_rate"])] for ig in igs_us]
		masked_pulses = []

		for i, tp in enumerate(temp_pulses):
			mask = False
			for ig in igs_s:
				if(tp > ig[0] and tp < ig[1]):
					mask = True
			if(mask == False):
				masked_pulses.append(tp)

		all_pulses += masked_pulses
		
		"""
		if(len(all_pulses) != 0):
			fig, ax = plt.subplots()
			print(thresh)
			ax.plot(self.wav, 'ko-')
			ax.scatter(all_pulses, self.wav[all_pulses], s=500)
			ax.axhline(thresh)
			ax.set_xlim(np.min(all_pulses)-100, np.max(all_pulses)+100)
			plt.show()
		"""
		
		return all_pulses

		

		

	#performs a windowed integral that rolls over the waveform.
	#populates self attributes that store that info but gets deleted
	#at the end of pulse processing. Window is in microseconds.
	#The shift in the window is half the width of the window. Thus,
	#the time series it creates is (total_samples)/(window_samples/2) roughly
	def rolling_integral(self, window=10):
		window_s = int(window * self.config["sampling_rate"]) #convert window to samples
		half_window = int(window_s/2)
		total_samples = len(self.wav)
		#start indexes of the windows
		window_idxs = np.arange(0, total_samples, half_window)
		integ = []
		for i in window_idxs:
			if(i+window_s >= total_samples):
				end = total_samples-1
			else: 
				end = i+window_s
			integ.append(np.trapz(y = self.wav[i:end], dx = 1/self.config["sampling_rate"]))

		self.integ = integ
		self.integ_idx = window_idxs


		

	#the main processing function after initial vetting of pulse properties. 
	#At this stage, usually the pulses are short and isolated, do not overlap
	#with ignore regions, and are not single data point glitches.
	def calculate_reduced_quantities(self):
		#find polarity, max, and min of the pulse
		self.d["max"] = Util.ADC_to_ENC(np.max(self.wav), self.config["gain"], self.config["pt"])
		self.d["min"] = Util.ADC_to_ENC(np.min(self.wav), self.config["gain"], self.config["pt"])
		if(abs(self.d["max"]) > abs(self.d["min"])):
			self.d["polarity"] = 1
		else:
			self.d["polarity"] = -1
		
		#find the time of the max and min
		max_idx = np.argmax(self.wav)
		min_idx = np.argmin(self.wav)
		#in absolute time relative to the original full waveform buffer
		self.d["tmax"] = (max_idx + self.idx_start)/self.config["sampling_rate"]
		self.d["tmin"] = (min_idx + self.idx_start)/self.config["sampling_rate"]

		
		#integrate around the peak, first find the peak index
		if(self.d["polarity"] == 1):
			peak_idx = max_idx
		else:
			peak_idx = min_idx
		integ_window_us = self.config["integ_window"]
		integ_window_s = [int(integ_window_us[0] * self.config["sampling_rate"]), int(integ_window_us[1] * self.config["sampling_rate"])]
		integ_wave = self.wav[peak_idx + integ_window_s[0]:peak_idx + integ_window_s[1]]
		#positive integral
		#set all negative values to 0
		integ_wave_p = integ_wave.copy() 
		integ_wave_p[integ_wave_p < 0] = 0
		self.d["pos_integral"] = np.trapz(y = integ_wave_p, dx = 1/self.config["sampling_rate"])
		self.d["pos_integral"] = Util.ADC_to_ENC(self.d["pos_integral"], self.config["gain"], self.config["pt"])
		#negative integral
		#set all positive values to 0
		integ_wave_n = integ_wave.copy()
		integ_wave_n[integ_wave_n > 0] = 0
		self.d["neg_integral"] = np.trapz(y = integ_wave_n, dx = 1/self.config["sampling_rate"])
		self.d["neg_integral"] = Util.ADC_to_ENC(self.d["neg_integral"], self.config["gain"], self.config["pt"])
		#combined
		self.d["integral"] = self.d["pos_integral"] + self.d["neg_integral"]
		
		#channel
		self.d["channel"] = self.ch

		#calculate the arrival time based on a constant fraction discriminator
		#method. Also do the same to calculate the width. 
		f_a = self.config["cfd_arrival"]
		f_w = self.config["cfd_width"]
		wav_pol = np.array(self.wav)*self.d["polarity"]
		thr_a = f_a*np.max(wav_pol)
		thr_w = f_w*np.max(wav_pol)
		#there is a bit of complexity at the CFD loop, so that we can
		#interpolate linearly finer than the sampling rate. that's all this code
		#about the i - interpolation buffer and such, also avoiding end-of-buffer errors. 
		interpolation_buffer = 1 #after finding the threshold crossing, we will interpolate a small number of samples
		#find the threshold crossing for arrival
		for i, samp in enumerate(wav_pol):
			if(samp > thr_a):
				begin = i - interpolation_buffer
				end = i + interpolation_buffer + 1
				if(begin < 0):
					begin = 0
				if(end >= len(wav_pol)):
					end = len(wav_pol) - 1
				interp_wav = wav_pol[begin:end]
				interp_idx = range(begin, end)
				s_interp = interp1d(interp_wav, interp_idx, kind='linear', bounds_error=False, fill_value=None)
				arrival_idx_float = s_interp(thr_a)
				self.d["t_arrival"] = (arrival_idx_float + self.idx_start)/self.config["sampling_rate"]
				break
		#find the threshold crossing on both sides for the width
		i = np.argmax(wav_pol)
		#this is a 
		j = i
		k = i
		j_idx_float = k_idx_float = None

		#right side of wave
		while(wav_pol[j] > thr_w):
			j += 1
			if(j >= len(wav_pol)):
				break
			

		if(j >= len(wav_pol)):
			#didn't find the CFD crossing
			j_idx_float = None
		else:
			begin = j - interpolation_buffer
			end = j + interpolation_buffer + 1
			if(begin < 0):
				begin = 0
			if(end >= len(wav_pol)):
				end = len(wav_pol) - 1

			interp_wav = wav_pol[begin:end]
			interp_idx = range(begin, end)
			s_interp = interp1d(interp_wav, interp_idx, kind='linear', bounds_error=False, fill_value=None)
			j_idx_float = s_interp(thr_w)

		#left side of wave
		while(wav_pol[k] > thr_w):
			k -= 1
			if(k <= 0):
				break
			
		if(k <= 0):
			#didn't find the CFD crossing
			k_idx_float = None
		else:	
			begin = k - interpolation_buffer
			end = k + interpolation_buffer + 1
			if(begin < 0):
				begin = 0
			if(end >= len(wav_pol)):
				end = len(wav_pol) - 1
			interp_wav = wav_pol[begin:end]
			interp_idx = range(begin, end)
			s_interp = interp1d(interp_wav, interp_idx, kind='linear', bounds_error=False, fill_value=None)
			k_idx_float = s_interp(thr_w)

		if(j_idx_float == None or k_idx_float == None):
			self.d["width"] = 0
		else:	
			self.d["width"] = (j_idx_float - k_idx_float)/self.config["sampling_rate"]
		if(self.d["width"] == 0):
			self.d["asymmetry"] = 0
		else:
			self.d["asymmetry"] = (j_idx_float - arrival_idx_float)/(j_idx_float - k_idx_float)/self.config["sampling_rate"]
	

	#for debugging
	def plot_pulse(self, ax=None, ax2=None, show=False):
		if(ax == None):
			fig, ax = plt.subplots()
			ax2 = ax.twiny()
		ax.plot(self.wav, 'ko-')
		ax2.plot(range(self.idx_start, self.idx_start + len(self.wav)), self.wav, 'ko-')
		if(show):
			plt.show()
		return ax, ax2




		
