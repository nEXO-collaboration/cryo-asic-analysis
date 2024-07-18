import sys
import pickle
import os
import yaml
import numpy as np 
import math
import matplotlib.pyplot as plt 
import pandas as pd
from scipy.signal import periodogram
from astropy.timeseries import LombScargle
from scipy.optimize import curve_fit
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.path as mpath
import matplotlib.transforms as transforms
from matplotlib.ticker import ScalarFormatter
from scipy import signal
from scipy.interpolate import interp1d

#used in pulse finding analysis
from operator import itemgetter
from itertools import groupby




class CryoAsicAnalysis:
	#the config input is the same format as commented in CryoAsicFile class,
	#namely is either a dictionary or a yaml file path. 
	def __init__(self, infile, config):
		self.infile = infile
		if(self.infile[-1:] != "p"):
			print("The input file you have provided for CryoAsicAnalysis object is not a pickle file: " + self.infile)
			return None
		
		#check if file exists
		if(os.path.isfile(self.infile) == False):
			print("The input file you have provided for CryoAsicAnalysis object does not exist: " + self.infile)
			return None
		
		self.configfile_or_dict = config
		self.config = None #has global analysis config dictionary contents
		self.chmap = None #has the channel and tile mappings. 
		self.load_config(config) #loads both of the above dictionaries


		print("loading the waveform dataframe from  " + self.infile)
		self.df = pickle.load(open(self.infile, "rb"))[0] #0th element of the list is dataframe. 
		print("Done loading")

		self.nevents_total = len(self.df.index)

		self.sf = float(self.config["sampling_rate"]) #MHz "sf": sampling_frequency
		self.dT = 1.0/self.sf
		self.nsamples = len(self.df.iloc[0]["Data"][0])
		self.times = np.arange(0, self.nsamples*self.dT, self.dT)
		self.live_time = self.nsamples*self.dT


		self.noise_df = pd.DataFrame(columns=["Channel", "Freqs", "PSD", "STD"]) #rows are channels, columns are noise information
		self.noise_df.set_index("Channel")

		self.corr_mat = None


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

	#get the channel type from the channel number
	def get_channel_type(self, ch):
		if(self.chmap is None):
			print("Channel map didn't properly load")
			return None
		local_ch = ch % 64 #the channel number on the asic level. 
		asic = math.floor(ch/64) # the asic ID that this ch corresponds to. 
		
		
		if(asic in self.chmap):
			if(local_ch in self.chmap[asic]["xstrips"]):
				return 'x'
			elif(local_ch in self.chmap[asic]["ystrips"]):
				return 'y'
			else:
				return 'dummy'
			
		else:
			print("Asic {:d} not found in the configuration file channel map".format(asic))
			return None
		
	#returns the global position of the channel in the TPC
	#using knowledge of the tile position. If it is a dummy, 
	#return 0, 0 
	def get_channel_pos(self, ch):
		if(self.chmap is None):
			print("Channel map didn't properly load")
			return None

		local_ch = ch % 64 #the channel number on the asic level. 
		asic = math.floor(ch/64) # the asic ID that this ch corresponds to. 
		tile_pos = self.chmap[asic]["tile_pos"]
		pitch = self.chmap[asic]["strip_pitch"] #in mm

		if(asic in self.chmap):
			if(local_ch in self.chmap[asic]["xstrips"]):
				local_pos = float(self.chmap[asic]["xstrips"][local_ch])
				return (tile_pos[0], tile_pos[1] + np.sign(local_pos)*(np.abs(local_pos) - 0.5)*pitch)
			elif(local_ch in self.chmap[asic]["ystrips"]):
				local_pos = float(self.chmap[asic]["ystrips"][local_ch])
				return (tile_pos[0] + np.sign(local_pos)*(np.abs(local_pos) - 0.5)*pitch, tile_pos[1])
			else:
				return tile_pos #this is a dummy capacitor
		else:
			print("Asic {:d} not found in the configuration file channel map".format(asic))
			return None

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

	#given a time, return the closest sample index (int)
	def time_to_sample(self, t):
		idx = (np.abs(np.asarray(self.times) - t)).argmin()
		return idx 

	def sample_to_time(self, idx):
		return self.times[idx]

	def save_noise_df(self):
		pickle.dump([self.noise_df], open(self.infile[:-3]+"_noisedf.p", "wb"))

	def load_noise_df(self):
		if(os.path.exists(self.infile[:-3]+"_noisedf.p")):
			self.noise_df = pickle.load(open(self.infile[:-3]+"_noisedf.p", "rb"))[0]
			return True
		else:
			return False

	def create_df_from_event(self, evno):
		evdf = pd.DataFrame()
		ev_dict = {}
		ev_dict["Channel"] = []
		ev_dict["Data"] = []
		ev_dict["ChannelType"] = []
		ev_dict["ChannelPos"] = []

		ev = self.df.iloc[evno]
		for i, ch in enumerate(ev["Channels"]):
			ev_dict["Channel"].append(ch)
			ev_dict["Data"].append(ev["Data"][i])
			ev_dict["ChannelType"].append(self.get_channel_type(ch))
			ev_dict["ChannelPos"].append(self.get_channel_pos(ch))

		evdf = pd.DataFrame.from_dict(ev_dict)
		return evdf


	def is_channel_strip(self, ch):
		result = self.get_channel_type(ch)
		if(result == "dummy"):
			return False
		else:
			return True

	#for the moment, we will baseline subtract based
	#on an input window 
	def baseline_subtract(self):
		#indexes of data stream to use to calculate baseline. 
		#comes from microsecond range input from configuration dict
		index_range = [self.time_to_sample(self.config["baseline"][0]), self.time_to_sample(self.config["baseline"][1])]

		#baseline subtract all events
		for i, row in self.df.iterrows():
			waves = row["Data"]
			for ch in range(len(waves)):
				base = np.mean(waves[ch][index_range[0]:index_range[1]])
				row["Data"][ch] = np.array(waves[ch]) - base 


	#this plots the waveforms from x and y all on the same plot, 
	#overlayed, but with traces shifted relative to eachother by 
	#some number of ADC counts. if tileno is not none, it only plots
	#one tile, associated with an integer passed as argument
	def plot_waves(self, evno, chs_to_plot = [], window = [ ], title =" ", ENC = False):
		if(evno < 0):
			evno = 0
		if(evno > self.nevents_total):
			print("That event is not in the dataframe: " + str(evno))
			return

		ev = self.df.iloc[evno]

		adc_shift = 0 #number of adc counts to shift traces

		chs = ev["Channels"]
		if(chs_to_plot == []):
			chs_to_plot = chs 
		waves = ev["Data"]
		#sort them simultaneously by channel number
		chs, waves = (list(t) for t in zip(*sorted(zip(chs, waves))))
		nch = len(chs)
		nsamp = len(waves[0])
		times = np.arange(0, nsamp*self.dT, self.dT)

		fig, ax = plt.subplots(figsize=(10,8))
		curshift = 0
		for i in range(nch):
			if(i in chs_to_plot):
				if ENC: ax.plot(times, self.ADC_to_ENC(waves[i] + curshift), label=str(chs[i]))
				else: ax.plot(times, waves[i] + curshift, label=str(chs[i]))
				ax.set_title(title)
				if window:
					ax.set_xlim(window[0], window[1])
			#curshift += adc_shift

		ax.set_xlabel('Time (us)')
		if ENC: ax.set_ylabel("Channel Shifted ENC ($e^-$)")
		else: ax.set_ylabel("Channel Shift ADC Counts")


		plt.show()

	def get_wave(self, evno, ch):
		if(evno < 0):
			evno = 0
		if(evno > self.nevents_total):
			print("That event is not in the dataframe: " + str(evno))
			return

		ev = self.df.iloc[evno]
		if(ch in ev["Channels"]):
			return ev["Data"][ch]
		else:
			return None
		
	def get_baseline_samples(self, wave):
		#indexes of data stream to use to calculate baseline. 
		#comes from microsecond range input from configuration dict
		index_range = [self.time_to_sample(self.config["baseline"][0]), self.time_to_sample(self.config["baseline"][1])]
		return wave[index_range[0]:index_range[1]]


	#this is a general pulse finding algorithm 
	#that uses a bipolar threshold discriminator 
	#with a N*sigma threshold, using baseline information. 
	#It is a prototype of what goes on in data reduction, but we need
	#something like this fast in order to do some processing before calculating
	#power spectral densities (pulse rejection). It assumes that all events
	#have been baseline subtracted using the baseline subtract function. 
	def find_pulses_in_channel(self, evno, ch, n_sigma=None):
		wave = self.get_wave(evno, ch)
		pulses = []
		if(n_sigma is None):
			n_sigma = self.config["pulse_threshold"]
		
		#get the standard deviation of the baseline. 
		std = np.std(self.get_baseline_samples(wave))

		#threshold is N*sigma above the baseline
		thr = n_sigma*std 

		#find all indices of samples above baseline and below baseline
		positive_clip = np.where(wave >= thr)[0]
		negative_clip = np.where(wave <= -thr)[0]

		#at this point, we can end early if they are both empty
		if(len(positive_clip) == 0 and len(negative_clip) == 0):
			return pulses

		for key, group in groupby(enumerate(positive_clip), lambda i: i[0] - i[1]):
			group = list(map(itemgetter(1), group))
			#this is the case if there are a few samples above threshold
			if(len(group) > 1):
				idxs = [group[0], group[-1]]
				#find the maximum amplitude in this group
				amp = max(wave[group])
				#find the time of the max amplitude
				time = float(self.sample_to_time(group[np.argmax(wave[group])]))
				#find the index of the first sample below threshold
				pulses.append({"channel": ch, "time": time, "index": idxs, "amplitude": amp})
			#if its one sample long, just slightly different syntax
			else:
				amp = wave[group[0]]
				time = float(self.sample_to_time(group[0]))
				pulses.append({"channel": ch, "time": time, "index": group, "amplitude": amp})

		#repeat for negative pulses
		for key, group in groupby(enumerate(negative_clip), lambda i: i[0] - i[1]):
			group = list(map(itemgetter(1), group))
			#this is the case if there are a few samples above threshold
			if(len(group) > 1):
				idxs = [group[0], group[-1]]
				#find the maximum amplitude in this group
				amp = min(wave[group])
				#find the time of the max amplitude
				time = float(self.sample_to_time(group[np.argmin(wave[group])]))
				#find the index of the first sample below threshold
				pulses.append({"channel": ch, "time": time, "index": idxs, "amplitude": amp})
			#if its one sample long, just slightly different syntax
			else:
				amp = wave[group[0]]
				time = float(self.sample_to_time(group[0]))
				pulses.append({"channel": ch, "time": time, "index": group, "amplitude": amp})

		#debugging
		"""
		fig, ax = plt.subplots()
		ax.plot(self.times, wave)
		ax.axhline(thr, color='r')
		ax.axhline(-thr, color='r')
		print(std)
		for pulse in pulses:
			if(pulse["channel"] == ch):
				ax.scatter(pulse["time"], pulse["amplitude"], color='r', s=40)

		plt.show()
		"""

		return pulses

	#similar to above, but just pneumonic for looping through all channels. 
	def find_pulses_in_event(self, evno, n_sigma=None):
		ev = self.df.iloc[evno]
		chs = ev["Channels"]
		if(n_sigma is None):
			n_sigma = self.config["pulse_threshold"]
		
		#the output is a list of "pulse" objects, which 
		#are dictionaries with information like:
		#"channel": channel number
		#"time": in microseconds of the maximum in the pulse 
		#"index": [first crossing index, leaving threshold index] 
		#"amplitude": in mV 
		pulses = []  
		for ch in chs:
			temp_pulses = self.find_pulses_in_channel(evno, ch, n_sigma)
			pulses = pulses + temp_pulses
		return pulses
	

	#given a channel and an event number, remove the samples
	#that contain any sort of glitch or pulse found by a pulsefinder. 
	def remove_pulses_in_wave(self, evno, ch, n_sigma=None):
		wave = self.get_wave(evno, ch)
		pulses = self.find_pulses_in_channel(evno, ch, n_sigma)
		indices_to_delete = []
		for p in pulses:
			if(len(p["index"]) == 1):
				indices_to_delete.append(p["index"][0])
			else:
				indices_to_delete = indices_to_delete + list(range(p["index"][0], p["index"][1]+1))
		
		if(len(wave) in indices_to_delete):
			indices_to_delete.remove(len(wave))
		
		wave = np.delete(wave, indices_to_delete)
		times = np.delete(self.times, indices_to_delete)
		return wave, times

	#Performs a pulse finding algorithm
	#to find any peaks above a threshold, then 
	#performs a periodogram with the Lomb-Scargle algorithm which
	#is robust to unevenly sampled data. This function is about 
	#10x slower than the regular PSD function. I would suggest not
	#using it unless you are really sure it matters. 
	def calculate_avg_psds_ignore_pulses(self):
		self.load_config(self.configfile_or_dict)
		self.baseline_subtract() #baseline subtract all events
		chs = self.df.iloc[0]["Channels"]
		nevents = len(self.df.index) #looping through all events
		#frequency range, to change units in the density to V^2/Hz
		#max freq - min freq
		freq_range = (self.sf/2.0 - 1.0/(self.dT*len(self.times)))*1e6
		#these are used to linearly interpolate so that we can average many lists
		#with different lengths.
		fine_freqs = np.array(np.linspace(1.0/(self.dT*len(self.times)), self.sf/2.0, len(self.times*10)))*1e6
		fine_freqs = fine_freqs[1:-1]
		coarse_freqs = np.array(np.linspace(1.0/(self.dT*len(self.times)), self.sf/2.0, len(self.times)))*1e6
		coarse_freqs = coarse_freqs[1:-1]
		for ch in chs:
			pxx_tot = None
			freqs = None
			avg_event_counter = 0 #number of events over which the avg is calculated
			for i in range(nevents):
				wave = self.get_wave(i, ch)
				times = self.times

				#find pulses in the waveform
				pulses = self.find_pulses_in_channel(i, ch)
				indices_to_delete = []
				if(len(pulses) > 0):
					#remove samples in the wave and a time stream associated with 
					#the places where pulses or glitches exist. 
					for p in pulses:
						if(len(p["index"]) == 1):
							indices_to_delete += p["index"]
						else:
							indices_to_delete = indices_to_delete + list(range(p["index"][0], p["index"][1]+1))
				
				if(len(wave) in indices_to_delete):
					indices_to_delete.remove(len(wave))
				
				wave = np.delete(wave, indices_to_delete)
				times = np.delete(times, indices_to_delete)

				wave = [_*self.config["mv_per_adc"]/1000. for _ in wave] #putting ADC units into volts
				
				times = np.array(times)*1e-6 #convert to seconds for the lombscargle
				fs, pxx = LombScargle(times, wave, normalization='psd').autopower(minimum_frequency = 1.0/(self.dT*len(self.times)*1e6), maximum_frequency = self.sf*1e6/2.0)
				

				pxx = pxx/freq_range #convert to V^2/Hz
				#interpolate to a fine frequency range, and then
				#go back to coarse once done averaging

				pxx_fine = interp1d(fs, pxx)(fine_freqs)
				if(pxx_tot is None):
					pxx = interp1d(fine_freqs, pxx_fine)(coarse_freqs)
					pxx_tot = pxx 
				else:
					pxx_tot_fine = np.array(interp1d(coarse_freqs, pxx_tot)(fine_freqs))
					pxx_tot_fine = pxx_tot_fine + pxx_fine
					pxx_tot = interp1d(fine_freqs, pxx_tot_fine)(coarse_freqs)
					
				avg_event_counter += 1

			pxx_tot = pxx_tot/float(avg_event_counter)


			#add to the noise dataframe. 
			#if the row for this channel already exists, just update columns
			if(ch in self.noise_df.index):
				self.noise_df.at[ch,"Freqs"] = coarse_freqs
				self.noise_df.at[ch,"PSD"] = pxx_tot
			else:
				s = pd.Series()
				s["Freqs"] = coarse_freqs
				s["PSD"] = pxx_tot
				s["Channel"] = ch 
				self.noise_df = pd.concat([self.noise_df, s.to_frame().transpose()], ignore_index=True)




	#for every channel, calculate a PSD using an event-by-event
	#calculation, averaging over all events. Save these periodogram
	#data in a dataframe. 
	def calculate_avg_psds(self):
		self.load_config(self.configfile_or_dict)
		self.baseline_subtract() #baseline subtract all events
		chs = self.df.iloc[0]["Channels"]
		nevents = len(self.df.index) #looping through all events

		for ch in chs:
			pxx_tot = None
			freqs = None
			avg_event_counter = 0 #number of events over which the avg is calculated
			for i in range(nevents):
				wave = self.get_wave(i, ch)
				wave = [_*self.config["mv_per_adc"]/1000. for _ in wave] #putting ADC units into volts
				
				fs, pxx = periodogram(wave, self.sf*1e6)

				if(pxx_tot is None):
					pxx_tot = np.array(pxx)
					freqs = fs
				else:
					pxx_tot = pxx_tot + np.array(pxx)

				avg_event_counter += 1

			pxx_tot = pxx_tot/float(avg_event_counter)

			#the first element is 0Hz, and we have no sensitivity there
			freqs = freqs[1:]
			pxx_tot = pxx_tot[1:]

			#add to the noise dataframe. 
			#if the row for this channel already exists, just update columns
			if(ch in self.noise_df.index):
				self.noise_df.at[ch,"Freqs"] = freqs
				self.noise_df.at[ch,"PSD"] = pxx_tot
			else:
				s = pd.Series()
				s["Freqs"] = freqs
				s["PSD"] = pxx_tot
				s["Channel"] = ch 
				self.noise_df = pd.concat([self.noise_df, s.to_frame().transpose()], ignore_index=True)

	#calculates the standard deviation of all samples in a dataset for
	#each channel. It also rejects any glitches (single sample) or pulses
	#on the waveform by performing the pulse finding algorithm and removing samples. 
	def calculate_stds(self, pulse_reject=True):
		chs = self.df.iloc[0]["Channels"]
		nevents = len(self.df.index) #looping through all events

		for ch in chs:
			if(ch == self.config["key_channel"]):
				continue
			all_samples = []
			for i in range(nevents):
				if(pulse_reject):
					wave, _ = self.remove_pulses_in_wave(i, ch)
					wave = list(wave)
				else:
					wave = list(self.get_wave(i, ch))

				all_samples = all_samples + wave

			self.noise_df.at[ch, "STD"] = np.std(all_samples)
			self.noise_df.at[ch, "Channel"] = ch


	#calculates the "correlation matrix" 
	#for each matrix elemnt i, j, ranging from N = len(chs)
	#calculate the cross-correlation with time lag which each i-j combination. 
	#take the abs maximum of the cc with lag (i.e. accepts correlated noise of all phases)
	#and store it in an img matrix. Normalize that row of the matrix relative
	#to the max value of the autocorrelation with time lag (i==j).
	# Over all events, average this matrix. Ignore events with glitches and pulses. 
	def make_correlation_matrix(self):
		chs = sorted(self.df.iloc[0]["Channels"])
		nevents = len(self.df.index) #looping through all events
		
		autocorr_norm = 1 #this is set as the max corr in autocorrelation
		img_full = np.zeros((len(chs), len(chs)))
		img_lags = np.zeros((len(chs), len(chs)))
		event_count = 0
		skip_event = False #flag to skip events if there is a glitch
		for evno in range(nevents):
			img = np.zeros((len(chs), len(chs)))
			for i in range(len(chs)):
				#zero out dead channels
				if(chs[i] in self.config["dead_channels"] or chs[i] == self.config["key_channel"]): continue
				evi = self.df.iloc[evno]
				wavei = evi["Data"][chs[i]]
				#if there are glitches in the event, and we dont expect a pulse, ignore
				if("noise" in self.infile and max(wavei, key=abs) > 500 or skip_event):
					skip_event = True
					break
				for j in range(len(chs)):
					#zero out dead channels
					if(chs[j] in self.config["dead_channels"] or chs[j] == self.config["key_channel"]): continue
					evj = self.df.iloc[evno]
					wavej = evj["Data"][chs[j]]
					#if there are glitches in the event, and we dont expect a pulse, ignore
					#if there are glitches in the event, and we dont expect a pulse, ignore
					if("noise" in self.infile and max(wavej, key=abs) > 500 or skip_event):
						skip_event = True
						break
					corr = signal.correlate(wavei, wavej)
					lags = signal.correlation_lags(len(wavej), len(wavei))
					corr = np.abs(corr)
					if(i == j):
						autocorr_norm = np.max(corr)

					img[i][j] = np.max(corr)
					img_lags[i][j] = lags[np.argmax(corr)]

				#normalize the column by the autocorrelation of this channel
				img[i] /= autocorr_norm

			#if event is skipped, do nothing
			if(skip_event):
				print("Skipped event " + str(evno))
				skip_event = False
				continue
			img_full = img_full + img 
			event_count += 1

		img_full /= event_count

		self.corr_mat = img_full 
		return img_full

	
	def plot_correlation_matrix(self):
		img = self.corr_mat
		fig, ax = plt.subplots(figsize=(15, 12))
		heat = ax.imshow(img.T, cmap='viridis', norm=LogNorm(vmin = 0.005, vmax = 1))
		ax.set_xlabel('ch')
		ax.set_ylabel('ch')
		cb = plt.colorbar(heat)
		cb.set_label("Max of lag cross-correlation normed to autocorrelation")
		ax.xaxis.set_ticks(np.arange(0, 63, 4))
		ax.yaxis.set_ticks(np.arange(0, 63, 4))
		#for (i, j), val in np.ndenumerate(img):
			#text = ax.text(i, j, round(img[i][j], 2), ha='center', va='center', color='w')	
		return ax 
		#plt.show()

	#returns the average of all off-diagonal elements in the correlation matrix
	def get_avg_correlation(self):
		if(self.corr_mat is None):
			return None 

		img = self.corr_mat 
		#python magic
		v = (img.sum(1) - np.diag(img))/(img.shape[1] - 1)
		return np.mean(v)
	

	def calc_glitch_rate(self, thresh = 10, sample_rate = 1):
		chs = sorted(self.df.iloc[0]["Channels"])
		nevents = len(self.df.index) #looping through all events
		glitch_count = 0
		total_time = 0

		for evt in range(nevents):
			for channel in chs:
				WVFM = self.get_wave(evt, channel)
				
				sigma = np.std(WVFM)
				mu = np.mean(WVFM)

				glitches = np.where(np.abs(WVFM-mu) > thresh*sigma)[0]
				glitch_count += len(glitches)
				total_time += len(WVFM)/sample_rate #time of waveform in us 
		
		glitch_rate = glitch_count/total_time
		return glitch_rate
	

	#take a look at find_pulses_in_event function. This one may want
	#to be deprecated eventually. 
	def event_finder(self, thresh = 2, show = True, window = 50):
		
		chs = sorted(self.df.iloc[0]["Channels"])
		nevents = len(self.df.index)
		self.evt_count = 0
		self.evt_rate = 0

		strip_channels = []
		for evt in range(nevents):
			evt_times = []
			for ch in chs:
				if not self.is_channel_strip(ch): continue
				WVFM = self.get_wave(evt, ch)
				if ch ==1: time = len(WVFM * self.dT)
				sigma = np.std(WVFM)
				datapoints = np.where(WVFM>=(sigma*thresh))[0]
				datapoints = np.where(WVFM>=(sigma*thresh))[0]
				for location in datapoints:
					if ((location>10) and (location<20)) or ((location >1000) and (location<1020)): continue  
					#if any(abs(time-location) <= 10 for time in self.evt_times): continue
					if WVFM[location - 1] <= thresh*sigma/1.5: continue
					self.evt_count += 1
					evt_times.append(location)
					if show:
						self.plot_waves(evt, chs_to_plot = strip_channels, window = [location - window, location + window], title="Candidate Event in Frame {0}".format(evt), ENC = True)

		self.evt_rate = self.evt_count/(self.live_time*1e-6)
		self.evt_rate_error = np.sqrt(self.evt_count)/(self.live_time*1e-6)

	
############################################################################################
# Event viewer functions
############################################################################################

	#plots the event in the same way that the CRYO ASIC GUI
	#event viewer plots, with a 2D greyscale hist using the
	#*channel numbers as the y axis bins and the time on x axis. 
	#this is distinct from other plots in that usually the positions
	#of the strips are used as the y axis; in this way, this is more
	#what the ASIC sees, ordering based on the asic channel IDs. 
	def plot_event_rawcryo(self, evno, ax=None, show=True):
		if(evno < 0):
			evno = 0
		if(evno > self.nevents_total):
			print("That event is not in the dataframe: " + str(evno))
			return

		ev = self.df.iloc[evno]
		chs = ev["Channels"]
		waves = ev["Data"]
		#sort them simultaneously by channel number
		chs, waves = (list(t) for t in zip(*sorted(zip(chs, waves))))
		nch = len(chs)
		nsamp = len(waves[0])
		times = np.arange(0, nsamp*self.dT, self.dT)

		img = np.zeros((nch, len(times)))


		#this is a fancy pandas way of making a 2D hist. I always
		#have trouble finding the meshgrid way of doing this. 
		#please modify this if you have a better way. 
		for i, ch in enumerate(chs):
			for j, t in enumerate(times):
				img[i][j] = waves[i][j]


		if(ax is None):
			fig, ax = plt.subplots(figsize=(12,8))

		heat = ax.imshow(img, cmap='viridis',interpolation='none', \
			extent=[min(times), max(times), max(chs) + 1, min(chs) - 1],\
			aspect=0.5*(max(times)/max(chs)))

		#mark dead channels
		for ch in self.config["dead_channels"]:
			ax.axhline(ch, linewidth=1, color='r')

		cbar = fig.colorbar(heat, ax=ax)
		cbar.set_label("ADC counts", labelpad=3)
		ax.set_xlabel("time (us)")
		ax.set_ylabel("ASIC ch number")
		ax.set_title("event number " + str(evno))
		
		if(show):
			plt.show()
		return fig, ax 



	#plots the event with a 2D hist where each tile has two subplots:
	#one for the X strips and one for the Y strips. Time on x axis, 
	#local channel position on Y axis. Finds all tiles and generates
	#subplots for each. For the moment, only does strips and not caps
	def plot_event_xysep(self, evno, ax=None, show=True):
		if(evno < 0):
			evno = 0
		if(evno > self.nevents_total):
			print("That event is not in the dataframe: " + str(evno))
			return

		ev = self.create_df_from_event(evno)
		nsamp = len(ev["Data"].iloc[0])
		times = np.arange(0, nsamp*self.dT, self.dT)

		#select only strips, X <= 51 is for strips and not capacitors
		xstrip_mask = (ev["ChannelType"] == 'x') 
		ystrip_mask = (ev["ChannelType"] == 'y') 
		xdf = ev[xstrip_mask]
		ydf = ev[ystrip_mask]

		#this is a fancy pandas way of making a 2D hist. I always
		#have trouble finding the meshgrid way of doing this. 
		#please modify this if you have a better way. 
		xstrip_img = np.zeros((len(xdf.index), len(times)))
		ystrip_img = np.zeros((len(ydf.index), len(times)))
		xpos = []
		ypos = []
		dead_xpos = [] #dead channel positions
		dead_ypos = [] #dead channel positions
		ch_idx = 0
		max_stdx = 0
		max_stdy = 0 #used to set color scale
		for i, row in xdf.iterrows():
			xpos.append(row["ChannelPos"][1])
			#is it a dead channel
			if(row["Channel"] in self.config["dead_channels"]):
				dead_xpos.append(row["ChannelPos"][1])

		#sort by positions
		xpos = sorted(xpos)
		for i, row in xdf.iterrows():
			wave = row["Data"]
			ch_idx = xpos.index(row["ChannelPos"][1])
			if(np.std(wave) > max_stdx):
					max_stdx = np.std(wave)
			for j in range(len(wave)):
				xstrip_img[ch_idx][j] = wave[j]
				



		for i, row in ydf.iterrows():
			ypos.append(row["ChannelPos"][0])
			#is it a dead channel
			if(row["Channel"] in self.config["dead_channels"]):
				dead_ypos.append(row["ChannelPos"][0])

		#sort by positions
		ypos = sorted(ypos)
		for i, row in ydf.iterrows():
			wave = row["Data"]
			ch_idx = ypos.index(row["ChannelPos"][0])
			if(np.std(wave) > max_stdy):
					max_stdy = np.std(wave)
			for j in range(len(wave)):
				ystrip_img[ch_idx][j] = wave[j]
				
		if(ax is None):
			fig, ax = plt.subplots(figsize=(12,16), nrows = 2)

		xheat = ax[0].imshow(xstrip_img, cmap='viridis', interpolation='none',\
			extent=[ min(times), max(times), max(xpos), min(xpos) - self.chmap[0]["strip_pitch"]],\
			aspect=0.5*(max(times)/(max(xpos) - min(xpos))), vmin=-2*max_stdx, vmax=2*max_stdx)

		yheat = ax[1].imshow(ystrip_img, cmap='viridis', interpolation='none',\
			extent=[min(times), max(times), max(ypos), min(ypos) - self.chmap[0]["strip_pitch"]],\
			aspect=0.5*(max(times)/(max(ypos) - min(ypos))), vmin=-2*max_stdy, vmax=2*max_stdy)

		xcbar = fig.colorbar(xheat, ax=ax[0])
		xcbar.set_label("ADC counts", labelpad=3)
		ax[0].set_xlabel("time (us)")
		ax[0].set_ylabel("x - strip position")
		ax[0].set_title("event number " + str(evno))
		ax[0].grid(False)

		ycbar = fig.colorbar(yheat, ax=ax[1])
		ycbar.set_label("ADC counts", labelpad=3)
		ax[1].set_xlabel("time (us)")
		ax[1].set_ylabel("y - strip position")
		ax[1].set_title("event number " + str(evno))
		ax[1].grid(False)

		#plot dead channels with lines through them
		for p in dead_xpos:
			ax[0].axhline(p-0.5*self.chmap[0]["strip_pitch"], linewidth=2, color='r')
		for p in dead_ypos:
			ax[1].axhline(p-0.5*self.chmap[0]["strip_pitch"], linewidth=2, color='r')

		if(show):
			plt.show()

		return fig, ax


	
	#this plots the waveforms from x and y all on the same plot, 
	#overlayed, but with traces shifted relative to eachother by 
	#some number of ADC counts. if tileno is not none, it only plots
	#one tile, associated with an integer passed as argument
	def plot_event_waveforms_separated(self, evno, ax=None, show=True):
		if(evno < 0):
			evno = 0
		if(evno > self.nevents_total):
			print("That event is not in the dataframe: " + str(evno))
			return

		ev = self.df.iloc[evno]

		adc_shift = 0 #number of adc counts to shift traces

		chs = ev["Channels"]
		waves = ev["Data"]
		#sort them simultaneously by channel number
		chs, waves = (list(t) for t in zip(*sorted(zip(chs, waves))))
		nch = len(chs)
		nsamp = len(waves[0])
		times = np.arange(0, nsamp*self.dT, self.dT)

		if(ax is None):
			fig, ax = plt.subplots(figsize=(10,8))
		curshift = 0
		for i in range(nch):
			ax.plot(times, waves[i] + curshift, label=str(chs[i]))
			curshift += adc_shift

		ax.set_xlabel('time (us)')
		ax.set_ylabel("channel shifted adc counts")

		if(show):
			plt.show()
		return fig, ax 


	#this plots the waveforms from x and y all on the same plot, 
	#overlayed, but with traces shifted relative to eachother by 
	#some number of ADC counts. if tileno is not none, it only plots
	#one tile, associated with an integer passed as argument
	def plot_strips_waveforms_separated(self, evno, fmt=None, fig = None, ax = None, show=True, sep=None):
		if(evno < 0):
			evno = 0
		if(evno > self.nevents_total):
			print("That event is not in the dataframe: " + str(evno))
			return


		ev = self.create_df_from_event(evno)
		nsamp = len(ev["Data"].iloc[0])
		times = np.arange(0, nsamp*self.dT, self.dT)

		#the create_df_from_event function identifies the type of each channel at play. 
		xstrip_mask = (ev["ChannelType"] == 'x')
		ystrip_mask = (ev["ChannelType"] == 'y')
		xdf = ev[xstrip_mask]
		ydf = ev[ystrip_mask]

		#this is a fancy pandas way of making a 2D hist. I always
		#have trouble finding the meshgrid way of doing this. 
		#please modify this if you have a better way. 
		xstrip_img = np.zeros((len(xdf.index), len(times)))
		ystrip_img = np.zeros((len(ydf.index), len(times)))
		xpos = []
		ypos = []
		xchs = []
		ychs = []
		for i, row in xdf.iterrows():
			xpos.append(row["ChannelPos"][1])
			xchs.append(row["Channel"])
		
		#sort the lists by position 
		xpos, xchs = (list(t) for t in zip(*sorted(zip(xpos, xchs))))
		for i, row in xdf.iterrows():
			wave = row["Data"]
			ch_idx = xchs.index(row["Channel"])
			for j in range(len(wave)):
				xstrip_img[ch_idx][j] = wave[j]*self.config["mv_per_adc"]

		for i, row in ydf.iterrows():
			ypos.append(row["ChannelPos"][0])
			ychs.append(row["Channel"])


		#sort the lists by position
		ypos, ychs = (list(t) for t in zip(*sorted(zip(ypos, ychs))))
		for i, row in ydf.iterrows():
			wave = row["Data"]
			ch_idx = ychs.index(row["Channel"])
			for j in range(len(wave)):
				ystrip_img[ch_idx][j] = wave[j]*self.config["mv_per_adc"]

		if(sep == None):
			mv_shift = 50 #number of adc counts to shift traces
		else:
			mv_shift = sep

		if(ax is None):
			fig, ax = plt.subplots(figsize=(12,16), nrows=2)

		curshift = 0
		if(fmt is None):
			fmt = '-'
		for i in range(len(xstrip_img)):
			if(xchs[i] in self.config["dead_channels"] or xchs[i] == self.config["key_channel"]):
				ax[0].plot(times, np.array(xstrip_img[i]) + curshift, 'k', label=str(xpos[i]))
			else:
				ax[0].plot(times, np.array(xstrip_img[i]) + curshift, fmt, label=str(xpos[i]))
			
			curshift += mv_shift

		curshift = 0
		for i in range(len(ystrip_img)):
			if(ychs[i] in self.config["dead_channels"] or ychs[i] == self.config["key_channel"]):
				ax[1].plot(times, np.array(ystrip_img[i]) + curshift, 'k', label=str(ypos[i]))
			else:
				ax[1].plot(times, np.array(ystrip_img[i]) + curshift, fmt, label=str(ypos[i]))
			
			curshift += mv_shift

		ax[0].set_xlabel('time (us)')
		ax[0].set_title("X-strips, event {:d}".format(evno))
		ax[0].set_ylabel("shifted mV")
		ax[1].set_xlabel('time (us)')
		ax[1].set_ylabel("shifted mV")
		ax[1].set_title("Y-strips, event {:d}".format(evno))
		#ax[0].set_ylim([-10, 200])
		#ax[1].set_ylim([-10, 200])

		if(show):
			plt.show()

		return fig, ax
	

	def plot_tile(self, evno, time=None, window = 10):

		evdf = self.create_df_from_event(evno)
		
		if time is None:
			max = 0
			max_channel = 0
			for channel in evdf["Channel"]:
				if np.max(evdf["Data"][channel]) > max:
					max = np.max(evdf["Data"][channel])
					max_channel = channel
			time = np.argmax(evdf["Data"][max_channel])

		plt.figure()
		for channel in evdf["Channel"]:
			type = self.get_channel_type(channel)
			if type == "dummy": continue
			
			channel_max = np.max((evdf["Data"][channel])[time-window:time+window])
			if type == "x":
				plt.fill_between([-2,-1,0,1,2],[0,0,0,0,0],[10,10,10,10,10])
				#plt.fill_between([evdf["ChannelPos"][channel]+6],[0],[40], alpha=np.max((evdf["Data"][channel])[time-window:time+window])/max)
			elif type == "y":
				plt.vlines(5,0,10)
				#plt.fill_between([0,40],[evdf["ChannelPos"][channel]], [evdf["ChannelPos"][channel]+6], alpha=np.max((evdf["Data"][channel])[time-window:time+window])/max)
		plt.show()
		


		
	
	def plot_event_ch(self, evno, ch, ax = None, show=True):
		ev = self.create_df_from_event(evno)
		ev_ch = ev[ev["Channel"] == ch]
		wave = list(ev_ch["Data"])[0]
		nsamp = len(wave)
		times = np.arange(0, nsamp*self.dT, self.dT)

		if(ax is None):
			fig, ax = plt.subplots(figsize=(12, 8))

		ax.plot(times, wave, label=str(ch))
		ax.set_xlabel("time (us)")
		ax.set_ylabel("adc counts")

		if(show):
			ax.set_title("Channel " + str(ch))
			plt.show()
		return fig, ax
	

	def plot_stds(self, ax=None, show=True):
		if(len(self.noise_df.index) == 0):
			print("No STD information present. Generating now:")
			self.calculate_stds()


		channels = self.df.iloc[0]["Channels"]

		xstrips = {"ch": [], "std": []}
		ystrips = {"ch": [], "std": []}
		dummies = {"ch": [], "std": []}

		for i, row in self.noise_df.iterrows():
			ch = row["Channel"]
			typ = self.get_channel_type(ch)
			if(typ == "x"):
				xstrips["ch"].append(ch)
				xstrips["std"].append(row["STD"])
			elif(typ == "y"):
				ystrips["ch"].append(ch)
				ystrips["std"].append(row["STD"])
			else:
				dummies["ch"].append(ch)
				dummies["std"].append(row["STD"])
		
		
		if(ax is None):
			fig, ax = plt.subplots()

		xmean = self.ADC_to_ENC(np.mean(xstrips["std"]))
		ymean = self.ADC_to_ENC(np.mean(ystrips["std"]))
		dmean = self.ADC_to_ENC(np.mean(dummies["std"]))
		ax.scatter(xstrips["ch"], xstrips["std"], label="X Strips: {:.1f} e- mean".format(xmean), s=100)
		ax.scatter(ystrips["ch"], ystrips["std"], label="Y Strips: {:.1f} e- mean".format(ymean), s=100)
		ax.scatter(dummies["ch"], dummies["std"], label="dummies: {:.1f} e- mean".format(dmean), s=100)
		ax.set_xlabel("Channel number")
		ax.set_ylabel("STD [ADC]")
		ax.set_title("Cryo ASIC Noise by Channel")
		ax.legend()
		ax.grid(False)

		axENC = ax.twinx()
		ENCLim = self.ADC_to_ENC(ax.get_ylim())
		axENC.set_ylim(ENCLim[0], ENCLim[1])
		axENC.set_ylabel("ENC [e^-]")
		axENC.grid(False)

		if(show):
			plt.show()
		
		return fig, ax
	



	def plot_stds_strip_position(self, ax=None, show=True):
		if(len(self.noise_df.index) == 0):
			print("No STD information present. Generating now:")
			self.calculate_stds()


		channels = self.df.iloc[0]["Channels"]

		xstrips = {"pos": [], "std": []}
		ystrips = {"pos": [], "std": []}
		dummies = {"pos": [], "std": []}

		for i, row in self.noise_df.iterrows():
			ch = row["Channel"]
			typ = self.get_channel_type(ch)
			if(typ == "x"):
				xstrips["pos"].append(self.get_channel_pos(ch)[1])
				xstrips["std"].append(row["STD"])
			elif(typ == "y"):
				ystrips["pos"].append(self.get_channel_pos(ch)[0])
				ystrips["std"].append(row["STD"])
			else:
				dummies["pos"].append(self.get_channel_pos(ch)[0])
				dummies["std"].append(row["STD"])
		
		
		if(ax is None):
			fig, ax = plt.subplots()

		xmean = self.ADC_to_ENC(np.mean(xstrips["std"]))
		ymean = self.ADC_to_ENC(np.mean(ystrips["std"]))
		dmean = self.ADC_to_ENC(np.mean(dummies["std"]))
		ax.scatter(xstrips["pos"], xstrips["std"], label="X Strips: {:.1f} e- mean".format(xmean), s=100)
		ax.scatter(ystrips["pos"], ystrips["std"], label="Y Strips: {:.1f} e- mean".format(ymean), s=100)
		ax.scatter(dummies["pos"], dummies["std"], label="dummies: {:.1f} e- mean".format(dmean), s=100)
		ax.set_xlabel("Strip position [mm]")
		ax.set_ylabel("STD [ADC]")
		ax.set_title("Cryo ASIC Noise by Channel")
		ax.legend()
		ax.grid(False)

		axENC = ax.twinx()
		ENCLim = self.ADC_to_ENC(ax.get_ylim())
		axENC.set_ylim(ENCLim[0], ENCLim[1])
		axENC.set_ylabel("ENC [e^-]")
		axENC.grid(False)

		if(show):
			plt.show()
		
		return fig, ax

				

