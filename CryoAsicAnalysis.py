import sys
import pickle
import os
import yaml
import numpy as np 
import math
import matplotlib.pyplot as plt 
import pandas as pd
from scipy.signal import periodogram
from scipy.optimize import curve_fit
from matplotlib.colors import LogNorm
from scipy import signal




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
		

		self.config = None #has global analysis config dictionary contents
		self.chmap = None #has the channel and tile mappings. 
		self.load_config(config) #loads both of the above dictionaries


		print("loading the waveform dataframe from  " + self.infile)
		self.df = pickle.load(open(self.infile, "rb"))[0] #0th element of the list is dataframe. 
		print("Done loading")

		self.nevents_total = len(self.df.index)

		self.sf = config["sampling_rate"] #MHz "sf": sampling_frequency
		self.dT = 1.0/self.sf
		self.nsamples = len(self.df.iloc[0]["Data"][0])
		self.times = np.arange(0, self.nsamples*self.dT, self.dT)


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
				return (tile_pos[0], tile_pos[1] + local_pos*pitch)
			elif(local_ch in self.chmap[asic]["ystrips"]):
				local_pos = float(self.chmap[asic]["xstrips"][local_ch])
				return (tile_pos[0] + local_pos*pitch, tile_pos[1])
			else:
				return tile_pos #this is a dummy capacitor
		else:
			print("Asic {:d} not found in the configuration file channel map".format(asic))
			return None

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
			ser = pd.Series()
			ser["Channel"].append(ch)
			ser["Data"].append(ev["Data"][i])
			ser["ChannelType"].append(self.get_channel_type(ch))
			ser["ChannelPos"].append(self.get_channel_pos(ch))

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
	def plot_waves(self, evno, chs_to_plot = []):
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
				ax.plot(times, waves[i] + curshift, label=str(chs[i]))
			#curshift += adc_shift

		ax.set_xlabel('time (us)')
		ax.set_ylabel("channel shifted adc counts")


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

	#for every channel, calculate a PSD using an event-by-event
	#calculation, averaging over all events. Save these periodogram
	#data in a dataframe. 
	def calculate_avg_psds(self):

		chs = self.df.iloc[0]["Channels"]
		nevents = len(self.df.index) #looping through all events

		for ch in chs:
			pxx_tot = None
			freqs = None
			avg_event_counter = 0 #number of events over which the avg is calculated
			for i in range(nevents):
				ev = self.df.iloc[i]
				wave = ev["Data"][ch]
				#ignore events with glitches!!! note this is temporary for our early datasets.
				#if any sample is above 50 mV, ignore the event. 
				if("noise" in self.infile and max(wave, key=abs) > 500):
					print("skipped event " + str(i))
					continue

				wave = [_*self.config["mv_per_adc"]/1000. for _ in wave] #putting ADC units into volts
				
				fs, pxx = periodogram(wave, self.sf*1e6)
				if(pxx_tot is None):
					pxx_tot = pxx 
					freqs = fs
				else:
					pxx_tot = [pxx[_] + pxx_tot[_] for _ in range(len(pxx))]
				avg_event_counter += 1

			pxx_tot = [_/float(avg_event_counter) for _ in pxx_tot]

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


	def calculate_stds(self, window=[0,-1]):
		chs = self.df.iloc[0]["Channels"]
		nevents = len(self.df.index) #looping through all events

		for ch in chs:
			if(ch == self.config["key_channel"]):
				continue
			all_samples = []
			for i in range(nevents):
				ev = self.df.iloc[i]
				wave = list(ev["Data"][ch][window[0]:window[1]]) #ADC counts
				all_samples = all_samples + wave

			self.noise_df.at[ch, "STD"] = np.std(all_samples)
			self.noise_df.at[ch, "Channel"] = ch


	def plot_stds(self):

		channels = self.df.iloc[0]["Channels"]
		nevents = len(self.df.index)
		
		try:
			stds = self.noise_df["STD"]
		except:
			print("No STD information present. Generating now:")
			self.calculate_stds()
		
		fig, ax = plt.subplots()
		ax.plot(channels, self.noise_df["STD"])
		ax.set_xlabel("Channel number [ ]")
		ax.set_ylabel("STD [ADC]")
		ax.set_title("Cryo ASIC Noise by Channel")

		axENC = ax.twinx()
		ENCLim = self.ADC_to_ENC(ax.get_ylim())
		axENC.set_ylim(ENCLim[0], ENCLim[1])
		axENC.set_ylabel("ENC [e^-]")



	def plot_stds_strip_position(self, show=True):
		if(len(self.noise_df.index) == 0):
			print("No STD information present. Generating now:")
			self.calculate_stds()
		
		xstrips = {"pos": [], "std": []}
		ystrips = {"pos": [], "std": []}
		caps = {"pos": [], "std": []}
		for i, row in self.noise_df.iterrows():
			ch = row["Channel"]
			if(ch in self.config["dead_channels"]):
				continue

			typ = self.get_channel_type(ch)
			if(typ == "x"):
				xstrips["pos"].append(self.get_channel_pos(ch)[1])
				xstrips["std"].append(row["STD"])
			elif(typ == "y"):
				ystrips["pos"].append(self.get_channel_pos(ch)[0])
				ystrips["std"].append(row["STD"])
			else:
				if(len(caps["pos"]) == 0);
					caps["pos"].append(0)
				else:	
					caps["pos"].append(caps["pos"][-1]+1)

				caps["std"].append(row["STD"])

		
		fig, ax = plt.subplots()
		axENC = ax.twinx()
		ax.scatter(xstrips["pos"], xstrips["std"], label="X Strips", s=100)
		ax.scatter(ystrips["pos"], ystrips["std"], label="Y Strips", s=100)
		ax.scatter(caps["pos"], caps["std"], label="Caps", s=100)
		ax.set_xlabel("Strip Position [mm]")
		ax.set_ylabel("STD [ADC]")
		ENCLim = self.ADC_to_ENC(ax.get_ylim(), self.config["gain"], self.config["pt"])
		axENC.set_ylim(ENCLim[0], ENCLim[1])
		axENC.set_ylabel("ENC [e^-]")
		ax.grid(False)
		axENC.grid(False)
		ax.legend()

		if(show):
			plt.show()

		return fig, ax, axENC 

				




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










				
			
				












