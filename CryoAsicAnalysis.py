import sys
import os
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from scipy.signal import periodogram
from scipy.optimize import curve_fit
from matplotlib.colors import LogNorm
from scipy import signal



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



class CryoAsicAnalysis:
	#takes a "config" dict which has analysis and data parameters
	def __init__(self, infile, config):
		self.infile = infile
		if(self.infile[-2:] != "h5" and self.infile[-4:] != "hdf5"):
			print("Input file to CryoAsicEventViewer is not an hdf5 file: " + self.infile)
			return


		#entries expected:
		#baseline: [mintime, maxtime]
		#pulse_threshold: thresh in adc counts
		#sampling_rate: MHz sampling rate
		#mv_per_adc: rough conversion, constant, assuming linearity
		self.config = config 

		print("loading hdf5 file " + self.infile)
		self.df = pd.read_hdf(self.infile, key='raw')
		print("Done loading")

		self.nevents_total = len(self.df.index)

		self.sf = config["sampling_rate"] #MHz "sf": sampling_frequency
		self.dT = 1.0/self.sf
		self.nsamples = len(self.df.iloc[0]["Data"][0])
		self.times = np.arange(0, self.nsamples*self.dT, self.dT)


		self.noise_df = pd.DataFrame(columns=["Channel", "Freqs", "PSD", "STD"]) #rows are channels, columns are noise information
		self.noise_df.set_index("Channel")


		self.corr_mat = None

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
		self.noise_df.to_hdf(self.infile[:-3]+"_noisedf.h5", key='noise')

	def load_noise_df(self):
		if(os.path.exists(self.infile[:-3]+"_noisedf.h5")):
			self.noise_df = pd.read_hdf(self.infile[:-3]+"_noisedf.h5")
			return True
		else:
			return False



	#strips and dummy capacitors are populated
	#on these boards. this function will tell if
	#if it is strip or cap
	def is_channel_strip(self, ch):
		chs = self.df["Channels"].iloc[0]
		chidx = chs.index(ch)
		p = self.df["ChannelPositions"].iloc[0]
		thisch_pos = p[chidx]
		if(thisch_pos[0] >= 51):
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
		looper = tqdm(self.df.iterrows(), total=len(self.df.index))
		for i, row in looper:
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
		
	def get_scope(self, evno):
		if(evno < 0):
			evno = 0
		if(evno > self.nevents_total):
			print("That event is not in the dataframe: " + str(evno))
			return

		ev = self.df.iloc[evno]
		return ev["Scope"]

	#for every channel, calculate a PSD using an event-by-event
	#calculation, averaging over all events. Save these periodogram
	#data in a dataframe. 
	def calculate_avg_psds(self):

		chs = self.df.iloc[0]["Channels"]
		nevents = len(self.df.index) #looping through all events
		looper = tqdm(chs, desc="Calculating avg PSD on channel...")

		for ch in looper:
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
		looper = tqdm(chs, desc="Calculating stds on each channel...")

		for ch in looper:
			if(ch == self.config["key_channel"]):
				continue
			all_samples = []
			for i in range(nevents):
				ev = self.df.iloc[i]
				wave = list(ev["Data"][ch][window[0]:window[1]]) #ADC counts
				all_samples = all_samples + wave

			def gausfit(x, a, b, c):
				return a*np.exp(-(x - c)**2/(2*b**2))

			#estimate of mean, std, and amp
			binwidth=0.5 #ADC counts
			bins = np.arange(min(all_samples), max(all_samples))
			n, bins = np.histogram(all_samples, bins, density=1)
			bc = (bins[:-1] + bins[1:])/2
			guess = [max(n), np.std(all_samples), np.median(all_samples)]
			try:
				popt, pcov = curve_fit(gausfit, bc, n, p0=guess)
				self.noise_df["STD"].iloc[ch] = popt[1] #ADC counts
			except:
				print("Fit failed..., just doing regular std")
				self.noise_df["STD"].iloc[ch] = np.std(all_samples)

			"""		
			fig, ax = plt.subplots(figsize=(8, 5))
			ax.hist(all_samples, bins, density=1)
			#ax.plot(bc, gausfit(bc, *popt))
			plt.show()
			"""

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
		looper = tqdm(range(nevents), desc="Calculating cross correlation matrix")
		skip_event = False #flag to skip events if there is a glitch
		for evno in looper:
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






				
			
				












