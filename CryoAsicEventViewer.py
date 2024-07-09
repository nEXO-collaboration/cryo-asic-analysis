import sys
import os
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from matplotlib.widgets import Slider, Button, RadioButtons


#load a viewer using an infile that is an 
#hdf5 file containing a waveform dataframe, processed
#from binary by the CryoAsicFile class

class CryoAsicEventViewer:
	def __init__(self, infile, config = {}):
		self.infile = infile
		if(self.infile[-2:] != "h5" and self.infile[-4:] != "hdf5"):
			print("Input file to CryoAsicEventViewer is not an hdf5 file: " + self.infile)
			return

		print("loading hdf5 file " + self.infile)
		self.df = pd.read_hdf(self.infile, key='raw')
		print("Done loading")

		self.nevents_total = len(self.df.index)

		self.sf = config["sampling_rate"] #MHz "sf": sampling_frequency
		self.dT = 1.0/self.sf
		self.config = config



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


	#plots the event in the same way that the CRYO ASIC GUI
	#event viewer plots, with a 2D greyscale hist using the
	#*channel numbers as the y axis bins and the time on x axis. 
	#this is distinct from other plots in that usually the positions
	#of the strips are used as the y axis; in this way, this is more
	#what the ASIC sees, ordering based on the asic channel IDs. 
	def plot_event_rawcryo(self, evno):
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


		#functions for controlling slider bars
		axvmin = plt.axes([0.25, 0.025, 0.65, 0.01], facecolor='green')
		axvmax = plt.axes([0.25, 0.04, 0.65, 0.01], facecolor='green')
		vminbar = Slider(axvmin, 'min adc counts', 0, 4096, valinit=img.min())
		vmaxbar = Slider(axvmax, 'max adc counts', 0, 4096, valinit=img.max())
		def update(val):
			vmin = vminbar.val 
			vmax = vmaxbar.val 
			heat.set_clim(vmin, vmax)
			fig.canvas.draw_idle()
		vminbar.on_changed(update)
		vmaxbar.on_changed(update)
		reset = plt.axes([0.0, 0.025, 0.1, 0.04])
		button = Button(reset, 'Reset', color='green', hovercolor='0.975')
		def reset(event):
			vminbar.reset()
			vmaxbar.reset()
		button.on_clicked(reset)

		plt.show()

	#the self.df structure is not great for event level
	#manipulation unfortunately. For example, the column
	#"ChannelPositions" is a list of all channel positions
	#that are involved in the event. If you would like to select
	#out a mask of some channel positions, it is more convenient for
	#the channels to be df rows. 
	def create_df_from_event(self, evno):
		evdf = pd.DataFrame()
		ev = self.df.iloc[evno]
		for i, ch in enumerate(ev["Channels"]):
			ser = pd.Series()
			ser["Channel"] = ch 
			ser["Data"] = ev["Data"][i]
			ser["ChannelX"] = ev["ChannelPositions"][i][0]
			ser["ChannelY"] = ev["ChannelPositions"][i][1]
			ser["ChannelType"] = ev["ChannelTypes"][i]
			evdf = pd.concat([evdf, ser.to_frame().transpose()], ignore_index=True)

		return evdf



	#plots the event with a 2D hist where each tile has two subplots:
	#one for the X strips and one for the Y strips. Time on x axis, 
	#local channel position on Y axis. Finds all tiles and generates
	#subplots for each. For the moment, only does strips and not caps
	def plot_event_xysep(self, evno, show=True):
		if(evno < 0):
			evno = 0
		if(evno > self.nevents_total):
			print("That event is not in the dataframe: " + str(evno))
			return

		ev = self.create_df_from_event(evno)
		nsamp = len(ev["Data"].iloc[0])
		times = np.arange(0, nsamp*self.dT, self.dT)

		#select only strips, X <= 51 is for strips and not capacitors
		xstrip_mask = (ev["ChannelType"] == 'x') & (ev["ChannelX"] <= 51)
		ystrip_mask = (ev["ChannelType"] == 'y') & (ev["ChannelX"] <= 51)
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
		for i, row in (xdf.sort_values("ChannelX")).iterrows():
			wave = row["Data"]
			xpos.append(row["ChannelX"])
			#is it a dead channel
			if(row["Channel"] in self.config["dead_channels"]):
				dead_xpos.append(row["ChannelX"])

			for j in range(len(wave)):
				xstrip_img[ch_idx][j] = wave[j]


			ch_idx += 1

		ch_idx = 0
		for i, row in (ydf.sort_values("ChannelY")).iterrows():
			wave = row["Data"]
			ypos.append(row["ChannelY"])
			#is it a dead channel
			if(row["Channel"] in self.config["dead_channels"]):
				dead_ypos.append(row["ChannelY"])

			for j in range(len(wave)):
				ystrip_img[ch_idx][j] = wave[j]

			ch_idx += 1

		pos_sort = sorted(xpos)
		pitch = abs(pos_sort[0] - pos_sort[1])

		fig, ax = plt.subplots(figsize=(12,16), nrows = 2)
		xheat = ax[0].imshow(xstrip_img, cmap='viridis', interpolation='none',\
			extent=[ min(times), max(times), max(xpos), min(xpos) - pitch],\
			aspect=0.5*(max(times)/(max(xpos) - min(xpos))))

		yheat = ax[1].imshow(ystrip_img, cmap='viridis', interpolation='none',\
			extent=[min(times), max(times), max(ypos), min(ypos) - pitch],\
			aspect=0.5*(max(times)/(max(ypos) - min(ypos))))

		xcbar = fig.colorbar(xheat, ax=ax[0])
		xcbar.set_label("ADC counts", labelpad=3)
		ax[0].set_xlabel("time (us)")
		ax[0].set_ylabel("x - strip position")
		ax[0].set_title("event number " + str(evno))

		ycbar = fig.colorbar(yheat, ax=ax[1])
		ycbar.set_label("ADC counts", labelpad=3)
		ax[1].set_xlabel("time (us)")
		ax[1].set_ylabel("y - strip position")
		ax[1].set_title("event number " + str(evno))

		#plot dead channels with lines through them
		for p in dead_xpos:
			ax[0].axhline(p-0.5*pitch, linewidth=2, color='r')
		for p in dead_ypos:
			ax[1].axhline(p-0.5*pitch, linewidth=2, color='r')




		"""
		#functions for controlling slider bars
		axvmin = plt.axes([0.25, 0.025, 0.65, 0.01], facecolor='green')
		axvmax = plt.axes([0.25, 0.04, 0.65, 0.01], facecolor='green')
		vminbar = Slider(axvmin, 'min adc counts', 0, 4096, valinit=plot_df["v"].min())
		vmaxbar = Slider(axvmax, 'max adc counts', 0, 4096, valinit=plot_df["v"].max())
		def update(val):
			vmin = vminbar.val 
			vmax = vmaxbar.val 
			heat.set_clim(vmin, vmax)
			fig.canvas.draw_idle()
		vminbar.on_changed(update)
		vmaxbar.on_changed(update)
		reset = plt.axes([0.0, 0.025, 0.1, 0.04])
		button = Button(reset, 'Reset', color='green', hovercolor='0.975')
		def reset(event):
			vminbar.reset()
			vmaxbar.reset()
		button.on_clicked(reset)
		"""
		if(show):
			plt.show()


	#This is a fancy plot where we remove the "time" domain of the data,
	#and instead plot interleaved X and Y strips together. The color of
	#the strip represents the maximum value of the waveform within the 
	#waveform buffer. 
	def plot_event_tile_maximum(self, evno):
		if(evno < 0):
			evno = 0
		if(evno > self.nevents_total):
			print("That event is not in the dataframe: " + str(evno))
			return

	#This is a fancy plot where we remove the "time" domain of the data,
	#and instead plot interleaved X and Y strips together. The color of
	#the strip represents the baseline of the waveform within the 
	#waveform buffer. Baseline is calculated as the mean of the 
	#final few microseconds, "baseline_buffer"
	def plot_event_tile_baseline(self, evno, baseline_buffer = 10):
		if(evno < 0):
			evno = 0
		if(evno > self.nevents_total):
			print("That event is not in the dataframe: " + str(evno))
			return


	#this plots the waveforms from x and y all on the same plot, 
	#overlayed, but with traces shifted relative to eachother by 
	#some number of ADC counts. if tileno is not none, it only plots
	#one tile, associated with an integer passed as argument
	def plot_event_waveforms_separated(self, evno):
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

		fig, ax = plt.subplots(figsize=(10,8))
		curshift = 0
		for i in range(nch):
			ax.plot(times, waves[i] + curshift, label=str(chs[i]))
			#curshift += adc_shift

		ax.set_xlabel('time (us)')
		ax.set_ylabel("channel shifted adc counts")


		plt.show()


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


	#given a time, return the closest sample index (int)
	def time_to_sample(self, t):
		times = range(0, len(self.df["Data"].iloc[0][0]))
		idx = (np.abs(np.asarray(times) - t)).argmin()
		return idx 


	#this plots the waveforms from x and y all on the same plot, 
	#overlayed, but with traces shifted relative to eachother by 
	#some number of ADC counts. if tileno is not none, it only plots
	#one tile, associated with an integer passed as argument
	def plot_strips_waveforms_separated(self, evno, show=True):
		if(evno < 0):
			evno = 0
		if(evno > self.nevents_total):
			print("That event is not in the dataframe: " + str(evno))
			return


		ev = self.create_df_from_event(evno)
		nsamp = len(ev["Data"].iloc[0])
		times = np.arange(0, nsamp*self.dT, self.dT)

		#select only strips, X <= 51 is for strips and not capacitors
		xstrip_mask = (ev["ChannelType"] == 'x') & (ev["ChannelX"] <= 51)
		ystrip_mask = (ev["ChannelType"] == 'y') & (ev["ChannelX"] <= 51)
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
		ch_idx = 0
		dead_chs = []
		for i, row in (xdf.sort_values("ChannelX")).iterrows():
			wave = row["Data"]
			xpos.append(row["ChannelX"])
			xchs.append(row["Channel"])
			for j in range(len(wave)):
				xstrip_img[ch_idx][j] = wave[j]*self.config["mv_per_adc"]
			if(max(xstrip_img[ch_idx]) < 50):
				dead_chs.append(xchs[-1])

			ch_idx += 1

		ch_idx = 0
		for i, row in (ydf.sort_values("ChannelY")).iterrows():
			wave = row["Data"]
			ypos.append(row["ChannelY"])
			ychs.append(row["Channel"])
			for j in range(len(wave)):
				ystrip_img[ch_idx][j] = wave[j]*self.config["mv_per_adc"]
			if(max(ystrip_img[ch_idx]) < 50):
				dead_chs.append(ychs[-1])

			ch_idx += 1

		mv_shift = 10 #number of adc counts to shift traces


		fig, ax = plt.subplots(figsize=(12,16), nrows=2)
		curshift = 0
		for i in range(len(xstrip_img)):
			if(xchs[i] in self.config["dead_channels"] or xchs[i] == self.config["key_channel"]):
				ax[0].plot(times, np.array(xstrip_img[i]) + curshift, 'k', label=str(xpos[i]))
			else:
				ax[0].plot(times, np.array(xstrip_img[i]) + curshift, label=str(xpos[i]))
			
			curshift += mv_shift

		curshift = 0
		for i in range(len(ystrip_img)):
			if(ychs[i] in self.config["dead_channels"] or ychs[i] == self.config["key_channel"]):
				ax[1].plot(times, np.array(ystrip_img[i]) + curshift, 'k', label=str(ypos[i]))
			else:
				ax[1].plot(times, np.array(ystrip_img[i]) + curshift, label=str(ypos[i]))
			
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
		return ax














