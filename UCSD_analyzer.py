import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import CryoAsicFile, CryoAsicAnalysis 
from os.path import exists
from scipy.signal import periodogram
import scipy.stats
from matplotlib.colors import LogNorm
plt.rcParams.update({'font.size': 22})
plt.rcParams['lines.linewidth'] = 2

def heatmap(data, row_labels, col_labels, ax=None,
	cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A atplotlib.axes.Axesinstance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to atplotlib.Figure.colorbar  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to mshow
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation='vertical',
             ha="left", va='center',
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    #return im, cbar
    return im

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
		 textcolors=("black", "white"),
		 threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        atplotlib.ticker.Formatter  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to extused to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a extfor each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def plot_correlation(FEMB, fig_dir, FE_setting):
	correlation = np.zeros((64,64))
	wfs = []
	for j in range(64):
		wf = FEMB.get_wave(0, j)
		wfs.append(wf)
		for k in range(64):
			wf1 = FEMB.get_wave(0, k)
			correlation[j, k] = scipy.stats.pearsonr(wf, wf1)[0]
	adcs = np.array(wfs)
	#implement the correlation matrix calculation used in Aldo's test
	adcs0 = adcs - adcs.mean(axis=-1, keepdims=True)
	mcorr = np.corrcoef(np.swapaxes(adcs0, 0, 1).reshape(64, -1))
	fig, ax = plt.subplots(figsize=(32,32),  clear=True)
	labels = [f'ch{ch:02}' for ch in range(64)]
	im = heatmap(mcorr, labels, labels, ax=ax, cmap='RdBu', vmax=1, vmin=-1)
	texts = annotate_heatmap(im, valfmt='{x:.1f}')
	fig.savefig('./{}/correlation_{}.png'.format(fig_dir, FE_setting))
	
def plot_sample(FEMB, fig_dir, FE_setting):
	fig, ax = plt.subplots(figsize=(15, 12))
	for j in range(64):
		wf = FEMB.get_wave(0, j)    
		wf = wf[:200]
		if j < 3:
			ax.plot(np.arange(len(wf))*0.5, wf)
	ax.set_xlabel('Time ($\mu s$)')
	ax.set_ylabel('ADC')
	fig.savefig('./{}/sample_wfs_{}.png'.format(fig_dir, FE_setting))

def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def plot_fft(FEMB, fig_dir, FE_setting):
	from scipy.fft import fft, fftfreq
	N = len(FEMB.get_wave(0, 0))
	T = 1.0/2000000
	#x = np.linspace(0.0, N*T, N, endpoint=False)
	fig, ax = plt.subplots(figsize=(15, 12))
	freqs = []
	for j in range(FEMB.nevents_total):
		wf = FEMB.get_wave(j, 10)
		yf = fft(wf)    
		freqs.append(2.0/N * np.abs(yf[1:N//2]))
	xf = fftfreq(N, T)[:N//2]
	ax.semilogx(xf[1:N//2], np.mean(freqs, axis=0), label=FE_setting)
	freq_av = movingaverage(np.mean(freqs, axis=0), 20)
	ax.semilogx(xf[1:len(freq_av)+1], freq_av, label='running average')
	ax.set_xlabel('Frequency (Hz)')
	ax.set_ylabel('Amplitude')
	ax.set_xlim(100, np.max(xf[1:N//2]))
	ax.set_ylim(0, np.max(np.mean(freqs, axis=0)))
	ax.legend()
	fig.savefig('./{}/fft_{}.png'.format(fig_dir, FE_setting))

def plot_std(FEMB, fig_dir, FE_setting):
	stds = []
	fig, ax = plt.subplots(2, figsize=(15, 12))
	for j in range(64):
		wf = FEMB.get_wave(0, j)
		stds.append(np.std(wf))
	ax[0].plot(np.arange(64), stds)
	ax[0].set_xlabel('Channel')
	ax[0].set_ylabel('STD (ADC)')
	ax[0].set_xlim(0,64)
	ax[1].hist(stds, bins = np.linspace(0, np.max(stds)+1, 10), histtype='step', color='blue', linewidth=2)
	ax[1].set_xlabel('STD (ADC)')
	ax[1].set_xlim(0, np.max(stds)+1)
	fig.savefig('./{}/stds_{}.png'.format(fig_dir, FE_setting))

def corr_correction(FEMB, fig_dir, FE_setting):
	single_wfs = []
	stds = []
	stds_corrected = []
	for j in range(64):
		wf = FEMB.get_wave(0, j)
		stds.append(np.std(wf))
		single_wfs.append(wf)

	for j in range(64):
		wf = FEMB.get_wave(0, j)
		wf = wf - np.mean(single_wfs, axis=0)
		stds_corrected.append(np.std(wf))
	fig, ax = plt.subplots(figsize=(15, 12))
	ax.hist(stds, bins = np.linspace(0, np.max(stds)+1, 10), histtype='step', color='blue', label='Raw', linewidth=2)
	ax.hist(stds_corrected, bins = np.linspace(0, np.max(stds)+1, 10), histtype='step', color='red', label='Subtract average', linewidth=2)
	ax.set_xlim(0, np.max(stds)+1)
	ax.set_xlabel('STD (ADC)')
	ax.legend()
	fig.savefig('./{}/stds_corretion_{}.png'.format(fig_dir, FE_setting))

def compare_fft(FEMB, fig_dir, FE_setting):
	from scipy.fft import fft, fftfreq
	N = len(FEMB.get_wave(0, 0))
	T = 1.0/2000000
	x = np.linspace(0.0, N*T, N, endpoint=False)
	fig, ax = plt.subplots(figsize=(15, 12))
	freq_matrix = []
	
	for i in [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]:
		freqs = []
		for j in range(FEMB.nevents_total):
			wf = FEMB.get_wave(j, i)
			wf = (wf - np.mean(wf))/np.std(wf)
			yf = fft(wf)    
			freqs.append(2.0/N * np.abs(yf[1:N//2]))
		freq_matrix.append(freqs)
	xf = fftfreq(N, T)[:N//2]
	caps = [7, 10, 15, 22, 27, 39, 47, 68, 470, 390, 330, 270, 200, 150, 100, 82]
	for i in range(8):
		freqs = freq_matrix[i]
		ax.plot(xf[1:N//2], np.mean(freqs, axis=0), label='{}pF'.format(caps[i]))
	ax.set_xlabel('Frequency (Hz)')
	ax.set_ylabel('Amplitude')
	ax.set_xlim(0, np.max(xf[1:N//2]))
	ax.set_ylim(0, 0.02)
	ax.legend()
	fig.savefig('./{}/ch_fft_{}.png'.format(fig_dir, FE_setting))

def plot_std(FEMB, fig_dir, FE_setting):
	stds = []
	caps = [7, 10, 15, 22, 27, 39, 47, 68, 470, 390, 330, 270, 200, 150, 100, 82]
	fig, ax = plt.subplots(2, figsize=(15, 12))
	for j in range(64):
		wf = FEMB.get_wave(0, j)
		stds.append(np.std(wf))
	ax[0].plot(np.arange(64), stds, color='blue', linewidth=2)
	ax[0].set_xlabel('Channel')
	ax[0].set_ylabel('STD (ADC)')
	ax[0].set_xlim(0,64)
	stds_group = np.array(stds).reshape(16,4)
	for i in range(16):
		for j in range(4):
			ax[1].scatter(caps[i], stds_group[i, j], color='blue')
	ax[1].set_xlabel('Capacitance (pF)')
	ax[1].set_ylabel('Noise (ADC)')
	ax[1].set_xlim(0, 500)
	ax[1].set_ylim(bottom=0)
	fig.savefig('./{}/stds_{}.png'.format(fig_dir, FE_setting))

if __name__ == '__main__':
	board_id = 'SN01'
	condition = 'RT'
	test_type = 'baseline'
	FE_setting = 'FE924'
	FE_settings = []
	for fe in range(896, 960, 4):
		if test_type == 'pulse':
			FE_settings.append('FE{}'.format(fe+1))
		else:
			FE_settings.append('FE{}'.format(fe))
	import glob
	files = glob.glob('/scratch/CRYO_ASIC/230317/SN01_RT/*.dat')
	print(FE_settings)
	for fbinary in files:
		fhdf = fbinary.replace('.dat', '.h5')
		FE_setting = fbinary[fbinary.find('FE'): fbinary.find('FE')+5]
		import os
		fig_dir = './{}_{}_{}'.format(board_id, condition, test_type)
		if not exists(fig_dir):
			os.mkdir(fig_dir)
		if not exists(fbinary):
			print('Binary file does NOT exist!')
			exit()
		if FE_setting not in FE_settings:
			continue
		elif not exists(fhdf):
			parsefile = CryoAsicFile.CryoAsicFile(fbinary, './config/channel_map_template.txt', './config/tile_map_template.txt')
			parsefile.load_raw_data()
			parsefile.group_into_pandas()
			parsefile.save_to_hdf5(fhdf)
		else:
			print('Binary file has already been converted to hdf5 file, move forward to analysis.')

	#baseline: [mintime, maxtime]
	#pulse_threshold: thresh in adc counts
	#sampling_rate: MHz sampling rate
	#mv_per_adc: rough conversion, constant, assuming linearity
		config = {}
		config['pulse_threshold'] = 0
		config['sampling_rate'] = 2
		config['mv_per_adc'] = 0.25
		FEMB = CryoAsicAnalysis.CryoAsicAnalysis(fhdf, config)
		print(FEMB.nevents_total)
		plot_correlation(FEMB, fig_dir, FE_setting)
		plot_sample(FEMB, fig_dir, FE_setting)
		plot_std(FEMB, fig_dir, FE_setting)
		plot_fft(FEMB, fig_dir, FE_setting)
		corr_correction(FEMB, fig_dir, FE_setting)
	
