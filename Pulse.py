import numpy as np 
import Utilities as Util
import matplotlib.pyplot as plt 


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
        fig, ax = plt.subplots()
        ax.plot(self.integ_idx, self.integ)
        plt.show()
        

    #populate the reduced quantities with the default values
    def populate_rqs(self, config, wvfm, i, params):

        l_edge = params["left_ips"][i] - params["widths"][i]/2
        r_edge = params["right_ips"][i] + 2*config["pt"]*config["sampling_rate"] # Adding two samples to the right edge to ensure the right edge is included in the sum

        l_samp = int(l_edge*config["sampling_rate"]) # Left and right edge in samples for ease of use below
        r_samp = int(r_edge*config["sampling_rate"])
        
        p = np.argmax(wvfm[l_samp:r_samp])+l_samp 

        self.d["tmax"] = p / config["sampling_rate"] # Note the different units from p
        self.d["width"] = params["widths"][i] / config["sampling_rate"] # Getting width in units of samples->us taking the default width at half prominence 
        self.d["max"] = params["peak_heights"][i] 
        self.d["min"] = np.min(wvfm[l_samp:r_samp])  
        self.d["tmin"] = (np.argmin(wvfm[l_samp:r_samp])+l_samp) / config["sampling_rate"] 

        self.d["t_arrival"] = self.d["tmax"] # For now without light, we'll just take arrival time to be the collection time, but could imagine later making this be drift time


        #self.d["q_collection"] = optimum_filter(p, wvfm) # This is where one could imagine eventually implimenting an optimum filter analysis
        #self.d["q_collection"] = self.Trap(l_edge, r_edge, wvfm, G=config[r_edge-l_edge], L = config["pt"]) Or maybe a trapazoid filter 
        self.d["q_collection"] = ADC_to_ENC(np.trapz(y = wvfm[l_samp:r_samp], dx = 1/config["sampling_rate"])) # For now just integrate the waveform in the region of the pulse
        self.d["q_induction"] = ADC_to_ENC(np.trapz(y = wvfm[p+int(config["pt"]*config["sampling_rate"]):r_samp], dx = 1/config["sampling_rate"])) # Integrate the waveform after the pulse

