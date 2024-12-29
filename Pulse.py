import numpy as np 
import Utilities as Util
import matplotlib.pyplot as plt 
from scipy.signal import find_peaks
import sys


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


    #a simple peak finder that ignores masked regions. 
    #It finds peaks of both polarities. 
    def find_peaks(self, width=None, thresh=None):
        if(thresh == None):
            #get a quick baseline noise estimate
            bl_window = [int(self.config["baseline"][0]*self.config["sampling_rate"]), int(self.config["baseline"][1]*self.config["sampling_rate"])]
            std = np.std(self.wav[bl_window[0]:bl_window[1]])
            thresh = self.config["high_threshold"]*std
        if(width == None):
            width = int(self.config["pt"]*self.config["sampling_rate"]) + 1

        thresh = float(thresh) #comes in as ndarray secretly 
        width = int(width) #comes in as ndarray secretly
        distance = width*3

        #collective for both polarities
        all_pulses = []
        all_properties = {}
        #POSITIVE POLARITY
        temp_pulses, properties = find_peaks(self.wav, height=thresh, prominence=thresh, distance=distance, wlen=width)

        #remove any peaks that are in the masked region
        igs_us = self.config["ignore_regions"]
        igs_s = [[int(ig[0]*self.config["sampling_rate"]), int(ig[1]*self.config["sampling_rate"])] for ig in igs_us]
        masked_pulses = []
        masked_properties = {}
        for key in properties:
            masked_properties[key] = []

        
        for i, tp in enumerate(temp_pulses):
            mask = False
            for ig in igs_s:
                if(tp > ig[0] and tp < ig[1]):
                    mask = True
            if(mask == False):
                masked_pulses.append(tp)
                for key in properties:
                    masked_properties[key].append(properties[key][i])
    

        #add a few properties to this dict that are not returned by find_peaks
        #first, the number of samples above threshold surrounding the peak
        masked_properties["widths"] = []
        masked_properties["left_crossing"] = []
        masked_properties["right_crossing"] = []
        for i, tp in enumerate(masked_pulses):
            j = tp
            k = tp
            while(self.wav[j] > thresh):
                j += 1
                if(j >= len(self.wav)):
                    break
            j -= 1 #make sure not to count the sample that is below thres
            while(self.wav[k] > thresh):
                k -= 1
                if(k < 0):
                    break
            k += 1 #make sure not to count the sample that is below thres
            masked_properties["widths"].append(j - k)
            masked_properties["left_crossing"].append(k)
            masked_properties["right_crossing"].append(j)


        #update the collection before moving to negative polarity
        for key in masked_properties:
            all_properties[key] = masked_properties[key]
        all_pulses = masked_pulses

        #NEGATIVE POLARITY
        temp_pulses, properties = find_peaks(-1*np.array(self.wav), height=thresh, prominence=thresh, distance=distance, wlen=width)

        #remove any peaks that are in the masked region
        igs_us = self.config["ignore_regions"]
        igs_s = [[int(ig[0]*self.config["sampling_rate"]), int(ig[1]*self.config["sampling_rate"])] for ig in igs_us]
        masked_pulses = []
        masked_properties = {}
        for key in properties:
            masked_properties[key] = []

        
        for i, tp in enumerate(temp_pulses):
            mask = False
            for ig in igs_s:
                if(tp > ig[0] and tp < ig[1]):
                    mask = True
            if(mask == False):
                masked_pulses.append(tp)
                for key in properties:
                    masked_properties[key].append(properties[key][i])
    

        #add a few properties to this dict that are not returned by find_peaks
        #first, the number of samples above threshold surrounding the peak
        masked_properties["widths"] = []
        masked_properties["left_crossing"] = []
        masked_properties["right_crossing"] = []
        for i, tp in enumerate(masked_pulses):
            j = tp
            k = tp
            while(self.wav[j] < thresh):
                j += 1
                if(j >= len(self.wav)):
                    break
            j -= 1 #make sure not to count the sample that is below thres
            while(self.wav[k] < thresh):
                k -= 1
                if(k < 0):
                    break
            k += 1 #make sure not to count the sample that is below thres
            masked_properties["widths"].append(j - k)
            masked_properties["left_crossing"].append(k)
            masked_properties["right_crossing"].append(j)

        #update the collection before moving to negative polarity
        for key in masked_properties:
            all_properties[key] += masked_properties[key]
        all_pulses += masked_pulses


        #for debugging
        """
        if(len(masked_pulses) > 0):
            tp = masked_pulses[0]
            fig, ax = plt.subplots()
            ax.plot(self.wav, 'ko-')
            ax.axhline(y=thresh, color='r')
            ax.axhline(y=-1*thresh, color='r')
            ax.scatter(masked_pulses, self.wav[masked_pulses], s=200)
            ax.set_xlim([min(masked_pulses) - 50, max(masked_pulses) + 50])
            plt.show()
        """
        
        
        return all_pulses, all_properties

        

        

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

        #find the precise baseline leading up to the pulse. 
        #uses first 25% of this truncated pulse waveform 
        baseline = np.mean(self.wav[:int(0.25*len(self.wav))])
        #baseline subtract
        self.wav = self.wav - baseline
        
        #find the integral of the pulse using this peak time
        #and integrating over a specified asymmetric window.
        #We will calculate the positive and negative integrals
        #relative to baseline. 
        integ_window_us = self.config["integ_window"]
        integ_window_s = [int(integ_window_us[0] * self.config["sampling_rate"]), int(integ_window_us[1] * self.config["sampling_rate"])]
        integ_wave = self.wav[max_idx + integ_window_s[0]:max_idx + integ_window_s[1]]
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
        #find the threshold crossing for arrival
        for i, samp in enumerate(wav_pol):
            if(samp > thr_a):
                self.d["t_arrival"] = (i + self.idx_start)/self.config["sampling_rate"]
                break
        #find the threshold crossing on both sides for the width
        i = np.argmax(wav_pol)
        j = k = i 
        while(wav_pol[j] > thr_w):
            j += 1
            if(j >= len(wav_pol)):
                break
        while(wav_pol[k] > thr_w):
            k -= 1
            if(k < 0):
                break
        #make sure the sample after the threshold is not counted
        j -= 1
        k += 1
        self.d["width"] = (j - k)/self.config["sampling_rate"]
        if(self.d["width"] == 0):
            self.d["asymmetry"] = 0
        else:
            self.d["asymmetry"] = (j - i)/(j - k)/self.config["sampling_rate"]
    

    #for debugging
    def plot_pulse(self):
        fig, ax = plt.subplots()
        ax2 = ax.twiny()
        ax.plot(self.wav, 'ko-')
        ax2.plot(range(self.idx_start, self.idx_start + len(self.wav)), self.wav, 'ko-')
        print(self.d)
        plt.show()




        
