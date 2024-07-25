import math 
import numpy as np 


#this is a set of importable utilities that is common to many of the classes in this project. 

#return the unique channel ID given an asic number 
#and a channel number local to that asic
def get_unique_id(asic, ch):
    return asic*64 + ch

#inverse of the above, return the asic and channel number
#given a unique channel ID
def get_asic_and_ch(ch):
    asic = math.floor(ch/64)
    ch = ch % 64
    return asic, ch


#get the channel type from the channel number
def get_channel_type(chmap, ch):
    if(chmap is None):
        print("Channel map didn't properly load")
        return None
    local_ch = ch % 64 #the channel number on the asic level. 
    asic = math.floor(ch/64) # the asic ID that this ch corresponds to. 
    
    
    if(asic in chmap):
        if(local_ch in chmap[asic]["xstrips"]):
            return 'x'
        elif(local_ch in chmap[asic]["ystrips"]):
            return 'y'
        else:
            return 'dummy'
        
    else:
        print("Asic {:d} not found in the configuration file channel map".format(asic))
        return None
    
#returns the global position of the channel in the TPC
#using knowledge of the tile position. If it is a dummy, 
#return 0, 0 
def get_channel_pos(chmap, ch):
    if(chmap is None):
        print("Channel map didn't properly load")
        return None

    local_ch = ch % 64 #the channel number on the asic level. 
    asic = math.floor(ch/64) # the asic ID that this ch corresponds to. 
    tile_pos = chmap[asic]["tile_pos"]
    pitch = chmap[asic]["strip_pitch"] #in mm

    if(asic in chmap):
        if(local_ch in chmap[asic]["xstrips"]):
            local_pos = float(chmap[asic]["xstrips"][local_ch])
            return (tile_pos[0], tile_pos[1] + np.sign(local_pos)*(np.abs(local_pos) - 0.5)*pitch)
        elif(local_ch in chmap[asic]["ystrips"]):
            local_pos = float(chmap[asic]["ystrips"][local_ch])
            return (tile_pos[0] + np.sign(local_pos)*(np.abs(local_pos) - 0.5)*pitch, tile_pos[1])
        else:
            return tile_pos #this is a dummy capacitor
    else:
        print("Asic {:d} not found in the configuration file channel map".format(asic))
        return None
    

def is_channel_strip(chmap, ch):
    result = get_channel_type(chmap, ch)
    if(result == "dummy"):
        return False
    else:
        return True

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