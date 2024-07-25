import numpy as np 


class Cluster:
    #initialize the cluster with its reduced quantities which
    #should be parsed externally (by the instantiator) from a yaml file
    def __init__(self, rqs):
        self.rqs = rqs
        self.d = {}
        #initialize the cluster dictionary, with
        #initialze values specified in the yaml file that 
        #defines RQs. 
        for key in self.rqs:
            self.d[key] = self.rqs[key]

    