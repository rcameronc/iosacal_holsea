import os  
import pandas as pd  
from math import cos, asin, sqrt  
import csv  
from pathlib import Path  
from timeit import default_timer as timer  
from datetime import datetime
from statistics import mean as statmean


class calcDelR:
    
    """ Calculate the delta R + uncertainty for n closest points to a site.
    
    Keyword arguments:
    data -- pandas dataframe with Lat, Lon, DeltaR and DeltaRErr columns
    lon -- longitude of site
    lat -- latitude of site
    nclosest -- the number of closest sites from which to calculate deltaR
    
    Returns:
    .getMean() -- average distance from point to del R locations.
    .getMu()  -- mean of delta Rs 
    .getUncert() -- weighted delta R uncertainty 
    
    Calculations follow:
    Data Reduct. & Err. Anal. for the Phys. Sci., PR Bevington, 1969, McGraw Hill
    (as described here: http://calib.org/marine/AverageDeltaR.html)
    
    """
    
    def __init__(self, data, lat, lon, nclosest):
        
        self.lat = lat
        self.lon = lon
        self.data = data
        self.nclosest = nclosest
        self.closedict = None
        self.uncert = None
        self.denom = None
        
        
    def distance(self, lat1, lon1, lat2, lon2): 
        
        p = 0.017453292519943295  #Pi/180
        t1 = cos((lat2-lat1)*p)/2
        t2 = cos(lat1*p)*cos(lat2*p)
        t3 = (1-cos((lon2-lon1)*p))
        
        a = 0.5 - t1 + t2 * t3 / 2
        return 12742 * asin(sqrt(a)) 

    def _getClosest(self):
        
        dl = []
        for i, p in self.data.iterrows():
            ap = {
                'lat':p['Lat'],
                'lon':p['Lon'],
                'deltaR':p['DeltaR'],
                'deltaRErr':p['DeltaRErr'],
                'location':p['Locality'],
                'reference':p['Reference'],
                'distance':self.distance(p['Lat'], p['Lon'], float(self.lat), float(self.lon))            
            }

            dl.append(ap)
        dl_sorted = sorted(dl, key=lambda k: k['distance'])
        self.closedict = dl_sorted[0:self.nclosest]
       
        
    def getMean(self):
        
        self._getClosest()
        
        distances = [d['distance'] for d in self.closedict]

        return  statmean(distances)
        
    
    def getMu(self):
        
        self._getClosest()

        num = sum((c['deltaR'])/c['deltaRErr']**2 for c in  self.closedict)
        self.denom = sum([1/nc['deltaRErr']**2 for nc in self.closedict])
        mu = num/self.denom
        self.uncert = (1/self.denom)**(1/2) ### Fixed this

#         print(mu)
        
        return mu

    
    def _getVar(self,mu):

        varnum = sum(((c['deltaR'] - mu)/c['deltaRErr'])**2 for c in self.closedict)
        var = ((1/(self.nclosest-1) * varnum) / (1/self.nclosest * self.denom))
        std = sqrt(var)
        return std

    
    def getUncert(self):
        
        mu = self.getMu()
        std = self._getVar(mu)
        
        return max(std, self.uncert)