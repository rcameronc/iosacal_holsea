import numpy as np
# import os  
import pandas as pd  
from math import cos, asin, sqrt  
# import csv  
# from pathlib import Path  
# from timeit import default_timer as timer  
# from datetime import datetime
from statistics import mean as statmean

from shapely.geometry.polygon import Polygon
import cartopy.crs as ccrs
import cartopy.feature as cfeature 
from matplotlib import pyplot as plt

# from iosacal.core import CalAge  # , CalibrationCurve # alsuren_hpd
from iosacal.core import RadiocarbonDetermination as R
# from iosacal.hpd import hpd_interval
# import iosacal as ios
# from iosacal.text import single_text



class calcDelR:
    
    """ 
    Calculate the delta R + uncertainty for n closest points to a site.
    
    Keyword arguments:
    data -- pandas dataframe with Lat, Lon, DeltaR and DeltaRErr columns
    lon -- longitude of site
    lat -- latitude of site
    nclosest -- the number of closest sites with which to calculate deltaR.
    farthest -- the cutoff distance for data.  
    
    Returns:
    .getMean() -- average distance from point to del R locations.
    .getMu()  -- mean of delta Rs 
    .getUncert() -- weighted delta R uncertainty 
    
    Calculations follow:
    Data Reduct. & Err. Anal. for the Phys. Sci., PR Bevington, 1969, McGraw Hill
    (as described here: http://calib.org/marine/AverageDeltaR.html)
    
    Note: if you want to select the n closest samples, you should specify
    leave maxdist blank and it will default to all samples. 
    If you want to select based on distance, leave nclosest blank and it 
    will default to all samples. 
    
    """

    
    def __init__(self, data, row, nclosest=10000, maxdist=10000):

                
        if not set(['Latitude', 'Longitude']).intersection(row.index):
            err = 'Name your lat/lon columns Latitude/Longitude'
            raise ValueError(err) 
        
        self.data = data[data.site == row.site] ## crucial selection of only data at site
        self.lat = row.Latitude
        self.lon = row.Longitude
        self.ismarine = row.ismarine
        self.nclosest = nclosest
        self.maxdist = maxdist
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
                'distance':self.distance(p['Lat'],
                                         p['Lon'], 
                                         float(self.lat), 
                                         float(self.lon)
                                        )            
            }

            dl.append(ap)
        dl_sorted = sorted(dl, key=lambda k: k['distance'])
        
        # Select only reservoir data closer than maximum distance from sample
        # And only fewer than max number of data
        self.closedict = [d for d in dl_sorted[0:self.nclosest] if d['distance'] < self.maxdist]
       
        
    def getMean(self):
        
        # reservoir correction should be zero if sample is not marine
        if not self.ismarine:
            return 0
        
        self._getClosest()
        
        distances = [d['distance'] for d in self.closedict]

        return  statmean(distances)
        
    
    def getMu(self, text=False):
        
        # reservoir correction should be zero if sample is not marine
        if not self.ismarine:
            return 0
        
        self._getClosest()
        if text:
            txt = f"{len(self.closedict)} data within {self.maxdist} km of [{self.lat}, {self.lon}]"
            print("\n", txt, "\n")
        
        if not self.closedict:
            return 0

        num = sum((c['deltaR'])/c['deltaRErr']**2 for c in self.closedict)
        self.denom = sum([1/nc['deltaRErr']**2 for nc in self.closedict])
        mu = num/self.denom
        self.uncert = (1/self.denom)**(1/2) ### Fixed this

#         print(mu)
        
        return mu

    
    def _getVar(self,mu):
        
        # reservoir correction should be zero if sample is not marine
        if not self.ismarine:
            return 0

        if not self.closedict:
            return 0
        
        varnum = sum(((c['deltaR'] - mu)/c['deltaRErr'])**2 for c in self.closedict)
        var = ((1/(self.nclosest-1) * varnum) / (1/self.nclosest * self.denom))
        std = sqrt(var)
        return std

    
    def getUncert(self):
        
        # reservoir correction should be zero if sample is not marine
        if not self.ismarine:
            return 0
        
        mu = self.getMu()
        std = self._getVar(mu)
        
        if not self.closedict:
            return 0
        
        return max(std, self.uncert)

    

class make_regions:    
    
    def __init__(self, df_R):
        
        self.df_R = df_R
        self.projection = ccrs.PlateCarree()
        
    def _get_coords(self):
        coords = {

            'indian':{
            'indian_ocean_1':[30, 116,  -60, 30, 'coral'],
            'indian_ocean_2':[116, 150,  -60, -30, 'coral'],
            'indian_ocean_3':[116, 150,  -30, -10,  'coral'],
            },

            'mediterranean':{
            'mediterranean_1':[2, 45, 30, 50,  'green'],
            'mediterranean_2':[-5, 2, 30, 42, 'green'],
            },

            'caspian':{
            'caspian_1':[45,65, 30, 50,  'dodgerblue'],
            },

            'pacific':{
            'pacific_northwest':[116, 180, -10, 67, 'blue'],
            'pacific_southwest':[150, 180, -60, -10,'blue'],
            'pacific_east1':[-180, -120, -60, 67, 'blue'],
            'pacific_east2':[-120, -70, -60, 0, 'blue'],
            'pacific_east3':[-120, -70, 0, 8.5, 'blue'],
            'pacific_east4':[-120, -84, 8.5, 15, 'blue'],
            'pacific_east5':[-120, -90, 15, 18, 'blue'],
            'pacific_east6':[-120, -100, 18, 36, 'blue'],
            },

            'baltic':{
            'baltic':[13, 33, 53, 66, 'gray'],
            },

            'northsea':{
            'northsea':[-3, 13, 50, 60, 'brown'],
            },

            'antarctic':{
            'antarctic':[-180, 180, -90, -60, 'skyblue'],
            },

            'atlantic':{
            'south_atlantic': [-70, 30, -60, 0, 'yellow'],
            'central_atlantic': [-60, 15, 0, 30,'yellow'],
            'northcentral_atlantic': [-85, -5, 30, 42,'yellow'],
            'nncentral_atlantic': [-75, 2, 42, 50, 'yellow'],
            'nnncentral_atlantic': [-68, -3, 50, 65,'yellow'],
            'north_atlantic': [-70, -20, 65, 70, 'yellow'],
            'nnorth_atlantic': [-85, -35, 70, 81, 'yellow'],
            },

            'arctic':{
            'arctic_greenland':[-85, -35, 81, 90, 'orange'],
            'arctic_greenlandsea':[-35,50, 70, 90, 'orange'],
            'arctic_norwegiansea':[-20, 20, 65, 70, 'orange'],
            'arctic_centralnorwegiansea':[-3, 15, 60, 65, 'orange'],   
            'arctic_whitesea':[30, 50, 63, 70,  'orange'],   
            'arctic_siberia':[50, 180,  67, 90, 'orange'],
            'arctic_canada':[-180, -85, 67, 90,'orange'],
            },


            'caribbean':{
            'eastcaribbean':[-84,-60, 8.5, 30,  'purple'],
            'centralcaribbean':[-90,-84, 15, 33, 'purple'],
            'westcaribbean':[-100,-90, 18, 33, 'purple'],
            },

            'hudson':{
            'southhudson_bay':[-100, -68, 50, 68, 'olive'],
            'northhudson_bay':[-85, -68, 68, 70, 'olive'],
            }

        }
        return coords
    
    def _get_sitedf(self, name, site):
    
        """
        Transform coords, a dict of sites, each of which is itself
        a dict of lat/lon boxes, into a dataframe.

        inputs:
        name: name of area in coord dictionary.
        site: subdictionary of lat/lon boxes that comprise site 
        """

        df = pd.DataFrame.from_dict(site, orient='index', columns=['lonmin', 
                                                               'lonmax', 
                                                               'latmin', 
                                                               'latmax', 
                                                               'color'])
        df['site'] = name
        
        return df
    
    def _get_boxes(self):
        
        """Make single dataframe with all boxes for all regions. """
        
        coords = self._get_coords()    
        df = pd.concat([self._get_sitedf(n, s) for n, s in coords.items()])

        return df
    
    
    
    def _label_pt(self, r, df_boxes):
        """
        Check if lat/lon point is within any of the ocean basins, then
        Assign to that point the name of the ocean basin. 
        """

        if 'Lon' in r.index:
            lon = r.Lon
            lat = r.Lat
        elif 'lon' in r.index:
            lon = r.lon
            lat = r.lat
        elif 'Longitude' in r.index: 
            lon = r.Longitude
            lat = r.Latitude
       
        else:
            err = "Rename lat/lon to Latitude, Lat, or lat"
            raise ValueError(err)

        # Catch datetime errors in lon
        if lon > 180:
            lon = lon - 360
        if lon < -180:
            lon = lon + 360

            
        for name, d in df_boxes.groupby('site'):

            inany = (lon >= d.lonmin) & (lon < d.lonmax) & (lat >= d.latmin) & (lat < d.latmax)
            if inany.any():
                return name
            else:
                pass
        return 'not in box'
    
    
    def apply_boxes_to_df(self, df):
        
        df = df.copy()

        df_boxes = self._get_boxes()

        df['site'] = df.apply(lambda r: self._label_pt(r, df_boxes), axis=1)
        return df

        

########## THESE METHODS ARE ALL FOR PLOTTING #############

    def _add_shape(self, ax, site):

        """Plot polygon drawn by lat/lon edges from site.

        inputs:
        ax = matplotlib axis.
        site = a single lat/lon box from sites,
                which is a subdict within coords.

        """

        xll, xur, yll, yur, color = site

        pgon = Polygon(
            ((xll, yll),
             (xll, yur),
             (xur, yur),
             (xur, yll),
             (xll, yll)))

        # this plots the polygon
        # must declare correct coordinate system of the data
        ax.add_geometries([pgon], crs=self.projection,
                          facecolor=color, edgecolor='none', 
                          alpha=0.5, zorder=6)



    def _add_features(self, ax):
        """ """
        ocean = cfeature.NaturalEarthFeature('physical', 'ocean', '50m')
        land = cfeature.NaturalEarthFeature('physical', 'land', '50m')
        ax.add_feature(ocean, color='lightgray', zorder=0)
        ax.add_feature(land, color='w', zorder=1)
        ax.gridlines(crs=ccrs.PlateCarree(), linewidth=1, 
                     color='w', alpha=0.5, draw_labels=False, zorder=4)
        ax.coastlines(resolution='50m', zorder=3)

        return ax


    def plot_polygons(self, **save):

        transform = ccrs.Robinson() 

        fig = plt.figure(figsize=(20, 10))

        ax = fig.add_subplot(111, projection=transform)

        self._add_features(ax)

        ax.scatter(self.df_R.Lon, self.df_R.Lat, 
                   color='orange', ec='k', 
                   s=100, transform=self.projection,
                  zorder=5)

        cs = self._get_coords()
        [self._add_shape(ax, s) for _, c in cs.items() for _, s in c.items()]
        
        if save:
            fig.savefig('./pictures/regionboxes.png')

       #  return fig







class iosacal_util:
    
    def findsorted(n, array):
        '''Return sorted array and index of n inside array.'''
        a = asarray(array)
        a.sort()
        i = a.searchsorted(n)
        return a, i


    def prev(n, array):
        '''Find interval between n and its previous, inside array.'''

        a,i = findsorted(n, array)
        if i-1 < 0:
            prev = None
        else:
            prev = a[i-1]
        return prev

    def next(n, array):
        '''Find interval between n and its next, inside array.'''

        a,i = findsorted(n, array)
        try:
            next = a[i+1]
        except IndexError:
            next = None
        return next


    def calc_median(calibrated_age):
        '''
        Return median calibrated age from calAge object.  

        '''
        cal_curve = calibrated_age.copy()
        # sort rows by second column in inverse order
        cal_cumsum = cal_curve[:,1].cumsum()
        median = cal_curve[np.argmin(abs(cal_cumsum - 0.5)),0]
        
        return median

        
        
        
        
#         hpd_sorted = hpd_curve[hpd_curve[:,1].argsort(),][::-1]
#         hpd_cumsum = hpd_sorted[:,1].cumsum()
#         # normalised values
#         hpd_cumsum /= hpd_cumsum[-1]

#         threshold_index = hpd_cumsum.searchsorted(1 - alpha)
#         threshold_p = hpd_sorted[threshold_index][1]
#         threshold_index = calibrated_age[:,1] > threshold_p
#         hpd = list(hpd_curve[threshold_index,0])

#         return np.mean(hpd) 
    
    
