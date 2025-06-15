#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
print(sys.executable)
import netCDF4 as nc
import random, glob, math, cartopy
from osgeo import gdal
import datetime, calendar, string, re
import numpy as np
import pandas as pd
from numpy import zeros, newaxis 
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import colors
import cartopy.crs as ccrs
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
from scipy import stats
from scipy.stats import linregress
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import statsmodels.formula.api as smf
from matplotlib.patches import Rectangle
import skimage.transform as st
from scipy.ndimage import zoom


# In[2]:


#from daily mean monthly to get daily mean annual  
def ann_mean(data,nyears,nlat,nlon):
    years = np.linspace(2001, 2016, nyears)
    ann_mean = np.zeros((nyears,nlat,nlon))
    ann_mean = np.ma.masked_values (ann_mean,0)
    for ny, yy in enumerate(years):       
        ann_mean[ny,:,:] = np.ma.mean(data[ny*12:ny*12+12, :, :], axis=0)
    return ann_mean #daily gpp


# In[3]:


#from daily mean monthly to get annual  
def ann_sum(data,nyears,days,nlat,nlon):
    years = np.linspace(2001, 2016, nyears)
    ann_sum = np.zeros((nyears,nlat,nlon))
    ann_sum = np.ma.masked_values (ann_sum,0)
    for ny, yy in enumerate(years):
        dt =data[ny*12:ny*12+12, :, :]
        for x in range(12):
            dt[x,:,:] = dt[x,:,:]*days[x]
        ann_sum[ny,:,:] = np.ma.sum(dt, axis=0)
    return ann_sum #annual gpp


# In[4]:


# function to create masked array
def masking(ar1, ar2):
    res_mask = np.ma.getmask(ar2)
    masked = np.ma.masked_array(ar1, mask=res_mask)
    return masked


# In[5]:


#functions to get regions 
def North_America (lat,lon,dt):
    m1 = np.ma.masked_where(lon<-125, dt )
    m2 = np.ma.masked_where(lon>-93, m1 )
    m3= np.ma.masked_where(lat<-53, m2 )
    m4= np.ma.masked_where(lat>-15, m3 )
    return m4


# In[6]:


def get_extent(fn):
    "'returns min_x, max_y, max_x, min_y'"
    ds = gdal.Open(fn)
    gt = ds.GetGeoTransform()
    return (gt[0], gt[3], gt[0]+gt[1]*ds.RasterXSize, gt[3]+gt[5]*ds.RasterYSize)


# In[7]:


def label_generator(case='lowercase letter', start='', end=''):
	"""
	Generates a label with a), b) etc
	"""
	choose_type = {'lowercase letter': string.ascii_lowercase, \
		       'uppercase letter': string.ascii_uppercase}
		       #'arabic number': string.digits}
		      # 'lowercase roman': [roman.toRoman(int(i)+1).lower() for i in string.digits]+[roman.toRoman(int(i)+11).lower() for i in string.digits], \
		      # 'uppercase roman': [roman.toRoman(int(i)+1) for i in string.digits]+[roman.toRoman(int(i)+11) for i in string.digits] }
	generator = ('%s%s%s' %(start, letter, end) for letter in choose_type[case])

	return generator


# In[8]:


def coord_to_gridcell(lat, lon, ul_lat=90., ul_lon=-180., grid_resolution=0.5):
		"""
		Takes a lat and lon coordinate and finds the right grid cell depending on the resolution
		Output row, column and grid cell number
		NOTE: assumes lat and lon in top lefthand corner (ACTUAL TOP LEFTHAND CORNER, **NOT** MIDDLE OF THE CELL, SO CHECK YOUR FORCING), and zero base.
		"""

		if type(grid_resolution) == float: grid_resolution = np.array([grid_resolution, grid_resolution])

		row = np.floor( (ul_lat - lat) / grid_resolution[0] )
		col = np.floor( (lon - ul_lon ) / grid_resolution[1] )
		nlon = 360 / grid_resolution[1]
		grid_cell = nlon * row + col
		print(lat,lon)

		return int(row), int(col), int(grid_cell)


# In[9]:


#getting dryland data
def get_dryland(AI,data):
    m1 = np.ma.masked_where(AI>0.5, data)
    m2 = np.ma.masked_where(AI<0.05, m1 )
    #masking out cold dryland
    #m3 = np.ma.masked_where(lat<-55, m2 )
    #dryland = np.ma.masked_where(lat>55, m3)
    return m2


# In[10]:


#getting dryland Aridity index 
dryland_files = glob.glob('/Users/rubayapervin/OneDrive - Indiana University/Research Projects/Evaluation_TRENDY_GPP/Data/Dryland/Global-AI_ET0_v3_annual/*.tif')
file = gdal.Open(dryland_files[0]) #AI file
data = file.ReadAsArray()
#getting the extent of 1st file in the file list 
min_x, max_y, max_x, min_y = get_extent(dryland_files[0])
gt = file.GetGeoTransform()
rows = math.ceil((max_y-min_y)/-gt[5])
columns = math.ceil((max_x-min_x)/gt[1])
#down scaleing AI
AI =np.zeros((360, 720))
for x in range(360):
    for y in range(720):
        Y=data[x*60:x*60+60,y*60:y*60+60]
        ai = np.ma.mean (Y)
        AI[x,y]=ai
#applying scale factor
scl_AI = AI/10000


# In[11]:


#read DryFlux
# Function to extract year and month from filename
def get_date_from_filename(filename):
    # Extract the date part (e.g., "2001May2001" or "2000Nov2000")
    match = re.search(r'DF_X(\d{4}[A-Za-z]+\d{4})\.tif', filename)
    if not match:
        return (0, 0)  # Default for files that don't match
    
    date_str = match.group(1)
    # Extract the first year (2001 in "2001May2001")
    year = int(date_str[:4])
    # Extract the month name (May in "2001May2001")
    month_str = re.search(r'\d{4}([A-Za-z]+)\d{4}', date_str).group(1)
    
    # Convert month name to number (1-12)
    month_dict = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    month = month_dict.get(month_str, 0)
    
    return (year, month)

# Get all files and sort them by year and month
DryFlux_files = glob.glob('/Users/rubayapervin/OneDrive - Indiana University/Research Projects/Evaluation_TRENDY_GPP/Data/DryFlux/*.tif')

# Sort files by year and month
DryFlux_files_sorted = sorted(DryFlux_files, key=lambda x: get_date_from_filename(x))

# Filter files to only include years 2001-2016
DryFlux_files_filtered = [f for f in DryFlux_files_sorted if 2001 <= get_date_from_filename(f)[0] <= 2016]

# Now process the files in order
# Read first file 
if DryFlux_files_filtered:
    first_file = DryFlux_files_filtered[0]
    first_year = get_date_from_filename(first_file)[0]
    
    # If the first year is 2000 skip to 2001
    if first_year == 2000:
        start_index = 12 - get_date_from_filename(first_file)[1]  # Skip remaining months of 2000
        DryFlux_files_filtered = DryFlux_files_filtered[start_index:]
    
    # Process the files
    if DryFlux_files_filtered:
        ds = gdal.Open(DryFlux_files_filtered[0])
        Dryflux_GPP = ds.ReadAsArray()
        Dryflux_GPP = Dryflux_GPP[np.newaxis, :, :]

        # Combine all dryflux data
        for x in DryFlux_files_filtered[1:]: 
            ds = gdal.Open(x)
            dt = ds.ReadAsArray()
            dt = dt[np.newaxis, :, :]
            Dryflux_GPP = np.concatenate((Dryflux_GPP, dt), axis=0)
        
        Dryflux_GPP = np.ma.masked_values(Dryflux_GPP, np.min(Dryflux_GPP))

        # Getting dryflux annual sum
        nyears = int(Dryflux_GPP.shape[0]/12)
        years = np.linspace(2001, 2016, nyears)
        nlat = Dryflux_GPP.shape[1]
        nlon = Dryflux_GPP.shape[2] 
        days = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]) 
        
        # annual sum
        Dryflux = ann_sum(Dryflux_GPP, nyears, days, nlat, nlon)
        
        # Convert to kg C 
        Dryflux = Dryflux/1000


# In[12]:


#getting trendy gpp
#  model order
gpp_source_order = ['CABLE-POP', 'CLASSIC', 'CLM5', 'IBIS', 'ISBA-CTRIP', 
                   'JSBACH', 'JULES', 'LPJ-GUESS', 'LPJwsl', 'LPX-Bern', 
                   'OCN', 'ORCHIDEE', 'SDGVM', 'VISIT', 'YIBs']

# Get all files
TRENDY_gpp_files = glob.glob('/Users/rubayapervin/OneDrive - Indiana University/Research Projects/Evaluation_TRENDY_GPP/Data/Trendy/V11/yearlyMean/*1979-2021_yearlymean_XYT.nc')

# Create a function to extract model name from filename
def get_model_name(filename):
    # Extract the part between 'fco2_' and '-S3'
    start = filename.find('fco2_') + 5
    end = filename.find('-S3')
    return filename[start:end]

# Sort files according to gpp_source_order
TRENDY_gpp_files_sorted = sorted(TRENDY_gpp_files, 
                                key=lambda x: gpp_source_order.index(get_model_name(x)))

# Verify the order
for file in TRENDY_gpp_files_sorted:
    print(file)

# Process the files in order
# Read first file
data = nc.Dataset(TRENDY_gpp_files_sorted[0], 'r')
gpp_data = data.variables['gpp'][:]
gpp_data = np.flip(gpp_data, 1)
dt = gpp_data[((2001-1979)*1):((2017-1979)*1), :, :]
# hourly mean annual gpp to annual gpp
TRENDY_gpp = dt * 24 * 365
TRENDY = TRENDY_gpp[np.newaxis, :, :, :]

# Combine all TRENDY_gpp data
for x in TRENDY_gpp_files_sorted[1:]: 
    data = nc.Dataset(x, 'r')
    gpp_data = data.variables['gpp'][:]
    gpp_data = np.flip(gpp_data, 1)
    dt = gpp_data[((2001-1979)*1):((2017-1979)*1), :, :]
    dt = dt * 24 * 365  # hourly mean annual gpp to annual gpp
    dt = dt[np.newaxis, :, :, :]
    TRENDY = np.concatenate((TRENDY, dt), axis=0)

print(TRENDY.shape)
TRENDY = np.ma.masked_values(TRENDY, 2.19e+37)
TRENDY = np.ma.masked_values(TRENDY, 1e+34)
TRENDY = np.ma.masked_values(TRENDY, 1.4600001e+37)
print(np.ma.min(TRENDY), np.ma.max(TRENDY))
print(TRENDY.shape)


# In[13]:


#crop mask and gpp data list 
mask_list = [ 'CLASSIC', 'CLM5','IBIS', 'ISBA CTRIP', 'JSBACH', 'JULES', 'LPJ GUESS', 'LPX Bern', 'OCN', 'ORCHIDEE', 'SDGVM', 'VISIT','YIBs']
gpp_source = ['CABLE POP', 'CLASSIC', 'CLM5', 'IBIS', 'ISBA CTRIP', 'JSBACH', 'JULES', 'LPJ GUESS', 'LPJwsl', 'LPX Bern', 'OCN', 'ORCHIDEE', 'SDGVM', 'VISIT', 'YIBs']


# In[14]:


#getting lat lon
TRENDY_gpp_file = TRENDY_gpp_files[0]
file= nc.Dataset(TRENDY_gpp_file,'r')
var_names=file.variables.keys() 
print(var_names)
dim_names=file.dimensions.keys()
print(dim_names)
lat = file.variables['latitude'][:]
lon = file.variables['longitude'][:]
lon, lat = np.meshgrid(lon, lat)


# In[15]:


crop_mask_files =glob.glob('/Users/rubayapervin/OneDrive - Indiana University/Research Projects/Evaluation_TRENDY_GPP/Data/Trendy/V11/PFT/TRENDY-v11_cropMasked/mod_masks/*')
for x  in crop_mask_files: 
    print(x)


# In[16]:


# Initialize list to store masks
all_masks = []

# Loop through each model in gpp_source
for mod in gpp_source:
    if mod in mask_list:  # Check if the model has a mask
        mask_index = mask_list.index(mod)
        mask = np.load(crop_mask_files[mask_index])['mask']
        
        # Resize mask to (360, 720) if needed (nearest-neighbor interpolation)
        if mask.shape != (360, 720):
            zoom_factors = (360 / mask.shape[0], 720 / mask.shape[1])
            mask = zoom(mask, zoom_factors, order=0)  # order=0 for nearest-neighbor
        
        all_masks.append(mask)

# Create common_mask (True where ANY model's mask is True)
if all_masks:
    common_mask = np.any(np.stack(all_masks, axis=0), axis=0)
else:
    common_mask = np.zeros((360, 720), dtype=bool)  # No masking if no masks exist


# In[17]:


# Flip the mask for TRENDY data orientation
common_mask_flipped = np.flip(common_mask, axis=0)

# Apply the common mask to all models and years
for idx in range(len(gpp_source)):  # Loop over all models
    for nyear in range(TRENDY.shape[1]):  # Loop over all years
        TRENDY[idx, nyear, :, :] = np.ma.masked_where(
            common_mask_flipped, 
            TRENDY[idx, nyear, :, :]  
        )


# In[18]:


#get north american drynad 
#get AI data in north America
AI_N = North_America (lat,lon,scl_AI)
AI = get_dryland(scl_AI, AI_N)


# In[19]:


#Getting dryland (arid and semi-arid regions)
dryland = get_dryland(scl_AI,scl_AI)
#dryland = masking(dryland, region_mask)


# In[20]:


# applying mask to keep common pixels of dryflux, trendy, and AI data
dryflux = masking(Dryflux, np.ma.mean(TRENDY,axis=0))
for nmod in np.arange(np.shape(TRENDY)[0]):
    TRENDY[nmod,:,:,:] = masking(TRENDY[nmod,:,:,:], dryflux)
dryflux_mean_gpp = np.ma.mean(dryflux, axis=0)
AI = masking(AI, dryflux_mean_gpp)


# In[21]:


# getting north american dryland dryflux gpp
for nyear in np.arange(np.shape(dryflux)[0]):
    dryflux[nyear] = get_dryland(AI,dryflux[nyear])
# getting north american dryland trendy gpp
for nmod in np.arange(np.shape(TRENDY)[0]):
    for nyear in np.arange(np.shape(TRENDY)[1]):
        TRENDY[nmod,nyear,:,:] = get_dryland(AI,TRENDY[nmod,nyear,:,:])
print(TRENDY.shape)


# In[ ]:





# In[22]:


# getting trendy mean gpp
trendy_mean_gpp = np.ma.mean(TRENDY, axis=1)
dryflux_mean_gpp = np.ma.mean(dryflux, axis=0)


# In[23]:


#getting standard deviation
dry_sd =np.ma.std(dryflux, axis=0)
trendy_sd = np.ma.std(TRENDY, axis=1)


# In[24]:


test_mask = np.ma.array(trendy_mean_gpp).mask
#getting  linregress
slope =np.empty_like(trendy_mean_gpp)
intercept =np.empty_like(trendy_mean_gpp)
r =np.empty_like(trendy_mean_gpp)
p =np.empty_like(trendy_mean_gpp)
se =np.empty_like(trendy_mean_gpp)
for nmod in np.arange(np.shape(TRENDY)[0]):
    for nlat in np.arange(np.shape(dryflux)[1]):
        for nlon in np.arange(np.shape(dryflux)[2]):
            if test_mask[nmod,nlat,nlon] == False:
                slope[nmod,nlat,nlon], intercept[nmod,nlat,nlon], r[nmod,nlat,nlon], p[nmod,nlat,nlon], se[nmod,nlat,nlon] = linregress( masking(dryflux, TRENDY[nmod])[:, nlat, nlon] ,  TRENDY[nmod, :, nlat, nlon])


# In[25]:


r2 = r*r


# In[26]:


# Define the order to plot the GPP sources
gpp_source_order = ['CABLE POP', 'CLASSIC', 'CLM5', 'IBIS', 'ISBA CTRIP', 'JSBACH', 'JULES', 'ORCHIDEE', 'VISIT', 'YIBs', 'LPJ GUESS', 'LPJwsl', 'LPX Bern', 'OCN', 'SDGVM']
valid_source = ['CLASSIC', 'CLM5','JULES', 'ORCHIDEE',  'JSBACH',  'ISBA CTRIP','CABLE POP','YIBs', 'IBIS', 'VISIT','SDGVM', 'LPJ GUESS', 'LPX Bern', 'OCN','LPJwsl']


# Create a mapping from GPP source to its index
gpp_source_mapping = {source: i for i, source in enumerate(gpp_source)}


# In[27]:


def resize(dt):
    dt = st.resize(dt, (360, 720), order=0, preserve_range=True, anti_aliasing=False)
    return dt


# In[28]:


model_list = ['CABLE POP', 'CLASSIC', 'CLM5','IBIS','ISBA CTRIP', 'JSBACH', 'JULES', 'LPJ GUESS','LPJwsl', 'LPX Bern', 'OCN', 'ORCHIDEE', 'SDGVM', 'VISIT', 'YIBs']
print(mask_list)


# In[29]:


from pathlib import Path

# Your directory containing the files
pft_dir = '/Users/rubayapervin/OneDrive - Indiana University/Research Projects/Evaluation_TRENDY_GPP/Data/Trendy/V11/PFT/TRENDY-v11_cropMasked/'

# Create a mapping of model names to their files
model_files = {}
for f in Path(pft_dir).glob('*_masked.npz'):
    model_name = f.stem.replace('_masked', '')
    if model_name == 'ISBA-CTRIP':
        model_name = 'ISBA CTRIP'
    model_files[model_name] = str(f)

# Also check for .npz files without '_masked'
for f in Path(pft_dir).glob('*.npz'):
    if '_masked' not in f.stem:
        model_name = f.stem
        if model_name == 'CABLE POP':
            model_files[model_name] = str(f)
        elif model_name == 'LPJwsl':
            model_files[model_name] = str(f)

# Now create the sorted list
sorted_pft_files = []
for model in model_list:
    if model in model_files:
        sorted_pft_files.append(model_files[model])
    else:
        print(f"Warning: No file found for model {model}")
        sorted_pft_files.append(None)

for x  in sorted_pft_files: 
    print(x)


# In[30]:


area = np.loadtxt('/Users/rubayapervin/OneDrive - Indiana University/Research Projects/Evaluation_TRENDY_GPP/Data/Area/orch05_grid_area.dat')


# In[31]:


dry_area = get_dryland(AI,area)


# In[36]:


def get_sd_pft(pft_type, pft_files, AI):
    sd_pft = []  # Initialize an empty list to store the standard deviation results.
    
    # Loop over each file in pft_files
    for idx, mod in enumerate(pft_files):
        data = np.load(mod)  # Load the data from the file using NumPy's load function (assumes .npy files).
        
        # Try to extract the data for the specified plant functional type (pft_type) from the loaded data.
        try:
            pft_dt = data[pft_type] if pft_type in data else None  # Check if pft_type exists in data, otherwise set to None.
            #print(pft_dt.shape, np.min(pft_dt),np.max(pft_dt))
            #plt.imshow(pft_dt[0])
            #plt.show()
            # If the pft_type data exists, calculate the standard deviation along the 0th axis (across a dimension of the data).
            pft_sd = np.ma.std(pft_dt, axis=0) if pft_dt is not None else None
            
            # Resize the standard deviation array (you may need to implement or provide a definition for the resize function).
            sd = resize(pft_sd) if pft_sd is not None else None  # Resize the standard deviation data.
            # getting North American Dryland 
            sd = get_dryland(AI,np.flip(sd,0))
            #plt.imshow(sd)
            #plt.show()
            # Append the result to the list (if valid data was found).
            sd_pft.append(sd)
        except Exception as e:
            print(f"Error processing {mod}: {e}")
    
    # Return the list of standard deviation arrays (some may be None if there was no valid data for a file).
    #print(sd_pft)
    return sd_pft


# In[37]:


def get_cv_pft(pft_type, pft_files, AI):
    cv_pft = []  # Initialize an empty list to store the standard deviation results.
    
    # Loop over each file in pft_files
    for idx, mod in enumerate(pft_files):
        data = np.load(mod)  
        
        # Try to extract the data for the specified plant functional type (pft_type) from the loaded data.
        try:
            pft_dt = data[pft_type] if pft_type in data else None  # Check if pft_type exists in data, otherwise set to None.
            #print(pft_dt.shape, np.min(pft_dt),np.max(pft_dt))
            #plt.imshow(pft_dt[0])
            #plt.show()
            # If the pft_type data exists, calculate the standard deviation along the 0th axis (across a dimension of the data).
            pft_cv = np.ma.std(pft_dt, axis=0)/np.ma.mean(pft_dt, axis=0) if pft_dt is not None else None
            
            # Resize the standard deviation array (you may need to implement or provide a definition for the resize function).
            cv = resize(pft_cv) if pft_cv is not None else None  # Resize the standard deviation data.
            # getting North American Dryland 
            cv = get_dryland(AI,np.flip(cv,0))
            #plt.imshow(cv)
            #plt.show()
            # Append the result to the list (if valid data was found).
            cv_pft.append(cv)
        except Exception as e:
            print(f"Error processing {mod}: {e}")
    
    # Return the list of standard deviation arrays (some may be None if there was no valid data for a file).
    #print(cv_pft)
    return cv_pft


# In[38]:


def get_mean_pft(pft_type, pft_files, AI):
    mean_pft = []  # Initialize the list to hold the mean values (or None)
    
    # Loop over each file in pft_files
    for idx, mod in enumerate(pft_files):
        data = np.load(mod)  # Load the data from the file using NumPy's load function (assumes .npy files).
        
        # Try to extract the data for the specified plant functional type (pft_type) from the loaded data.
        try:
            pft_dt = data[pft_type] if pft_type in data else None  # Check if pft_type exists in data, otherwise set to None.
            
            if pft_dt is not None:
                #print(pft_dt.shape, np.min(pft_dt), np.max(pft_dt))
                #plt.imshow(pft_dt[0])
                #plt.show()
                
                # Calculate the mean along the 0th axis (across a dimension of the data)
                pft_mean = np.ma.mean(pft_dt, axis=0)
                
                # Resize the mean array (implement or provide a resize function)
                pft_mean = resize(pft_mean) if pft_mean is not None else None
                
                # Apply dryland masking with AI
                pft_mean = get_dryland(AI, np.flip(pft_mean, 0))
                
                #plt.imshow(pft_mean)
                #plt.show()
            else:
                pft_mean = None
            
        except Exception as e:
            print(f"Error processing {mod}: {e}")
            pft_mean = None  # If there's an error, ensure pft_mean is set to None

        # Append the result (or None) to the list
        mean_pft.append(pft_mean)
    
    # Return the list of mean arrays (or None values for files without valid data)
    return mean_pft


# In[39]:


allveg_sd = get_sd_pft('allveg', sorted_pft_files, AI)
woody_sd = get_sd_pft('woody_allveg', sorted_pft_files, AI)
nonwoody_sd = get_sd_pft('nonwoody_allveg', sorted_pft_files, AI)
c4grass_sd = get_sd_pft('c4grass', sorted_pft_files, AI)
c3grass_sd = get_sd_pft('c3grass', sorted_pft_files, AI)
crop_sd = get_sd_pft('allcrop', sorted_pft_files, AI)


# In[40]:


allveg_cv = get_cv_pft('allveg', sorted_pft_files, AI)
woody_cv = get_cv_pft('woody_allveg', sorted_pft_files, AI)
nonwoody_cv = get_cv_pft('nonwoody_allveg', sorted_pft_files, AI)
c4grass_cv = get_cv_pft('c4grass', sorted_pft_files, AI)
c3grass_cv = get_cv_pft('c3grass', sorted_pft_files, AI)
crop_cv = get_cv_pft('allcrop', sorted_pft_files, AI)


# In[41]:


c4grass_mean = get_mean_pft('c4grass', sorted_pft_files, AI)
c3grass_mean = get_mean_pft('c3grass', sorted_pft_files, AI)
allcrop_mean = get_mean_pft('allcrop', sorted_pft_files, AI)
allgrass_mean  = get_mean_pft('allgrass', sorted_pft_files, AI)


# In[42]:


def get_df(pft_type, pft_data, metric_name, metric_data, AI, gpp_source, model_list):
    df = pd.DataFrame(columns=['Models', pft_type, metric_name, 'Aridity'])   
    models = []
    pft_dt = []
    metric_dt = []
    aridity = []

    for idx, model in enumerate(model_list):
        # Check if gpp_source has the model before proceeding
        if model not in gpp_source:
            print(f"Skipping model {model} because it is not in gpp_source")
            continue

        gpp_source_index = gpp_source.index(model)

        # Check if pft_data[idx] or metric_data[gpp_source_index] exists and is not None
        if idx >= len(pft_data) or pft_data[idx] is None:
            print(f"Skipping pft_data[{idx}] because it is None or out of range")
            continue
        if gpp_source_index >= len(metric_data) or metric_data[gpp_source_index] is None:
            print(f"Skipping metric_data[{gpp_source_index}] because it is None or out of range")
            continue

        # Check if the shape is valid
        if pft_data[idx].shape != (360, 720):
            print(f"Skipping pft_data[{idx}] due to incorrect shape: {pft_data[idx].shape}")
            continue
        if metric_data[gpp_source_index].shape != (360, 720):
            print(f"Skipping metric_data[{gpp_source_index}] due to incorrect shape: {metric_data[gpp_source_index].shape}")
            continue

        # Apply masking
        pft_data[idx] = masking(pft_data[idx], metric_data[gpp_source_index])
        metric_data[gpp_source_index] = masking(metric_data[gpp_source_index], pft_data[idx])
        AI = masking(AI, metric_data[gpp_source_index])
        
        # Get the mask for valid data points
        test_mask = np.ma.array(AI).mask
        
        # Iterate over latitude and longitude
        for nlat in range(np.shape(AI)[0]):
            for nlon in range(np.shape(AI)[1]):
                if not test_mask[nlat, nlon] and 0.05 <= AI[nlat, nlon] <= 0.2:
                    models.append(model)
                    aridity.append('Arid')
                    pft_dt.append(pft_data[idx][nlat, nlon])
                    metric_dt.append(metric_data[gpp_source_index][nlat, nlon])
                elif not test_mask[nlat, nlon] and 0.2 < AI[nlat, nlon] <= 0.5:
                    models.append(model)
                    aridity.append('Semi-Arid')
                    pft_dt.append(pft_data[idx][nlat, nlon])
                    metric_dt.append(metric_data[gpp_source_index][nlat, nlon])
    
    # Populate DataFrame
    df['Models'] = models
    df['Aridity'] = aridity
    df[pft_type] = pft_dt
    df[metric_name] = metric_dt
    
    # Convert to float type
    df[pft_type] = df[pft_type].astype('float')
    df[metric_name] = df[metric_name].astype('float')
    
    return df


# In[43]:


def get_df(pft_type, pft_data, metric_name, metric_data, AI, gpp_source, model_list, dry_area=dry_area):
    df = pd.DataFrame(columns=['Models', pft_type, metric_name, 'Aridity', 'Area'])
    models = []
    pft_dt = []
    metric_dt = []
    aridity = []
    area = []
    
    for idx, model in enumerate(model_list):
        # Check if the model exists in gpp_source
        if model not in gpp_source:
            print(f"Model {model} not in gpp_source, adding empty values for it.")
            models.append(model)
            aridity.append('No Data')
            pft_dt.append(0)
            metric_dt.append(0)
            area.append(0)
            continue

        gpp_source_index = gpp_source.index(model)

        # Check pft_data and metric_data validity
        if idx >= len(pft_data) or pft_data[idx] is None:
            print(f"pft_data[{idx}] is None or out of range, adding empty values for {model}.")
            models.append(model)
            aridity.append('No Data')
            pft_dt.append(0)
            metric_dt.append(0)
            area.append(0)
            continue
        
        if gpp_source_index >= len(metric_data) or metric_data[gpp_source_index] is None:
            print(f"metric_data[{gpp_source_index}] is None or out of range, adding empty values for {model}.")
            models.append(model)
            aridity.append('No Data')
            pft_dt.append(0)
            metric_dt.append(0)
            area.append(0)
            continue

        # Check if the shape is valid
        if pft_data[idx].shape != (360, 720):
            print(f"pft_data[{idx}] has incorrect shape: {pft_data[idx].shape}, adding empty values for {model}.")
            models.append(model)
            aridity.append('No Data')
            pft_dt.append(0)
            metric_dt.append(0)
            area.append(0)
            continue
        
        if metric_data[gpp_source_index].shape != (360, 720):
            print(f"metric_data[{gpp_source_index}] has incorrect shape: {metric_data[gpp_source_index].shape}, adding empty values for {model}.")
            models.append(model)
            aridity.append('No Data')
            pft_dt.append(0)
            metric_dt.append(0)
            area.append(0)
            continue

        # Apply masking
        pft_data[idx] = masking(pft_data[idx], metric_data[gpp_source_index])
        metric_data[gpp_source_index] = masking(metric_data[gpp_source_index], pft_data[idx])
        AI = masking(AI, metric_data[gpp_source_index])
        dry_area = masking(dry_area, metric_data[gpp_source_index])
        
        # Get the mask for valid data points
        test_mask = np.ma.array(AI).mask
        
        # Iterate over latitude and longitude
        for nlat in range(np.shape(AI)[0]):
            for nlon in range(np.shape(AI)[1]):
                if not test_mask[nlat, nlon] and 0.05 <= AI[nlat, nlon] <= 0.2:
                    models.append(model)
                    aridity.append('Arid')
                    pft_dt.append(pft_data[idx][nlat, nlon])
                    metric_dt.append(metric_data[gpp_source_index][nlat, nlon])
                    area.append(dry_area[nlat, nlon])
                elif not test_mask[nlat, nlon] and 0.2 < AI[nlat, nlon] <= 0.5:
                    models.append(model)
                    aridity.append('Semi-Arid')
                    pft_dt.append(pft_data[idx][nlat, nlon])
                    metric_dt.append(metric_data[gpp_source_index][nlat, nlon])
                    area.append(dry_area[nlat, nlon])
    
    # Ensure consistency across all lists
    min_len = min(len(models), len(aridity), len(pft_dt), len(metric_dt), len(area))
    models, aridity, pft_dt, metric_dt, area = models[:min_len], aridity[:min_len], pft_dt[:min_len], metric_dt[:min_len], area[:min_len]
    
    # Populate DataFrame
    df['Models'] = models
    df['Aridity'] = aridity
    df[pft_type] = pft_dt
    df[metric_name] = metric_dt
    df['Area'] = area
    
    # Convert to float type
    df[pft_type] = df[pft_type].astype('float')
    df[metric_name] = df[metric_name].astype('float')
    df['Area'] = df['Area'].astype('float')
    
    return df


# In[44]:


df_sd_woody = get_df('Woody PFT SD', woody_sd, 'GPP SD', trendy_sd, AI, gpp_source, model_list, dry_area)
df_sd_nonwoody = get_df('Non-woody PFT SD', nonwoody_sd, 'GPP SD', trendy_sd, AI, gpp_source, model_list, dry_area)
df_sd_allveg = get_df('All Veg PFT SD', allveg_sd, 'GPP SD', trendy_sd, AI, gpp_source, model_list, dry_area)
df_sd_c4grass = get_df('C4 Grass PFT SD', c4grass_sd, 'GPP SD', trendy_sd, AI, gpp_source, model_list, dry_area)
df_sd_c3grass = get_df('C3 Grass PFT SD', c3grass_sd, 'GPP SD', trendy_sd, AI, gpp_source, model_list, dry_area)
df_sd_crop = get_df('Crop PFT SD', crop_sd, 'GPP SD', trendy_sd, AI, gpp_source, model_list, dry_area)


# In[45]:


trendy_cv = np.ma.std(TRENDY, axis=1)/np.ma.mean(TRENDY, axis=1)
df_cv_woody = get_df('Woody PFT CV', woody_cv, 'GPP CV', trendy_cv, AI, gpp_source, model_list, dry_area)
df_cv_nonwoody = get_df('Non-woody PFT CV', nonwoody_cv, 'GPP CV', trendy_cv, AI, gpp_source, model_list, dry_area)
df_cv_allveg = get_df('CV of annual total PFT', allveg_cv, 'CV of annual GPP', trendy_cv, AI, gpp_source, model_list, dry_area)
df_cv_c4grass = get_df('C4 Grass PFT CV', c4grass_cv, 'GPP CV', trendy_cv, AI, gpp_source, model_list, dry_area)
df_cv_c3grass = get_df('C3 Grass PFT CV', c3grass_cv, 'GPP CV', trendy_cv, AI, gpp_source, model_list, dry_area)
df_cv_crop = get_df('Crop PFT CV', crop_cv, 'GPP CV', trendy_cv, AI, gpp_source, model_list, dry_area)


# In[46]:


def plot_scatter_fit(df, x_col, y_col, aridity_col, model_col, model_order, 
                    xlabel=None, ylabel=None, xlim=None, ylim=None, add_1to1_line=False):
    """
    Create scatter plots with fit lines for models in model_order. If a model is not available, leave the plot space blank.
    
    Parameters:
    - df: DataFrame containing data to plot
    - x_col: column name for x-axis data
    - y_col: column name for y-axis data
    - aridity_col: column name for hue (e.g., 'Aridity')
    - model_col: column name that contains model names
    - model_order: list of model names specifying the order of plotting
    - xlabel: label for x-axis in the bottom row of subplots
    - ylabel: label for y-axis in the left column of subplots
    - xlim: tuple (xmin, xmax) for x-axis limits
    - ylim: tuple (ymin, ymax) for y-axis limits
    - add_1to1_line: boolean, whether to add a 1:1 line to each subplot
    """
    # Create a figure and a grid of subplots
    ncols = 4
    nrows = 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 4))
    
    # Flatten the axes array for easy indexing
    axes = axes.flatten()

    # Define the colors for the 'Arid' and 'Semi-Arid' categories
    aridity_colors = {'Arid': 'orange', 'Semi-Arid': 'blue'}

    # Iterate through model_order and plot accordingly
    for idx, model in enumerate(model_order):
        # Filter the DataFrame for the current model and exclude 0 or negative values in x_col
        df_model = df[(df[model_col] == model) & (df[x_col] >= 0)]  # Filter out 0 or negative values in x_col
        
        if not df_model.empty:
            ax = axes[idx]
            # Plot scatter plot with color corresponding to aridity categories
            sns.scatterplot(data=df_model, x=x_col, y=y_col, ax=ax, color='blue', alpha=0.5)
            
            # Add 1:1 line if requested
            if add_1to1_line:
                min_val = min(df_model[x_col].min(), df_model[y_col].min())
                max_val = max(df_model[x_col].max(), df_model[y_col].max())
                ax.plot([min_val, max_val], [min_val, max_val], 
                        color='black', linestyle=':', linewidth=1, label='1:1 line')
            
            # Calculate and plot regression line
            slope_line, intercept_line, r_value, p_value, std_err = stats.linregress(df_model[x_col], df_model[y_col])
            print(model, ':', p_value)
            fit_line = slope_line * df_model[x_col] + intercept_line
            ax.plot(df_model[x_col], fit_line, color='red', linestyle='--', label='Regression line')
            
            r_squared = r_value ** 2
            
            ax.text(0.05, 0.95, f'$R^2$: {r_squared:.2f}', 
                    transform=ax.transAxes, fontsize=14, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            ax.set_title(model)
            
            # Set axis limits
            if xlim:
                ax.set_xlim(xlim)
            else:
                ax.set_xlim([df_model[x_col].min() * 0.9, df_model[x_col].max() * 1.1])
                
            if ylim:
                ax.set_ylim(ylim)
            else:
                ax.set_ylim([df_model[y_col].min() * 0.9, df_model[y_col].max() * 1.1])

            # Set x-axis label for bottom row of subplots
            if idx // ncols == nrows - 1 and xlabel:
                ax.set_xlabel(xlabel, fontsize=12)

            # Set y-axis label for left column of subplots
            if idx % ncols == 0 and ylabel:
                ax.set_ylabel(ylabel, fontsize=12)
            else: 
                ax.set_ylabel('')

            # Add legend in bottom right if 1:1 line is shown
            if add_1to1_line:
                ax.legend(loc='lower right')

        else:
            # If model is missing, turn off the subplot (blank space)
            axes[idx].axis('off')

    # Turn off any remaining subplots if model_order has fewer models than subplot spaces
    for i in range(len(model_order), len(axes)):
        axes[i].axis('off')

    # Adjust layout for better visualization
    fig.tight_layout()
    plt.show()


# In[48]:


# Define the model order for plotting
model_order = ['JULES', 'LPJ GUESS', 'LPJwsl', 'LPX Bern']
plot_scatter_fit (df_cv_allveg, 'CV of annual total PFT', 'CV of annual GPP','Aridity' ,'Models', model_order, xlabel = 'CV of annual total PFT', ylabel='CV of annual GPP', add_1to1_line = True)


# In[49]:


allveg_mean = get_mean_pft('allveg', sorted_pft_files, AI)
woody_mean = get_mean_pft('woody_allveg', sorted_pft_files, AI)
nonwoody_mean = get_mean_pft('nonwoody_allveg', sorted_pft_files, AI)
c4grass_mean = get_mean_pft('c4grass', sorted_pft_files, AI)
c3grass_mean = get_mean_pft('c3grass', sorted_pft_files, AI)
allcrop_mean = get_mean_pft('allcrop', sorted_pft_files, AI)
allgrass_mean  = get_mean_pft('allgrass', sorted_pft_files, AI)


# In[50]:


df_mean_allveg = get_df('All Veg Fractional Mean', allveg_mean, 'Slope', slope, AI, gpp_source, model_list, dry_area)
df_mean_woody = get_df('Woody Fractional Mean', woody_mean,'Slope', slope, AI, gpp_source, model_list, dry_area)
df_mean_nonwoody = get_df('Non-woody Fractional Mean', nonwoody_mean, 'Slope', slope, AI, gpp_source, model_list, dry_area)
df_mean_c4grass = get_df('C4 grass Fractional Mean', c4grass_mean, 'Slope', slope, AI, gpp_source, model_list, dry_area)
df_mean_c3grass = get_df('C3 grass Fractional Mean', c3grass_mean, 'Slope', slope, AI, gpp_source, model_list, dry_area)
df_mean_grass = get_df('Grass Fractional Mean', allgrass_mean, 'Slope', slope, AI, gpp_source, model_list, dry_area)
df_mean_crop = get_df('Crop Fractional Mean', allcrop_mean, 'Slope', slope, AI, gpp_source, model_list, dry_area)


# In[51]:


import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Define the GPP source order and colors
gpp_source_order = [
    'CABLE POP', 'CLASSIC', 'CLM5', 'IBIS', 'ISBA CTRIP', 
    'JSBACH', 'JULES', 'ORCHIDEE', 'VISIT', 'YIBs', 
    'LPJ GUESS', 'LPJwsl', 'LPX Bern', 'OCN', 'SDGVM'
]

# Define colors for each model in gpp_source_order: first 10 in brown hues, last 5 in blue hues
colors = sns.color_palette("copper_r", 10) + sns.color_palette("Blues", 5)
model_colors = dict(zip(gpp_source_order, colors))

def plot_mean_cover_vs_gpp(df, cover_col, gpp_metric_col, model_col='Models', threshold=0.5, ax=None, xlabel='', ylabel=''):
    """
    Plot the mean of a specified fractional cover type above a threshold on the x-axis,
    and the mean of a specified GPP metric on the y-axis, for each model, with circle size
    representing the number of pixels and error bars representing the standard deviation 
    for both cover and GPP metrics.
    """
    # Filter for cells where cover_col is greater than the threshold
    df_high_cover = df[df[cover_col] > threshold]

    # Calculate statistics
    mean_cover = df_high_cover.groupby(model_col)[cover_col].mean().reset_index(name='mean_cover')
    std_cover = df_high_cover.groupby(model_col)[cover_col].std().reset_index(name='std_cover')
    mean_gpp = df_high_cover.groupby(model_col)[gpp_metric_col].mean().reset_index(name='mean_gpp_metric')
    std_gpp = df_high_cover.groupby(model_col)[gpp_metric_col].std().reset_index(name='std_gpp_metric')
    pixel_count = df_high_cover.groupby(model_col).size().reset_index(name='pixel_count')

    # Merge results
    model_stats = mean_cover.merge(std_cover, on=model_col).merge(mean_gpp, on=model_col)
    model_stats = model_stats.merge(std_gpp, on=model_col).merge(pixel_count, on=model_col)

    # Plot each model with its specific color
    cover_values = []
    gpp_values = []

    for model in gpp_source_order:
        if model in model_stats[model_col].values:
            model_data = model_stats[model_stats[model_col] == model]
            
            # Scatter plot for mean cover vs mean GPP
            ax.scatter(
                model_data['mean_cover'], model_data['mean_gpp_metric'], 
                s=model_data['pixel_count'], color=model_colors[model], label=model,
                alpha=0.8, linewidths=1, edgecolors='black'
            )
            
            # Collect mean_cover and mean_gpp values for calculating R^2
            cover_values.append(model_data['mean_cover'].values[0])
            gpp_values.append(model_data['mean_gpp_metric'].values[0])

            # Error bars
            ax.errorbar(
                model_data['mean_cover'], model_data['mean_gpp_metric'], 
                xerr=model_data['std_cover'], yerr=model_data['std_gpp_metric'], 
                color=model_colors[model], alpha=0.5
            )

    # Calculate R^2 between mean_cover and mean_gpp for all models
    slope, intercept, r_value, p_value, std_err = stats.linregress(cover_values, gpp_values)
    r_squared = r_value ** 2

    # Set labels for the bottom row and left column
    ax.set_xlabel(xlabel, fontsize=24)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.axhline(1, color='grey', linestyle='--', linewidth=1)

    # Set limits for main or inset plot
    if threshold == 0.1:
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 3.5)
    else:
        ax.set_xlim(0.5, 1)

    return r_squared


# In[52]:


def create_figure_with_inset_plots(df_mean_c4grass, df_mean_c3grass, df_mean_grass, df_mean_woody, gpp_metric_col='Slope', threshold_inset = 0.5):
    fig, axs = plt.subplots(2, 2, figsize=(28, 18))  # Create a 2x2 grid for main plots
    
    # Adjust spacing between subplots
    fig.subplots_adjust(hspace=0.05, wspace=0.05, right=0.85)  # Smaller values for hspace and wspace reduce space

    # Datasets and cover columns
    datasets = [
        (df_mean_woody, 'Woody Fractional Mean', 'a) Woody fCover > 0.1'),
        (df_mean_grass, 'Grass Fractional Mean', 'b) Grass fCover > 0.1'),
        (df_mean_c3grass, 'C3 grass Fractional Mean', 'c) C3 Grass fCover > 0.1'),
        (df_mean_c4grass, 'C4 grass Fractional Mean', 'd) C4 Grass fCover > 0.1')
    ]
    
    # Plot main subplots with threshold 0.1
    for i, (ax, (df, cover_col, title_text)) in enumerate(zip(axs.flatten(), datasets)):
        # Set labels only for plots in the left column and bottom row
        xlabel = 'Spatial mean PFT fCover' if i // 2 == 1 else ''
        ylabel = 'Spatial mean of GPP slope (model, DryFlux)' if i % 2 == 0 else ''
        
        # Call plotting function for main plot
        r_squared = plot_mean_cover_vs_gpp(df, cover_col=cover_col, gpp_metric_col=gpp_metric_col, threshold=0.1, ax=ax, xlabel=xlabel, ylabel=ylabel)
        
        # Add title text at the top-left of each main subplot
        ax.text(0.02, 0.98, title_text, transform=ax.transAxes, ha='left', va='top', fontsize=24, weight='bold')
        
        # Add R^2 value beneath the title text
        ax.text(0.02, 0.92, f'$R^2$: {r_squared:.2f}', transform=ax.transAxes, ha='left', va='top', fontsize=20)
        
        ax.tick_params(labelsize=20)
        
        # Remove x-tick labels for the top row and y-tick labels for the right column
        if i // 2 == 0:  # Top row
            ax.tick_params(labelbottom=False)
        if i % 2 == 1:  # Right column
            ax.tick_params(labelleft=False)

        # Add an inset subplot within each main subplot with threshold 0.5
        inset_location = [0.6, 0.6, 0.35, 0.35]   # Bottom right for the first plot
        inset_ax = ax.inset_axes(inset_location)
        
        # Call plotting function for inset plot
        inset_r_squared = plot_mean_cover_vs_gpp(df, cover_col=cover_col, gpp_metric_col=gpp_metric_col, threshold=threshold_inset, ax=inset_ax)
        
        # Add title text for inset at top-left corner
        inset_ax.text(0.02, 0.98, 'fCover > 0.5', transform=inset_ax.transAxes, ha='left', va='top', fontsize=20)

        # Add R^2 value next to the inset title
        inset_ax.text(0.02, 0.88, f'$R^2$: {inset_r_squared:.2f}', transform=inset_ax.transAxes, ha='left', va='top', fontsize=16)
        
        # Enable tick marks on inset plots and set tick label font size
        inset_ax.tick_params(which='both', direction='in', labelsize=14)
    
    # Create a legend on the right side of the figure with all models in specified order
    handles, labels = [], []
    for model in gpp_source_order:
        # Increase the markersize to adjust the circle size in the legend
        handles.append(plt.Line2D([], [], marker='o', color=model_colors[model], linestyle='', markersize=20))  # Set markersize to a larger value (e.g., 15)
        labels.append(model)

    # Create the legend
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.85, 0.5), fontsize=28)

    plt.show()


create_figure_with_inset_plots(df_mean_c4grass, df_mean_c3grass, df_mean_grass, df_mean_woody, gpp_metric_col='Slope')


# In[ ]:




