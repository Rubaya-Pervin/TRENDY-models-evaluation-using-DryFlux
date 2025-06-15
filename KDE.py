#!/usr/bin/env python
# coding: utf-8

# In[94]:


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
from scipy.stats import linregress
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import statsmodels.formula.api as smf
from matplotlib.patches import Rectangle
import skimage.transform as st
from scipy.ndimage import zoom


# In[95]:


#from daily mean monthly to get daily mean annual  
def ann_mean(data,nyears,nlat,nlon):
    years = np.linspace(2001, 2016, nyears)
    ann_mean = np.zeros((nyears,nlat,nlon))
    ann_mean = np.ma.masked_values (ann_mean,0)
    for ny, yy in enumerate(years):       
        ann_mean[ny,:,:] = np.ma.mean(data[ny*12:ny*12+12, :, :], axis=0)
    return ann_mean #daily gpp


# In[96]:


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


# In[97]:


# function to create masked array
def masking(ar1, ar2):
    res_mask = np.ma.getmask(ar2)
    masked = np.ma.masked_array(ar1, mask=res_mask)
    return masked


# In[98]:


#functions to get regions 
def North_America (lat,lon,dt):
    m1 = np.ma.masked_where(lon<-125, dt )
    m2 = np.ma.masked_where(lon>-93, m1 )
    m3= np.ma.masked_where(lat<-53, m2 )
    m4= np.ma.masked_where(lat>-15, m3 )
    return m4


# In[99]:


def get_extent(fn):
    "'returns min_x, max_y, max_x, min_y'"
    ds = gdal.Open(fn)
    gt = ds.GetGeoTransform()
    return (gt[0], gt[3], gt[0]+gt[1]*ds.RasterXSize, gt[3]+gt[5]*ds.RasterYSize)


# In[100]:


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


# In[101]:


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


# In[102]:


#getting dryland data
def get_dryland(AI,data):
    m1 = np.ma.masked_where(AI>0.5, data)
    m2 = np.ma.masked_where(AI<0.05, m1 )
    #masking out cold dryland
    #m3 = np.ma.masked_where(lat<-55, m2 )
    #dryland = np.ma.masked_where(lat>55, m3)
    return m2


# In[103]:


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


# In[104]:


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


# In[105]:


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


# In[106]:


#crop mask and gpp data list 
mask_list = [ 'CLASSIC', 'CLM5','IBIS', 'ISBA CTRIP', 'JSBACH', 'JULES', 'LPJ GUESS', 'LPX Bern', 'OCN', 'ORCHIDEE', 'SDGVM', 'VISIT','YIBs']
gpp_source = ['CABLE POP', 'CLASSIC', 'CLM5', 'IBIS', 'ISBA CTRIP', 'JSBACH', 'JULES', 'LPJ GUESS', 'LPJwsl', 'LPX Bern', 'OCN', 'ORCHIDEE', 'SDGVM', 'VISIT', 'YIBs']


# In[107]:


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


# In[108]:


crop_mask_files =glob.glob('/Users/rubayapervin/OneDrive - Indiana University/Research Projects/Evaluation_TRENDY_GPP/Data/Trendy/V11/PFT/TRENDY-v11_cropMasked/mod_masks/*')
for x  in crop_mask_files: 
    print(x)


# In[109]:


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


# In[110]:


# Flip the mask for TRENDY data orientation
common_mask_flipped = np.flip(common_mask, axis=0)

# Apply the common mask to all models and years
for idx in range(len(gpp_source)):  # Loop over all models
    for nyear in range(TRENDY.shape[1]):  # Loop over all years
        TRENDY[idx, nyear, :, :] = np.ma.masked_where(
            common_mask_flipped, 
            TRENDY[idx, nyear, :, :]  
        )


# In[111]:


#get north american drynad 
#get AI data in north America
AI_N = North_America (lat,lon,scl_AI)
AI = get_dryland(scl_AI, AI_N)


# In[112]:


#Getting dryland (arid and semi-arid regions)
dryland = get_dryland(scl_AI,scl_AI)
#dryland = masking(dryland, region_mask)


# In[113]:


# applying mask to keep common pixels of dryflux, trendy, and AI data
dryflux = masking(Dryflux, np.ma.mean(TRENDY,axis=0))
for nmod in np.arange(np.shape(TRENDY)[0]):
    TRENDY[nmod,:,:,:] = masking(TRENDY[nmod,:,:,:], dryflux)
dryflux_mean_gpp = np.ma.mean(dryflux, axis=0)
AI = masking(AI, dryflux_mean_gpp)


# In[114]:


# getting north american dryland dryflux gpp
for nyear in np.arange(np.shape(dryflux)[0]):
    dryflux[nyear] = get_dryland(AI,dryflux[nyear])
# getting north american dryland trendy gpp
for nmod in np.arange(np.shape(TRENDY)[0]):
    for nyear in np.arange(np.shape(TRENDY)[1]):
        TRENDY[nmod,nyear,:,:] = get_dryland(AI,TRENDY[nmod,nyear,:,:])
print(TRENDY.shape)


# In[115]:


# getting trendy mean gpp
trendy_mean_gpp = np.ma.mean(TRENDY, axis=1)
dryflux_mean_gpp = np.ma.mean(dryflux, axis=0)


# In[116]:


#getting standard deviation
dry_sd =np.ma.std(dryflux, axis=0)
trendy_sd = np.ma.std(TRENDY, axis=1)


# In[117]:


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


# In[118]:


r2 = r*r


# In[119]:


# Define the order to plot the GPP sources
gpp_source_order = ['CABLE POP', 'CLASSIC', 'CLM5', 'IBIS', 'ISBA CTRIP', 'JSBACH', 'JULES', 'ORCHIDEE', 'VISIT', 'YIBs', 'LPJ GUESS', 'LPJwsl', 'LPX Bern', 'OCN', 'SDGVM']
valid_source = ['CLASSIC', 'CLM5','JULES', 'ORCHIDEE',  'JSBACH',  'ISBA CTRIP','CABLE POP','YIBs', 'IBIS', 'VISIT','SDGVM', 'LPJ GUESS', 'LPX Bern', 'OCN','LPJwsl']


# Create a mapping from GPP source to its index
gpp_source_mapping = {source: i for i, source in enumerate(gpp_source)}


# In[120]:


area = np.loadtxt('/Users/rubayapervin/OneDrive - Indiana University/Research Projects/Evaluation_TRENDY_GPP/Data/Area/orch05_grid_area.dat')
dry_area = get_dryland(AI,area)


# In[121]:


#this is actually for plotting SD
def plot_area_weighted_ridge(slope, area, gpp_source_order, bins=50, figsize=(10, 6), colors=None, text=None, vline=None, xlim=None, ylim=None):
    """
    Plot an area-weighted KDE plot for multiple models.

    Parameters:
    - slope: 3D numpy array of slope values (models, lat, lon)
    - area: 2D numpy array of area values (lat, lon)
    - bins: int, number of bins for the histogram (optional for KDE)
    - figsize: tuple, size of the figure
    - xlim: tuple, limits for the x-axis (min, max)
    - ylim: tuple, limits for the y-axis (min, max)
    """
    # Number of models
    num_models = slope.shape[0]+1
    last_plot_color='#424242'

    # Initialize the plot
    fig, axs = plt.subplots(nrows=num_models, ncols=1, figsize=figsize)
    axs = axs.flatten()
    style = ['--', '--', '--', '--', '--', '--', '--', '--', '--', '--', '-', '-', '-', '-', '-']
    
    for model_index in range(num_models):
        if model_index < num_models - 1: 
            gpp_source_index = gpp_source_mapping.get(gpp_source_order[model_index])
            # Flatten the slope and area arrays for the current model
            slope_flat = slope[gpp_source_index].flatten()
            area_flat = area.flatten()
        
            mask = (~np.isnan(slope_flat)) & (~np.isnan(area_flat))
        
            if isinstance(slope_flat, np.ma.MaskedArray):
                mask = mask & (~slope_flat.mask)
            if isinstance(area_flat, np.ma.MaskedArray):
                mask = mask & (~area_flat.mask)
        
            slope_flat = slope_flat[mask]
            area_flat = area_flat[mask]
        
            if isinstance(slope_flat, np.ma.MaskedArray):
                slope_flat = slope_flat.data
            if isinstance(area_flat, np.ma.MaskedArray):
                area_flat = area_flat.data
        
            # Create a DataFrame for proper weight handling
            df = pd.DataFrame({
            'slope': slope_flat,
            'weights': area_flat
            })
        
            bandwidth = 1  # Control smoothness
            sns.kdeplot(
            data=df,
            x='slope',
            weights='weights',
            fill=True,
            color=colors[model_index],
            label=gpp_source[gpp_source_index],
            alpha=0.7,
            linestyle=style[model_index],
            linewidth=1,
            bw_adjust=bandwidth,
            ax=axs[model_index]
            )
        if model_index == num_models - 1:
            dry_flat = dry_sd.flatten()
            dry_flat = dry_flat[mask]
        
            # Ensure we have regular numpy arrays
            if isinstance(dry_flat, np.ma.MaskedArray):
                dry_flat = dry_flat.data
        
            # Create a DataFrame for proper weight handling
            df = pd.DataFrame({
            'dry_flat': dry_flat,
            'weights': area_flat
            })
            sns.kdeplot(df,
                    x= 'dry_flat',
                    weights=area_flat,
                    shade=True,
                    color=last_plot_color,  
                    label='DryFlux',
                    alpha=0.7,
                    #linestyle=style[model_index],
                    linewidth=1,
                    bw_adjust=bandwidth,
                    ax=axs[model_index]
                    )
        
        # compute quantiles
        quantiles = np.percentile(slope_flat, [2.5, 10, 25, 75, 90, 97.5])
        quantiles = quantiles.tolist()

        # fill space between each pair of quantiles
        for j in range(len(quantiles) - 1):
            axs[model_index].fill_between(
                [quantiles[j], # lower bound
                 quantiles[j+1]], # upper bound
                0, # max y=0
                0.0002, # max y=0.0002
                color='purple'
            )
        
        # slope 1 reference line
        if vline is not None:
            axs[model_index].axvline(vline, linestyle='--')
    
        # Set x-axis and y-axis limits if provided
        axs[model_index].set_xlim(xlim)
        axs[model_index].set_ylim(ylim)
        axs[model_index].set_ylabel('')
        
        # x axis scale for last ax
        if model_index == num_models-1:
            values = np.linspace(xlim[0], xlim[1], num=5)
            for value in values:
                axs[model_index].text(
                    value, -ylim[1]/3,
                    f'{value}',
                    ha='center',
                    fontsize=20
                )
        
        
        # remove axis
        axs[model_index].set_axis_off()
        
        # Display models on the left
        if model_index == num_models - 1:
            # For the last plot, add "DryFlux" as the model name
            axs[model_index].text(
            xlim[1]-0.5, 0.5,
            "DryFlux".upper(),  # Replace with "DryFlux" for the last plot
            ha='left',
            fontsize=20
        )
        else:
        # For other models, keep the original names
            axs[model_index].text(
            xlim[1]-0.5, 0.5,
            gpp_source[gpp_source_index].upper(),
            ha='left',
            fontsize=20
        )

        
           
    # x axis label
    if text is not None:
        text = text
    fig.text(
        0.5, 0.08,
        text,
        ha='center',
        fontsize=20
    )
    
    
    if xlim is not None:
        plt.xlim(0,1)
    if ylim is not None:
        plt.ylim(ylim)

    #plt.legend(loc='upper right')
    #plt.grid(True)
    plt.show()


# In[122]:


low_sd = '#BB8EE8'
high_sd = '#003366'
SD_colors = [low_sd, low_sd , low_sd,low_sd, low_sd, low_sd, low_sd, low_sd, low_sd, low_sd, high_sd, high_sd , high_sd, high_sd, high_sd]
plot_area_weighted_ridge(trendy_sd, dry_area, gpp_source_order, bins=10, figsize=(5, 20), colors = SD_colors, text = 'GPP Standard Deviation', vline = None, xlim=(0,1), ylim=(0, 12))


# In[123]:


#plotting slope
def plot_area_weighted_ridge(slope, area, gpp_source_order,  bins=50, figsize=(10, 6), colors=None, text=None, vline=None, xlim=None, ylim=None):

    # Number of models
    num_models = slope.shape[0]
    
    # Initialize the plot
    fig, axs = plt.subplots(nrows=num_models, ncols=1, figsize=figsize)
    axs = axs.flatten()
    style = ['--', '--', '--', '--', '--', '--', '--', '--', '--', '--', '-', '-', '-', '-', '-']
    
    for model_index in range(num_models):
        gpp_source_index = gpp_source_mapping.get(gpp_source_order[model_index])
        # Flatten the slope and area arrays for the current model
        slope_flat = slope[gpp_source_index].flatten()
        area_flat = area.flatten()
        
        # Create a single mask for both arrays
        mask = (~np.isnan(slope_flat)) & (~np.isnan(area_flat))
        
        # Handle masked arrays if present
        if isinstance(slope_flat, np.ma.MaskedArray):
            mask = mask & (~slope_flat.mask)
        if isinstance(area_flat, np.ma.MaskedArray):
            mask = mask & (~area_flat.mask)
        
        # Apply the mask to both arrays
        slope_flat = slope_flat[mask]
        area_flat = area_flat[mask]
        
        # Ensure we have regular numpy arrays
        if isinstance(slope_flat, np.ma.MaskedArray):
            slope_flat = slope_flat.data
        if isinstance(area_flat, np.ma.MaskedArray):
            area_flat = area_flat.data
        
        # Create a DataFrame for proper weight handling
        import pandas as pd
        df = pd.DataFrame({
            'slope': slope_flat,
            'weights': area_flat
        })
        
        bandwidth = 1  # Control smoothness
        sns.kdeplot(
            data=df,
            x='slope',
            weights='weights',
            fill=True,
            color=colors[model_index],
            label=gpp_source[gpp_source_index],
            alpha=0.7,
            linestyle=style[model_index],
            linewidth=1,
            bw_adjust=bandwidth,
            ax=axs[model_index]
        )
        
        # Compute quantiles
        quantiles = np.percentile(slope_flat, [2.5, 10, 25, 75, 90, 97.5])
        
        # Fill space between quantiles
        for j in range(len(quantiles) - 1):
            axs[model_index].fill_between(
                [quantiles[j], quantiles[j+1]],
                0,
                0.0002,
                color=colors[model_index],
                alpha=0.3
            )
        
        if vline is not None:
            axs[model_index].axvline(vline, color='#525252', linestyle='--')
        
        # Set axis limits
        if xlim:
            axs[model_index].set_xlim(xlim)
        if ylim:
            axs[model_index].set_ylim(ylim)
        axs[model_index].set_ylabel('')
        
        # X-axis scale for last ax
        if model_index == num_models-1 and xlim:
            values = np.linspace(xlim[0], xlim[1], num=5)
            for value in values:
                axs[model_index].text(
                    value, -ylim[1]/3 if ylim else -0.0002/3,
                    f'{value}',
                    ha='center',
                    fontsize=22
                )
        
        # Remove axis
        axs[model_index].set_axis_off()
        
        # Display model names
        axs[model_index].text(
            xlim[1]-1.25 if xlim else 0, 0.5,
            gpp_source[gpp_source_index].upper(),
            ha='left',
            fontsize=22
        )
    
    # X-axis label
    if text is not None:
        fig.text(
            0.5, 0,
            text,
            ha='center',
            fontsize=22
        )

    plt.tight_layout()
    plt.show()


# In[124]:


low_slope =  '#a0826e'
high_slope = '#48b2bf'
slope_colors = [low_slope, low_slope, low_slope,low_slope, low_slope, low_slope, low_slope, low_slope, low_slope, low_slope, high_slope, high_slope , high_slope, high_slope, high_slope]
plot_area_weighted_ridge(slope, dry_area, gpp_source_order, bins=10, figsize=(5, 20), colors =slope_colors, text = 'Slope', vline = 1, xlim=(-1, 3), ylim=(0, 2))


# In[36]:


def create_combined_plot(slope, dry_area, figsize=(12, 15)):
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    plt.subplots_adjust(wspace=0.3)
    
    # Define color schemes and model orders
    darkgreen = '#9BC184'
    lightgreen = '#E7E5CB'
    BrightOrange = '#FF8C00'
    
    # Dynamic Vegetation Plot (left)
    dynamic_colors = [lightgreen]*11 + [darkgreen]*4
    dynamic_order = ['CABLE POP', 'CLASSIC', 'CLM5', 'IBIS', 'ISBA CTRIP', 
                    'JSBACH', 'OCN', 'ORCHIDEE', 'SDGVM', 'VISIT', 
                    'YIBs', 'JULES', 'LPJ GUESS', 'LPJwsl', 'LPX Bern']
    dynamic_info = ['Off']*11 + ['On']*4
    
    # Fire Plot (right)
    Fire_colors = [lightgreen]*5 + [BrightOrange]*10
    Fire_order = ['CABLE POP', 'IBIS', 'OCN', 'ORCHIDEE', 'YIBs', 
                 'CLASSIC', 'CLM5', 'ISBA CTRIP', 'JSBACH', 'JULES', 
                 'LPJ GUESS', 'LPJwsl', 'LPX Bern', 'SDGVM', 'VISIT']
    Fire_info = ['Off']*5 + ['On']*10
    
    # Plot Dynamic Vegetation (left)
    plot_single_panel(ax1, slope, dry_area, dynamic_order, dynamic_info, 
                     dynamic_colors, 'a)', 'Dyn Veg', vline=1, 
                     xlim=(-1, 3), ylim=(0, 2))
    
    # Plot Fire (right)
    plot_single_panel(ax2, slope, dry_area, Fire_order, Fire_info, 
                     Fire_colors, 'b)', 'Fire', vline=1, 
                     xlim=(-1, 3), ylim=(0, 2))
    
    # Add common x-axis label
    fig.text(0.25, 0, '(Dynamic Vegetation)', ha='center', fontsize=20)
    fig.text(0.73, 0, '(Fire)', ha='center', fontsize=20)
    plt.tight_layout()
    plt.show()

def plot_single_panel(ax, slope, area, model_order, info, colors, title, process,  
                     vline=None, xlim=None, ylim=None):
    """Helper function to plot a single panel"""
    num_models = slope.shape[0]
    style = ['--', '--', '--', '--', '--', '--', '--', '--', '--', '--', 
             '-', '-', '-', '-', '-']
    
    # Find transition between Off/On
    transition_index = info.index('On') if 'On' in info else None
    
    for model_index in range(num_models):
        # Create subplot for this model
        model_ax = ax.inset_axes([0, (num_models-model_index-1)/num_models, 1, 1/num_models])
        
        # Get data processing (same as before)
        gpp_source_index = gpp_source_mapping.get(model_order[model_index])
        slope_flat = slope[gpp_source_index].flatten()
        area_flat = area.flatten()
        mask = (~np.isnan(slope_flat)) & (~np.isnan(area_flat))
        slope_flat = slope_flat[mask]
        area_flat = area_flat[mask]
        
        # Create DataFrame and plot KDE
        df = pd.DataFrame({'GPP Slope': slope_flat, 'weights': area_flat})
        sns.kdeplot(data=df, x='GPP Slope', weights='weights', fill=True,
                   color=colors[model_index], alpha=0.7,
                   linestyle=style[model_index], linewidth=1,
                   bw_adjust=1, ax=model_ax)
        
        # Add model name
        model_ax.text(2.5 if xlim else 0, 0.5, 
                     gpp_source[gpp_source_index].upper(),
                     ha='left', fontsize=16)
        
        # Add divider line between Off/On groups
        if transition_index and model_index == transition_index:
            model_ax.axhline(1.4, color='red', linestyle='--', linewidth=1.5, 
                            xmin=0, xmax=1, clip_on=False)
            model_ax.text(-0.95, 1.5, f' {process} Off',
                         ha='left', fontsize=16)
            model_ax.text(-0.95, 1, f' {process} On',
                         ha='left', fontsize=16)
        
        # Formatting
        model_ax.set_xlim(xlim)
        if model_index == num_models-1 and xlim:
          model_ax.xaxis.set_visible(True) 
        else: model_ax.xaxis.set_visible(False)
        model_ax.set_ylim(ylim)
        model_ax.yaxis.set_visible(False)
        model_ax.set_yticks([])  # Remove y-axis ticks
        model_ax.spines['top'].set_visible(False)
        model_ax.spines['left'].set_visible(False)  # Hide left spine (y-axis)
        model_ax.spines['right'].set_visible(False)  # Hide right spine
        if vline:
            model_ax.axvline(vline, color='#525252', linestyle='--')
    
    # Add panel title
    ax.text(0.05, 1.01, title, fontsize=22, transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

# Run the combined plot
create_combined_plot(slope, dry_area, figsize=(12, 15))





