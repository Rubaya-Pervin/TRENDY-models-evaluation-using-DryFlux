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
trendy_cv = np.ma.std(TRENDY, axis=1)/np.ma.mean(TRENDY, axis=1)


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


# In[32]:


# Changing the CWD
os.chdir('/Users/rubayapervin/OneDrive - Indiana University/Research Projects/Evaluation_TRENDY_GPP/Data/Trendy/V11/burntArea/')
files = glob.glob('*.nc')
print(files)
model_list = ['CLASSIC', 'CLM5', 'ISBA CTRIP', 'JSBACH', 'JULES', 'LPJ GUESS', 'LPJwsl', 'SDGVM', 'VISIT']
print(model_list)


# In[33]:


# Create a mapping from model name to priority based on model_list
priority = {model: idx for idx, model in enumerate(model_list)}

def get_model_name(filename):
    # Extract the model name from the filename
    model_name = filename.split('_')[0]
    # Handle special cases
    if model_name == 'CLM5.0':
        return 'CLM5'
    elif model_name == 'LPJ-GUESS':
        return 'LPJ GUESS'
    elif model_name == 'LPJ':
        return 'LPJwsl'  # Assuming this is what you meant by LPJwsl
    elif model_name == 'ISBA-CTRIP':
        return 'ISBA CTRIP'
    return model_name

# Sort the files based on their position in model_list
sorted_files = sorted(files, key=lambda x: priority[get_model_name(x)])

print(sorted_files)


# In[34]:


area = np.loadtxt('/Users/rubayapervin/OneDrive - Indiana University/Research Projects/Evaluation_TRENDY_GPP/Data/Area/orch05_grid_area.dat')


# In[35]:


dry_area = get_dryland(AI,area)


# In[36]:


plt.imshow(dry_area)


# In[37]:


file = nc.Dataset(sorted_files[2],'r')
var_names=file.variables.keys() 
print(var_names)


# In[38]:


all_ba = []
for mod in sorted_files: 
    print(mod)
    file = nc.Dataset(mod,'r')
    ba = file.variables['burntArea'][:]
    print(ba.shape)
    # Append the result to all_et list
    all_ba.append(ba)
    print(f"Number of models processed: {len(all_ba)}")


# In[39]:


#As the the data longitude starts from 0 at the begining, we want it at the middle. Therefore we need to shift the data
def shift_x_axis(array,ax=2):
    # Calculate the shift amount
    mid_x_index = array.shape[ax] // 2
    shift_amount = mid_x_index

    # Shift the array along the longitude axis so that the midpoint of the x-axis starts at the beginning
    shifted_array = np.roll(array, -shift_amount, axis=ax)

    return shifted_array


# In[40]:


# lat lon shift for the models where necessary to match GPP alignment 

all_ba[1] = shift_x_axis(all_ba[1], ax=2)
all_ba[4] = shift_x_axis(all_ba[4], ax=2)
all_ba[8] = np.flip(all_ba[8],1)


# In[41]:


#getting lat lon ISBA CTRIP
file= nc.Dataset(sorted_files[2],'r')
var_names=file.variables.keys() 
print(var_names)
dim_names=file.dimensions.keys()
print(dim_names)
#getting lat lon
lat = file.variables['lat_FULL'][:]
lon = file.variables['lon_FULL'][:]
print(lat.shape)


# In[42]:


#global scale to align GPP
#change lat shape
new_lat = np.linspace(-89.5, 89.5, 180)
#changing data array shape
# Step 1: Create the new empty masked array with the desired shape (16, 19, 180, 360)
new_shape = (17, 180, 360)
new_ba = np.ma.masked_all(new_shape)

# Step 2: Find the indices of the original latitudes in the new lat array
lat_indices = np.searchsorted(new_lat, lat)

# Step 3: Fill the new_data array with values from the old data at the correct latitude indices
for i, idx in enumerate(lat_indices):
    new_ba[:, idx, :] = all_ba[2][:, i, :]
all_ba[2] = new_ba
print(all_ba[2].shape)


# In[43]:


#resize to allign GPP
for n,x in enumerate(all_ba):
    data = x
    # Resize and handle only non-None layers
    ba = np.array([
    st.resize(layer, (360, 720), order=0, preserve_range=True, anti_aliasing=False)
    for layer in data
    ])


# In[44]:


#getting lat lon
file= nc.Dataset(sorted_files[5],'r')
var_names=file.variables.keys() 
print(var_names)
dim_names=file.dimensions.keys()
print(dim_names)
lat = file.variables['latitude'][:]
lon = file.variables['longitude'][:]
lon, lat = np.meshgrid(lon, lat)
print(lat.shape)


# In[46]:


def ba_sd_plot(all_ba, model_list, CB_label='burntArea sd'):
    fig = plt.figure(figsize=(30, 46))
    model_order = ['CLASSIC', 'CLM5', 'ISBA CTRIP', 'JSBACH', 'JULES', 'LPJ GUESS', 'LPJwsl', 'SDGVM', 'VISIT']
    ncols = 3
    nrows = 3
    spec = fig.add_gridspec(ncols=ncols, nrows=nrows, wspace=0.05, hspace=0.0)
    plt.rc('axes', titlesize=50)
    plt.rc('legend', fontsize=40)
    plt.rc('font', size=40)
    plt.rc('axes', labelsize=40)
    plt.rc('xtick', labelsize=40)
    plt.rc('ytick', labelsize=40)
    labels = label_generator(case='lowercase letter', start='(', end=')')

    cmap = plt.cm.YlOrRd
    norm = colors.Normalize(vmin=0, vmax=20)

    for i in range(nrows):
        for j in range(ncols):
            idx = i * ncols + j
            ax = fig.add_subplot(spec[i, j], projection=ccrs.PlateCarree())
            ax.coastlines()
            ax.add_feature(cartopy.feature.STATES, edgecolor='black', linewidth=1)
            ax.set_global()
            ax.set_ylim(15, 55)
            ax.set_xlim(-125, -97)
            gl = ax.gridlines(draw_labels=True, xlocs=[-120, -110, -100, -90], ylocs=[15, 25, 35, 45, 55],
                                  linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            if j != 0:
                gl.left_labels = False
            if i != nrows - 1:
                gl.bottom_labels = False
            #ax.text(-125, 56, '%s' % (next(labels)), fontsize=50)
            ax.set_title(model_order[idx], fontsize=50)

            # Check if model is in the model_list
            if model_order[idx] in model_list:
                model_index = model_list.index(model_order[idx])
                #print(idx,model_index)
                #print(all_ba[model_index].shape)
                data = all_ba[model_index]
                # Resize and handle only non-None layers
                ba = np.array([
                    st.resize(layer, (360, 720), order=0, preserve_range=True, anti_aliasing=False)
                    for layer in data
                        ])                            
                sd = np.ma.std(ba, axis=0)
                s = ax.pcolormesh(lon, lat, np.flip(get_dryland(AI,np.flip(sd,0)),0), cmap=cmap, norm=norm)
                #ax.set_ylabel('g C m^-2')

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.22, 0.01, 0.55])
    cb = fig.colorbar(s, cax=cbar_ax, cmap=cmap, norm=norm, label=CB_label)
    cb.ax.tick_params(labelsize=30)

    plt.tight_layout()
    plt.show()


# In[47]:


ba_sd_plot(all_ba, model_list, CB_label='SD (annual burnt area, 2001-2016)')


# In[50]:


def get_sd_ba(all_ba, AI):
    sd_ba = []  # Initialize an empty list to store the cv results.
    
    # Loop over each file in pft_files
    for idx, mod in enumerate(all_ba):
        # Resize and handle only non-None layers
        ba = np.array([
                    st.resize(layer, (360, 720), order=0, preserve_range=True, anti_aliasing=False)
                    for layer in mod
                        ])                            
        ba_sd = np.ma.std(ba, axis=0) if ba is not None else None            
        # getting North American Dryland 
        sd = get_dryland(AI,np.flip(ba_sd,0))
        #plt.imshow(sd)
        #plt.show()
        # Append the result to the list (if valid data was found).
        sd_ba.append(sd)
    return sd_ba


# In[51]:


ba_sd = get_sd_ba(all_ba, AI)


# In[52]:


def get_df(x_type, x_data, metric_name, metric_data, AI, gpp_source, model_list, dry_area=dry_area):
    df = pd.DataFrame(columns=['Models', x_type, metric_name, 'Aridity', 'Area'])
    models = []
    x_dt = []
    metric_dt = []
    aridity = []
    area = []
    
    for idx, model in enumerate(model_list):
        # Check if the model exists in gpp_source
        if model not in gpp_source:
            print(f"Model {model} not in gpp_source, adding empty values for it.")
            models.append(model)
            aridity.append('No Data')
            x_dt.append(0)
            metric_dt.append(0)
            area.append(0)
            continue

        gpp_source_index = gpp_source.index(model)

        # Check x_data and metric_data validity
        if idx >= len(x_data) or x_data[idx] is None:
            print(f"pft_data[{idx}] is None or out of range, adding empty values for {model}.")
            models.append(model)
            aridity.append('No Data')
            x_dt.append(0)
            metric_dt.append(0)
            area.append(0)
            continue
        
        if gpp_source_index >= len(metric_data) or metric_data[gpp_source_index] is None:
            print(f"metric_data[{gpp_source_index}] is None or out of range, adding empty values for {model}.")
            models.append(model)
            aridity.append('No Data')
            x_dt.append(0)
            metric_dt.append(0)
            area.append(0)
            continue

        # Check if the shape is valid
        if x_data[idx].shape != (360, 720):
            print(f"pft_data[{idx}] has incorrect shape: {pft_data[idx].shape}, adding empty values for {model}.")
            models.append(model)
            aridity.append('No Data')
            x_dt.append(0)
            metric_dt.append(0)
            area.append(0)
            continue
        
        if metric_data[gpp_source_index].shape != (360, 720):
            print(f"metric_data[{gpp_source_index}] has incorrect shape: {metric_data[gpp_source_index].shape}, adding empty values for {model}.")
            models.append(model)
            aridity.append('No Data')
            x_dt.append(0)
            metric_dt.append(0)
            area.append(0)
            continue

        # Apply masking
        x_data[idx] = masking(x_data[idx], metric_data[gpp_source_index])
        metric_data[gpp_source_index] = masking(metric_data[gpp_source_index], x_data[idx])
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
                    x_dt.append(x_data[idx][nlat, nlon])
                    metric_dt.append(metric_data[gpp_source_index][nlat, nlon])
                    area.append(dry_area[nlat, nlon])
                elif not test_mask[nlat, nlon] and 0.2 < AI[nlat, nlon] <= 0.5:
                    models.append(model)
                    aridity.append('Semi-Arid')
                    x_dt.append(x_data[idx][nlat, nlon])
                    metric_dt.append(metric_data[gpp_source_index][nlat, nlon])
                    area.append(dry_area[nlat, nlon])
    
    # Ensure consistency across all lists
    min_len = min(len(models), len(aridity), len(x_dt), len(metric_dt), len(area))
    models, aridity, x_dt, metric_dt, area = models[:min_len], aridity[:min_len], x_dt[:min_len], metric_dt[:min_len], area[:min_len]
    
    # Populate DataFrame
    df['Models'] = models
    df['Aridity'] = aridity
    df[x_type] = x_dt
    df[metric_name] = metric_dt
    df['Area'] = area
    
    # Convert to float type
    df[x_type] = df[x_type].astype('float')
    df[metric_name] = df[metric_name].astype('float')
    df['Area'] = df['Area'].astype('float')
    
    return df


# In[53]:


df_sd = get_df('SD(annual burnt area, 2001-2016)', ba_sd, 'SD(annual GPP, 2001-2016)', trendy_sd, AI, gpp_source, model_list, dry_area)


# In[54]:


import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def plot_scatter_sd_fit(df, x_col, y_col, aridity_col, model_col, model_order, xlabel=None, ylabel=None):
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
    """
    # Number of columns and rows
    ncols = 3
    nrows = (len(model_order) // ncols) + (len(model_order) % ncols > 0)  # Calculate number of rows needed
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(30, 30))
    axes = axes.flatten()  # Flatten the axes array

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
            
            # Perform linear regression to get the fit line
            slope_line, intercept_line, r_value, p_value, std_err = stats.linregress(df_model[x_col], df_model[y_col])
            print(model, ':', p_value)
            fit_line = slope_line * df_model[x_col] + intercept_line
            ax.plot(df_model[x_col], fit_line, color='red', linestyle='--')
            
            r_squared = r_value ** 2
            # Format p-value to display more decimal places or in scientific notation
            #p_value_formatted = f'{p_value:.3e}' if p_value < 1e-3 else f'{p_value:.5f}'
            
            # Display R-squared and p-value on the plot
            ax.text(0.05, 0.95, f'$R^2$: {r_squared:.2f}', #\n$p$: {p_value_formatted}', 
                    transform=ax.transAxes, fontsize=24, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            ax.set_title(model, fontsize=34)
            
            # Set x and y limits
            ax.set_xlim(0, 20)
            ax.set_ylim(0, 0.5)

            # Set x-axis label for bottom row of subplots
            if idx // ncols == nrows - 1:  # Bottom row
                if xlabel:
                    ax.set_xlabel(xlabel, fontsize=24)
            else:
                ax.set_xlabel('')  # Turn off x-label for non-bottom row

            # Set y-axis label for left column of subplots
            if idx % ncols == 0:  # Left column
                if ylabel:
                    ax.set_ylabel(ylabel, fontsize=24)
            else:
                ax.set_ylabel('')  # Turn off y-label for non-left column

            # Adjust tick label size
            ax.tick_params(axis='both', which='major', labelsize=20)

        else:
            # If model is missing, turn off the subplot (blank space)
            axes[idx].axis('off')

    # Turn off any remaining subplots if model_order has fewer models than subplot spaces
    for i in range(len(model_order), len(axes)):
        axes[i].axis('off')

    # Adjust layout for better visualization
    fig.tight_layout()
    plt.show()


# In[55]:


plot_scatter_sd_fit(df_sd, 'SD(annual burnt area, 2001-2016)', 'SD(annual GPP, 2001-2016)', 'Aridity', 'Models', model_list, xlabel='SD(annual burnt area, 2001-2016)', ylabel='SD(annual GPP, 2001-2016)')


# In[ ]:




