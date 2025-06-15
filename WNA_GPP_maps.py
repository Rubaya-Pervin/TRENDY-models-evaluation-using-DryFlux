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
import cartopy.feature as cfeature
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# In[2]:


#function from daily mean monthly to get daily mean annual  
def ann_mean(data,nyears,nlat,nlon):
    years = np.linspace(2001, 2016, nyears)
    ann_mean = np.zeros((nyears,nlat,nlon))
    ann_mean = np.ma.masked_values (ann_mean,0)
    for ny, yy in enumerate(years):       
        ann_mean[ny,:,:] = np.ma.mean(data[ny*12:ny*12+12, :, :], axis=0)
    return ann_mean #daily gpp


# In[3]:


#function from daily mean monthly to get annual  
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


# function to get extent 
def get_extent(fn):
    "'returns min_x, max_y, max_x, min_y'"
    ds = gdal.Open(fn)
    gt = ds.GetGeoTransform()
    return (gt[0], gt[3], gt[0]+gt[1]*ds.RasterXSize, gt[3]+gt[5]*ds.RasterYSize)


# In[7]:


#function to generate labels
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


# In[16]:


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


# In[17]:


#crop mask and gpp data list 
mask_list = [ 'CLASSIC', 'CLM5','IBIS', 'ISBA CTRIP', 'JSBACH', 'JULES', 'LPJ GUESS', 'LPX Bern', 'OCN', 'ORCHIDEE', 'SDGVM', 'VISIT','YIBs']
gpp_source = ['CABLE POP', 'CLASSIC', 'CLM5', 'IBIS', 'ISBA CTRIP', 'JSBACH', 'JULES', 'LPJ GUESS', 'LPJwsl', 'LPX Bern', 'OCN', 'ORCHIDEE', 'SDGVM', 'VISIT', 'YIBs']


# In[18]:


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


# In[19]:


crop_mask_files =glob.glob('/Users/rubayapervin/OneDrive - Indiana University/Research Projects/Evaluation_TRENDY_GPP/Data/Trendy/V11/PFT/TRENDY-v11_cropMasked/mod_masks/*')
for x  in crop_mask_files: 
    print(x)


# In[21]:


# Initialize list to store masks
all_masks = []

# Loop through each model in gpp_source
for mod in gpp_source:
    if mod in mask_list:  
        mask_index = mask_list.index(mod)
        mask = np.load(crop_mask_files[mask_index])['mask']
        
        # Resize mask to (360, 720) if needed 
        if mask.shape != (360, 720):
            zoom_factors = (360 / mask.shape[0], 720 / mask.shape[1])
            mask = zoom(mask, zoom_factors, order=0)  # order=0 for nearest-neighbor
        
        all_masks.append(mask)

# Create common_mask (True where ANY model's mask is True)
if all_masks:
    common_mask = np.any(np.stack(all_masks, axis=0), axis=0)
else:
    common_mask = np.zeros((360, 720), dtype=bool)  # No masking if no masks exist


# In[22]:


# Flip the mask for TRENDY data orientation
common_mask_flipped = np.flip(common_mask, axis=0)

# Apply the common mask to all models and years
for idx in range(len(gpp_source)):  # Loop over all models
    for nyear in range(TRENDY.shape[1]):  # Loop over all years
        TRENDY[idx, nyear, :, :] = np.ma.masked_where(
            common_mask_flipped, 
            TRENDY[idx, nyear, :, :]  
        )


# In[23]:


# applying mask to keep common pixels of dryflux, trendy, and AI data
dryflux = masking(Dryflux, np.ma.mean(TRENDY,axis=0))
for nmod in np.arange(np.shape(TRENDY)[0]):
    TRENDY[nmod,:,:,:] = masking(TRENDY[nmod,:,:,:], dryflux)
dryflux_mean_gpp = np.ma.mean(dryflux, axis=0)
AI = masking(scl_AI, dryflux_mean_gpp)


# In[24]:


#get north american drynad 
#get AI data in north America
AI_N = North_America (lat,lon,scl_AI)
AI = get_dryland(scl_AI, AI_N)


# In[25]:


#Getting dryland study area with crop masked
dryland = get_dryland(scl_AI,scl_AI)
dl = np.ma.masked_where(
            common_mask_flipped,  
            AI  
        )
# Save the data and mask
np.savez('/Users/rubayapervin/OneDrive - Indiana University/Research Projects/Evaluation_TRENDY_GPP/Data/Dryland/studyarea.npz',data = dl.data)
plt.imshow(dl)


# In[26]:


# getting north american dryland dryflux gpp
for nyear in np.arange(np.shape(dryflux)[0]):
    dryflux[nyear] = get_dryland(AI,dryflux[nyear])
# getting north american dryland trendy gpp
for nmod in np.arange(np.shape(TRENDY)[0]):
    for nyear in np.arange(np.shape(TRENDY)[1]):
        TRENDY[nmod,nyear,:,:] = get_dryland(AI,TRENDY[nmod,nyear,:,:])
print(TRENDY.shape)


# In[37]:


fig = plt.figure(figsize=(80, 60))
spec = fig.add_gridspec(ncols=4, nrows=5)

# Set global font sizes
plt.rcParams.update({
    'axes.titlesize': 30,
    'legend.fontsize': 20,
    'font.size': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20
})

# Create the map axis
ax = fig.add_subplot(spec[0, 0], projection=ccrs.PlateCarree())

states = cfeature.STATES.with_scale('10m')
ax.add_feature(states, facecolor='#f0f0f0', edgecolor='none', zorder=1)  

cmap = colors.ListedColormap([
    '#a0522d',  # Medium brown 
    '#deb887'   # Light brown 
])
bounds = [0.05, 0.2, 0.5]
norm = colors.BoundaryNorm(bounds, cmap.N)

s = ax.pcolormesh(lon, lat, np.flip(dl, 0), 
                 cmap=cmap, norm=norm,
                 transform=ccrs.PlateCarree(),
                 zorder=2)

#ax.add_feature( edgecolor='black', linewidth=0.5)

ax.add_feature(cartopy.feature.STATES,
              edgecolor='black', facecolor='none', 
              linewidth=0.8, zorder=3) 

ax.set_ylim(15, 55)
ax.set_xlim(-125, -97)
ax.coastlines(zorder=3)

gl = ax.gridlines(draw_labels=True, 
                 xlocs=[-120, -100, -80, -60], 
                 ylocs=[10, 20, 30, 40, 50, 60],
                 linestyle='--',
                 zorder=4)
gl.xlabels_top = False
gl.ylabels_right = False

# Add colorbar with descriptive labels
axins = inset_axes(ax,
                  width="5%",
                  height="100%",
                  loc='right',
                  borderpad=-4)
cbar = plt.colorbar(s, cax=axins, extend='neither',
                   label='Aridity Classification', 
                   ticks=[0.125, 0.35],
                   aspect=10, shrink=0.9)
cbar.ax.set_yticklabels(['Arid\n(0.05-0.2)', 'Semi-arid\n(0.2-0.5)'])
cbar.ax.tick_params(size=0)

plt.show()


# In[28]:


#Getting mean difference 
# getting trendy mean gpp
trendy_mean_gpp = np.ma.mean(TRENDY, axis=1)
dryflux_mean_gpp = np.ma.mean(dryflux, axis=0)
mean_diff = np.empty_like(trendy_mean_gpp)
mean_diff_ratio = np.empty_like(trendy_mean_gpp)
#getting trendy - dryflux mean
for nmod in np.arange(np.shape(trendy_mean_gpp)[0]):
    mean_diff[nmod] = trendy_mean_gpp[nmod]-masking(dryflux_mean_gpp, trendy_mean_gpp[nmod])
    mean_diff_ratio[nmod] = mean_diff[nmod]/masking(dryflux_mean_gpp, trendy_mean_gpp[nmod])


# In[29]:


print(np.ma.max(trendy_mean_gpp))


# In[30]:


#getting standard deviation
dry_sd =np.ma.std(dryflux, axis=0)
trendy_sd = np.ma.std(TRENDY, axis=1)


# In[31]:


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


# In[32]:


r2 = r*r


# In[33]:


# Define the order to plot the GPP sources
gpp_source_order = ['CABLE POP', 'CLASSIC', 'CLM5', 'IBIS', 'ISBA CTRIP', 'JSBACH', 'JULES', 'ORCHIDEE', 'VISIT', 'YIBs', 'LPJ GUESS', 'LPJwsl', 'LPX Bern', 'OCN', 'SDGVM']
valid_source = ['CLASSIC', 'CLM5','JULES', 'ORCHIDEE',  'JSBACH',  'ISBA CTRIP','CABLE POP','YIBs', 'IBIS', 'VISIT','SDGVM', 'LPJ GUESS', 'LPX Bern', 'OCN','LPJwsl']


# Create a mapping from GPP source to its index
gpp_source_mapping = {source: i for i, source in enumerate(gpp_source)}


# In[34]:


#plotting slope
fig = plt.figure(figsize=(50, 45))
ncols = 5
nrows = 3
spec = fig.add_gridspec(ncols=ncols, nrows=nrows, wspace=0.05, hspace=0.0)

# Font settings
plt.rc('axes', titlesize=50)  
plt.rc('legend', fontsize=50)  
plt.rc('font', size=40)  
plt.rc('axes', labelsize=40)  
plt.rc('xtick', labelsize=40)  
plt.rc('ytick', labelsize=40)  

# Colormap
cmap = plt.cm.terrain_r
norm = colors.Normalize(vmin=-1, vmax=3)

# Loop through subplots
for i in range(nrows):
    for j in range(ncols):
        index = i * ncols + j
        if index < len(gpp_source_order):
            ax = fig.add_subplot(spec[i, j], projection=ccrs.PlateCarree())
            
            # Map features
            ax.coastlines()
            ax.add_feature(cartopy.feature.STATES, edgecolor='black', linewidth=2)
            ax.set_extent([-125, -97, 15, 55], crs=ccrs.PlateCarree())  # Set bounds
            
            # Gridlines 
            gl = ax.gridlines(
                draw_labels=True,
                xlocs=[-120, -110, -100, -90],
                ylocs=[15, 25, 35, 45, 55],
                linestyle='--',
                alpha=0.5
            )
            
            # Hide labels where not needed
            gl.top_labels = False
            gl.right_labels = False
            if j != 0:  # Hide y-labels except for leftmost column
                gl.left_labels = False
            if i != nrows - 1:  # Hide x-labels except for bottom row
                gl.bottom_labels = False
            
            # Plot data
            gpp_source_index = gpp_source_mapping.get(gpp_source_order[index])
            s = ax.pcolormesh(lon, lat, np.flip(slope[gpp_source_index], 0), cmap=cmap, norm=norm)
            ax.set_title(gpp_source_order[index], fontsize=50)
            
            # Add subplot label (a), (b), etc.
            #ax.text(-124, 55.5, f'({chr(97 + index)})', fontsize=50) #, weight='bold')  # a, b, c, ...
        else:
            ax.set_axis_off()  # Hide unused subplots

# Add colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
#cbar = fig.colorbar(s, cax=cbar_ax, label='GPP Slope', extend='both')
cbar = fig.colorbar(s, cax=cbar_ax, cmap=cmap, norm=norm, label='Slope of the linear regression between model and DryFlux annual GPP', aspect=10, shrink=0.9, extend='both')
cbar.ax.tick_params(labelsize=40)

plt.tight_layout()
plt.show()

# Save the plot
outdir = '/Users/rubayapervin/OneDrive - Indiana University/Research Projects/Evaluation_TRENDY_GPP/Figures/'
plt.savefig(outdir + 'SWUS_slope.png')
plt.show()


# In[35]:


fig = plt.figure(figsize=(50, 40))
ncols = 6  
nrows = 3
spec = fig.add_gridspec(ncols=ncols, nrows=nrows, wspace=0.05, hspace=0.0)

plt.rc('axes', titlesize=50)
plt.rc('legend', fontsize=40)

# Colormap setup
cmap = plt.cm.PuBu
norm = colors.Normalize(vmin=0, vmax=0.25)

# Initialize label generator
labels = label_generator(case='lowercase letter', start='(', end=')')

# Plot TRENDY models (15 total)
for idx, model in enumerate(gpp_source_order[:15]):  
    row = idx // 5  
    col = idx % 5  
    
    ax = fig.add_subplot(spec[row, col], projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cartopy.feature.STATES, edgecolor='black', linewidth=2)
    ax.set_extent([-125, -97, 15, 55], crs=ccrs.PlateCarree())
    
    # Gridlines configuration
    gl = ax.gridlines(
        draw_labels=True,
        xlocs=[-120, -110, -100, -90],
        ylocs=[15, 25, 35, 45, 55],
        linestyle='--'
    )
    gl.top_labels = False
    gl.right_labels = False
    if col != 0: 
        gl.left_labels = False
    if row != 2:  
        gl.bottom_labels = False
    
    # Plot data
    gpp_idx = gpp_source_mapping[model]
    s = ax.pcolormesh(lon, lat, np.flip(trendy_sd[gpp_idx], 0), cmap=cmap, norm=norm)
    #ax.text(-124, 55.5, f'{next(labels)}', fontsize=50)
    ax.set_title(model, fontsize=50)

# Plot Dryflux in the last position (row 2, column 5)
ax = fig.add_subplot(spec[2, 5], projection=ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cartopy.feature.STATES, edgecolor='black', linewidth=2)
ax.set_extent([-125, -97, 15, 55], crs=ccrs.PlateCarree())

gl = ax.gridlines(
    draw_labels=True,
    xlocs=[-120, -110, -100, -90],
    ylocs=[15, 25, 35, 45, 55],
    linestyle='--'
)
gl.top_labels = False
gl.right_labels = False
gl.left_labels = False  

s = ax.pcolormesh(lon, lat, np.flip(get_dryland(scl_AI, dry_sd), 0), cmap=cmap, norm=norm)
#ax.text(-124, 55.5, f'{next(labels)}', fontsize=50)
ax.set_title('Dryflux', fontsize=50)

# Colorbar
cbar_ax = fig.add_axes([0.85, 0.4, 0.01, 0.45])
fig.colorbar(s, cax=cbar_ax, label='GPP Standard Deviation', extend='both')
fig.colorbar(s, cax=cbar_ax, cmap=cmap, norm=norm, label=r'$SD(\mathrm{annual\ GPP,\ 2001-2016})$', aspect=10, shrink=0.9, extend='max')


plt.tight_layout()
# Save the plot
outdir = '/Users/rubayapervin/OneDrive - Indiana University/Research Projects/Evaluation_TRENDY_GPP/Figures/'
plt.savefig(outdir + 'SWUS_sd.png')
plt.show()
plt.show()


# In[36]:


fig = plt.figure(figsize=(50, 40))
ncols = 6  
nrows = 3
spec = fig.add_gridspec(ncols=ncols, nrows=nrows, wspace=0.05, hspace=0.0)

plt.rc('axes', titlesize=50)
plt.rc('legend', fontsize=40)

# Colormap setup
cmap = plt.cm.viridis
norm = colors.Normalize(vmin=0, vmax=1.5)

# Initialize label generator
labels = label_generator(case='lowercase letter', start='(', end=')')

# Plot TRENDY models (15 total)
for idx, model in enumerate(gpp_source_order[:15]):  
    row = idx // 5 
    col = idx % 5   
    ax = fig.add_subplot(spec[row, col], projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cartopy.feature.STATES, edgecolor='black', linewidth=2)
    ax.set_extent([-125, -97, 15, 55], crs=ccrs.PlateCarree())
    
    # Gridlines configuration
    gl = ax.gridlines(
        draw_labels=True,
        xlocs=[-120, -110, -100, -90],
        ylocs=[15, 25, 35, 45, 55],
        linestyle='--'
    )
    gl.top_labels = False
    gl.right_labels = False
    if col != 0:  
        gl.left_labels = False
    if row != 2: 
        gl.bottom_labels = False
    
    # Plot data
    gpp_idx = gpp_source_mapping[model]
    s = plt.pcolormesh(lon, lat, np.flip(trendy_mean_gpp[gpp_idx], 0), cmap=cmap, norm=norm)
    plt.ylabel('g C m^-2')    
    ax.text(-124, 55.5, f'{next(labels)}', fontsize=50)
    ax.set_title(model, fontsize=50)

# Plot Dryflux in the last position (row 2, column 5)
ax = fig.add_subplot(spec[2, 5], projection=ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cartopy.feature.STATES, edgecolor='black', linewidth=2)
ax.set_extent([-125, -97, 15, 55], crs=ccrs.PlateCarree())

gl = ax.gridlines(
    draw_labels=True,
    xlocs=[-120, -110, -100, -90],
    ylocs=[15, 25, 35, 45, 55],
    linestyle='--'
)
gl.top_labels = False
gl.right_labels = False
gl.left_labels = False  # No y-label for Dryflux

# Plot data
s = plt.pcolormesh(lon, lat, np.flip(get_dryland(scl_AI, dryflux_mean_gpp), 0), cmap=cmap, norm=norm)
plt.ylabel('g C m^-2')
ax.text(-124, 55.5, f'{next(labels)}', fontsize=50)
ax.set_title('Dryflux', fontsize=50)

# Add colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
fig.colorbar(s, cax=cbar_ax, cmap = cmap, norm = norm, label = 'Mean GPP ($KgCm^-2y^-1$)', aspect=10, shrink=0.9, extend = 'max')

plt.tight_layout()

# Save the plot
outdir = '/Users/rubayapervin/OneDrive - Indiana University/Research Projects/Evaluation_TRENDY_GPP/Figures/'
plt.savefig(outdir + 'SWUS_gpp.png')
plt.show()


# In[ ]:




