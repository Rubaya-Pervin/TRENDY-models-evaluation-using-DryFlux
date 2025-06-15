#!/usr/bin/env python
# coding: utf-8

# In[1]:


import netCDF4 as nc
import random, glob, math, cartopy, re
from osgeo import gdal
import datetime, calendar, string
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


#getting dryland data
def get_dryland(AI,lat,lon,data):
    m1 = np.ma.masked_where(AI>0.5, data)
    m2 = np.ma.masked_where(AI<0.05, m1 )
    #masking out cold dryland
    m3 = np.ma.masked_where(lat<-55, m2 )
    dryland = np.ma.masked_where(lat>55, m3)
    return dryland


# In[9]:


#data files
dryland_files = glob.glob('/Users/rubayapervin/OneDrive - Indiana University/Research Projects/Evaluation_TRENDY_GPP/Data/Dryland/Global-AI_ET0_v3_annual/*.tif')


# In[11]:


#getting dryland Aridity index 
AI_file = gdal.Open(dryland_files[0])
data = AI_file.ReadAsArray()
#getting the extent of 1st file in the CHM file list 
min_x, max_y, max_x, min_y = get_extent(dryland_files[0])
gt = AI_file.GetGeoTransform()
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


# In[12]:


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


# In[13]:


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


mask_list = [ 'CLASSIC', 'CLM5','IBIS','ISBA-CTRIP', 'JSBACH', 'JULES', 'LPJ GUESS', 'LPX Bern', 'OCN', 'ORCHIDEE', 'SDGVM', 'VISIT','YIBs']
gpp_source = ['CABLE POP', 'CLASSIC', 'CLM5', 'IBIS', 'ISBA CTRIP', 'JSBACH', 'JULES', 'LPJ GUESS', 'LPJwsl', 'LPX Bern', 'OCN', 'ORCHIDEE', 'SDGVM', 'VISIT', 'YIBs']


# In[17]:


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


# In[18]:


# Flip the mask for TRENDY data orientation
common_mask_flipped = np.flip(common_mask, axis=0)

# Apply the common mask to all models and years
for idx in range(len(gpp_source)):  # Loop over all models
    for nyear in range(TRENDY.shape[1]):  # Loop over all years
        TRENDY[idx, nyear, :, :] = np.ma.masked_where(
            common_mask_flipped, 
            TRENDY[idx, nyear, :, :]  
        )


# In[19]:


# applying mask to keep common pixels of dryflux, trendy, and AI data
dryflux = masking(Dryflux, np.ma.mean(TRENDY,axis=0))
for nmod in np.arange(np.shape(TRENDY)[0]):
    TRENDY[nmod,:,:,:] = masking(TRENDY[nmod,:,:,:], dryflux)
dryflux_mean_gpp = np.ma.mean(dryflux, axis=0)
AI = masking(scl_AI, dryflux_mean_gpp)


# In[20]:


print(dryflux.shape)


# In[21]:


#Getting dryland (arid and semi-arid regions)
dryland = get_dryland(AI,lat,lon,AI)

#get north american drynad 
#get AI data in north America
AI_N = North_America (lat,lon,scl_AI)
AI = get_dryland(scl_AI,lat,lon, AI_N)

# getting dryland dryflux gpp
for nyear in np.arange(np.shape(dryflux)[0]):
    dryflux[nyear] = get_dryland(AI,lat,lon,dryflux[nyear])
# getting dryland trendy mean gpp
for nmod in np.arange(np.shape(TRENDY)[0]):
    for nyear in np.arange(np.shape(TRENDY)[1]):
        TRENDY[nmod,nyear,:,:] = get_dryland(AI,lat,lon,TRENDY[nmod,nyear,:,:])
print(TRENDY.shape)


# In[22]:


region_mask = np.zeros((360, 720))
region_mask = np.ma.masked_values (region_mask,0)
for nlat in np.arange(np.shape(scl_AI)[0]):
    for nlon in np.arange(np.shape(scl_AI)[1]):
        # US_Maxico
        if lon[nlat, nlon] >= -125 and lon[nlat, nlon] <= -93 and lat[nlat, nlon] >= -55 and lat[nlat, nlon] <= -15:
            region_mask[nlat,nlon] = 1
        #south america
        elif lon[nlat, nlon] >= -75 and lon[nlat, nlon] <= -35 and lat[nlat, nlon] >= 3 and lat[nlat, nlon] <= 55:
            region_mask[nlat,nlon] = 2  
        #Australia
        elif lon[nlat, nlon] >= 112 and lon[nlat, nlon] <= 154 and lat[nlat, nlon] >= 10 and lat[nlat, nlon] <= 44:
            region_mask[nlat,nlon] = 6  
        #Sub_Sahara
        elif lon[nlat, nlon] >= -18 and lon[nlat, nlon] <= 51 and lat[nlat, nlon] >= -22 and lat[nlat, nlon] <= 9:
            region_mask[nlat,nlon] = 3
        #S_Africa
        elif lon[nlat, nlon] >= 11 and lon[nlat, nlon] <= 51 and lat[nlat, nlon] >= 9 and lat[nlat, nlon] <= 35:
            region_mask[nlat,nlon] = 4  
        #Euro_Asia
        elif lon[nlat, nlon] >= 25 and lon[nlat, nlon] <= 128 and lat[nlat, nlon] >= -55 and lat[nlat, nlon] <= -22:
            region_mask[nlat,nlon] = 5  


# In[23]:


mask = masking(dryland, region_mask)
region_mask = masking(region_mask, mask)


# In[24]:


arid_zone =np.empty_like(dryland)
if dryland[nlat,nlon]<=0.2 and dryland[nlat,nlon]>=0.05:
    arid_zone[nlat,nlon]=1
elif dryland[nlat,nlon]>0.2 and dryland[nlat,nlon]<=0.5:
    arid_zone[nlat,nlon]=2


# In[25]:


#Getting and plotting yearly trends for each regions
def get_plot_YS(data):
    gppsource = data.Source.unique()
    for x in gppsource:
        model_data = data[data['Source']==x]
        yr = model_data['Year']
        #site_data_YS = site_data.groupby('year').sum().reindex(years)
        # saving as a CSV file
        #df.to_csv('us_src_MM.csv')
        #data['year'] = years
        # time series plot for multiple columns
        sns.lineplot(x=yr, y="GPP", data=data, label = "GPP")
        #sb.lineplot(x="year", y="pred_GPP", data=site_data_YS, label = "pred_GPP")

        # set label
        plt.ylabel("GPP")
        plt.legend()
        plt.show()
    return()    


# In[26]:


all_source =gpp_source.insert(19,'DryFlux')

       
#gpp_source[-5] ='ORCHIDEEv2'
print(gpp_source, all_source)


# In[27]:


all_source =gpp_source.insert(20,'TRENDY-mean')

       
#gpp_source[-5] ='ORCHIDEEv2'
print(gpp_source, all_source)


# In[28]:


#getting each regional dryflux GPP data 
def get_region_dryflux_GPP(region_name,region_number,lat,lon, gpp_source , dryland, years, gpp_data):
    df = pd.DataFrame(columns=['Regions','Source', 'Year', 'GPP','Aridity','Standard Deviation'])    
    regions = []
    sources = []
    nyears = []
    gpp = []
    aridity = []
    for nyear in np.arange(np.shape(gpp_data)[0]):
        for nlat in np.arange(np.shape(gpp_data)[1]):
            for nlon in np.arange(np.shape(gpp_data)[2]):
                if region_mask[nlat, nlon] == region_number and dryland[nlat,nlon]<=0.2 and dryland[nlat,nlon]>=0.05:
                    regions.append(region_name) 
                    sources.append('DryFlux') 
                    nyears.append(years[nyear]) 
                    gpp.append(gpp_data[nyear,nlat,nlon])
                    aridity.append('Arid')
                elif region_mask[nlat, nlon] == region_number and dryland[nlat,nlon]>0.2 and dryland[nlat,nlon]<=0.5:
                    regions.append(region_name) 
                    sources.append('DryFlux') 
                    nyears.append(years[nyear]) 
                    gpp.append(gpp_data[nyear,nlat,nlon])               
                    aridity.append('Semi-Arid')      
    df['Regions'] = regions
    df['Source'] = sources
    df['Year'] = nyears
    df['Aridity'] = aridity
    df['GPP'] = gpp
    
    df['GPP']=df['GPP'].astype('float')

    return df


# In[29]:


#getting each regional GPP data 
def get_region_GPP(region_name,region_number,lat,lon, gpp_source , dryland, years, gpp_data):
    df = pd.DataFrame(columns=['Regions','Source', 'Year', 'GPP','Aridity'])    
    regions = []
    sources = []
    nyears = []
    gpp = []
    aridity = []
    for nmod in np.arange(np.shape(gpp_data)[0]):
        for nyear in np.arange(np.shape(gpp_data)[1]):
            for nlat in np.arange(np.shape(gpp_data)[2]):
                for nlon in np.arange(np.shape(gpp_data)[3]):
                    if region_mask[nlat, nlon] == region_number and dryland[nlat,nlon]<=0.2 and dryland[nlat,nlon]>=0.05:
                        regions.append(region_name) 
                        sources.append(gpp_source[nmod]) 
                        nyears.append(years[nyear]) 
                        gpp.append(gpp_data[nmod,nyear,nlat,nlon])
                        aridity.append('Arid')
                    elif region_mask[nlat, nlon] == region_number and dryland[nlat,nlon]>0.2 and dryland[nlat,nlon]<=0.5:
                        regions.append(region_name) 
                        sources.append(gpp_source[nmod]) 
                        nyears.append(years[nyear]) 
                        gpp.append(gpp_data[nmod,nyear,nlat,nlon])               
                        aridity.append('Semi-Arid')      
    df['Regions'] = regions
    df['Source'] = sources
    df['Year'] = nyears
    df['Aridity'] = aridity
    df['GPP'] = gpp
    
    df['GPP']=df['GPP'].astype('float')

    return df


# In[30]:


region_names = ['North America']


# In[31]:


#getting regional dryflux GPP data 

regional_dryflux_GPP = []
for nreg, reg in enumerate(region_names):
    region_number = nreg+1
    region_name = reg
    print(region_number,region_name)
    df = get_region_dryflux_GPP (region_name,region_number,lat,lon, gpp_source , dryland, years, dryflux)
    regional_dryflux_GPP.append(df) 


# In[32]:


#getting regional GPP data for each models
regional_GPP = []
for nreg, reg in enumerate(region_names):
    region_number = nreg+1
    region_name = reg
    print(region_number,region_name)
    df = get_region_GPP (region_name,region_number,lat,lon, gpp_source , dryland, years, TRENDY)
    regional_GPP.append(df) 


# In[33]:


all_GPP = pd.concat([ regional_GPP[0],  regional_dryflux_GPP[0]])


# In[34]:


print(regional_GPP[0])


# In[35]:


y = regional_GPP[0].Year.unique()
TRENDy_mean = regional_GPP[0].groupby('Year')['GPP'].mean().reindex(y).reset_index()
TRENDy_mean['Regions']="North America"
TRENDy_mean['Source']="TRENDY-mean"
TRENDy_mean['Year']=y
#del TRENDy_mean[]
TRENDy_mean = TRENDy_mean[['Regions', 'Source', 'Year', 'GPP']]
print(TRENDy_mean)


# In[36]:


everything = pd.concat([all_GPP, TRENDy_mean])
print(everything)


# In[37]:


gpp_source_order = ['CABLE POP', 'CLASSIC', 'CLM5', 'IBIS', 'ISBA CTRIP', 'JSBACH', 'JULES', 'ORCHIDEE', 'VISIT', 'YIBs', 'LPJ GUESS', 'LPJwsl', 'LPX Bern', 'OCN', 'SDGVM', 'DryFlux', 'TRENDY-mean']


# In[38]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define colors and styles
gpp_source_order = [
    'CABLE POP', 'CLASSIC', 'CLM5', 'IBIS', 'ISBA CTRIP', 
    'JSBACH', 'JULES', 'ORCHIDEE', 'VISIT', 'YIBs', 
    'LPJ GUESS', 'LPJwsl', 'LPX Bern', 'OCN', 'SDGVM', 'DryFlux', 'TRENDY-mean'
]

colors = sns.color_palette("copper_r", 10) + sns.color_palette("Blues", 5)
model_colors = dict(zip(gpp_source_order[:-2], colors))
model_colors['DryFlux'] = 'black'
model_colors['TRENDY-mean'] = 'grey'

styles = ['-.', '--', ':', '--', '-.', ':', '-.', '--', ':', '--', 
          '-', '--', ':', '-.', '-', 'solid', 'solid']

def plot_gpp_time_series(ax, data, sources, ylabel, y_col):
    """Plot GPP time series for each source"""
    for i, source in enumerate(sources):
        # Filter data for current source
        model_data = data[data['Source'] == source]
        
        if model_data.empty:
            print(f"Warning: No data found for {source}. Skipping.")
            continue
            
        # Get unique years and calculate mean (only for numeric columns)
        years = model_data['Year'].unique()
        numeric_data = model_data.select_dtypes(include=[np.number])
        means = numeric_data.groupby('Year').mean().reindex(years)
        
        # For normalized IAV
        if y_col == 'norm.GPP IAV':
            means[y_col] = means['GPP'] - means['GPP'].mean()
        
        # Plot
        sns.lineplot(
            x="Year", 
            y=y_col if y_col in means.columns else 'GPP', 
            data=means, 
            linestyle=styles[i],
            color=model_colors[source],
            linewidth=4 if source in ['DryFlux', 'TRENDY-mean'] else 2,
            ax=ax,
            label=source  # Add label for legend
        )

    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_xlabel("Year", fontsize=16)

# Create figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

# Plot data
plot_gpp_time_series(ax1, everything, gpp_source_order, "GPP ($KgCyr^{-1}$)", "GPP")
plot_gpp_time_series(ax2, everything, gpp_source_order, "GPP anomaly ($KgCyr^{-1}$)", "norm.GPP IAV")

# Add titles
ax1.set_title("a)", fontsize=20, loc='left', pad=1, y=0.9)
ax2.set_title("b)", fontsize=20, loc='left', pad=1, y=0.9)

# Adjust tick labels
ax1.tick_params(axis='both', labelsize=14)
ax2.tick_params(axis='both', labelsize=14)

# Create unified legend
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.2, 0.54), fontsize=16, frameon=False)

# Remove individual plot legends
ax1.get_legend().remove()
ax2.get_legend().remove()

# Adjust layout
fig.tight_layout()
plt.show()


# In[39]:


# Extract the data for 'DryFlux' and 'TRENDY mean'
dryflux_data = everything[everything['Source'] == 'DryFlux']
trendy_mean_data = everything[everything['Source'] == 'TRENDY-mean']

# Check if data exists for both models
if dryflux_data.empty or trendy_mean_data.empty:
    print("Warning: One or both of the models 'DryFlux' or 'TRENDY mean' have no data.")
else:
    # Group by year and calculate the mean for each model
    dryflux_data_y = dryflux_data.groupby('Year')['GPP'].mean()
    trendy_mean_data_y = trendy_mean_data.groupby('Year')['GPP'].mean()

    # Ensure both datasets are aligned by year
    merged_data = pd.merge(dryflux_data_y, trendy_mean_data_y, left_index=True, right_index=True, suffixes=('_DryFlux', '_TRENDY_mean'))

    # Calculate Pearson correlation coefficient (r) between 'GPP' of DryFlux and TRENDY mean
    correlation = merged_data['GPP_DryFlux'].corr(merged_data['GPP_TRENDY_mean'])
    print(f"Pearson correlation (r) between DryFlux and TRENDY mean: {correlation:.4f}")

