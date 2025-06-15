#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, random, glob, math, cartopy
from osgeo import gdal
import datetime, calendar, string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import colors
import cartopy.crs as ccrs
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import statsmodels.formula.api as smf
from matplotlib.patches import Rectangle
import skimage.transform as st
from scipy.ndimage import zoom
import netCDF4 as nc


# In[2]:


#data files
dryland_files = glob.glob('C:/Users/rpervin/OneDrive - Indiana University/Research Projects/Evaluation_TRENDY_GPP/Data/Dryland/Global-AI_ET0_v3_annual/*.tif')
#DryFlux_file = glob.glob('C:/Users/rpervin/OneDrive - Indiana University/Research Projects/Evaluation_TRENDY_GPP/Data/DryFlux/*.tif')
TRENDY_gpp_files = glob.glob('/Users/rubayapervin/OneDrive - Indiana University/Research Projects/Evaluation_TRENDY_GPP/Data/Trendy/V11/yearlyMean/*1979-2021_yearlymean_XYT.nc')
for x in TRENDY_gpp_files: 
    print(x)


# In[3]:


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


# In[4]:


#getting dryland data
def get_dryland(AI,data):
    m1 = np.ma.masked_where(AI>0.5, data)
    dryland = np.ma.masked_where(AI<0.05, m1 )
    #masking out cold dryland
    #m3 = np.ma.masked_where(lat<-55, m2 )
    #dryland = np.ma.masked_where(lat>55, m3)
    return dryland


# In[5]:


def resize(dt):
    for n,x in enumerate(dt):
        dt[n] = st.resize(x, (360, 720), order=0, preserve_range=True, anti_aliasing=False)
    return dt


# In[6]:


def get_lon_lat(num_rows,num_cols,resolution,top_left_y,top_left_x):           # Define the dimensions of the raster dataset
    # Generate the row and column indices
    row = np.arange(num_rows)
    col = np.arange(num_cols)

    # Calculate latitude and longitude for each pixel
    lat = top_left_y - resolution * row
    lon = top_left_x + resolution * col
    return lon,lat


# In[7]:


def get_extent(fn):
    "'returns min_x, max_y, max_x, min_y'"
    ds = gdal.Open(fn)
    gt = ds.GetGeoTransform()
    return (gt[0], gt[3], gt[0]+gt[1]*ds.RasterXSize, gt[3]+gt[5]*ds.RasterYSize)


# In[8]:


#getting dryland Aridity index at 0.5 degree spatial resolution
scl_AI = np.load('/Users/rubayapervin/OneDrive - Indiana University/Research Projects/Evaluation_TRENDY_GPP/Data/Dryland/AIpoint5.npy')


# In[9]:


#getting dryland Aridity index at 0.05 degree spatial resolution
AI05 = np.load('/Users/rubayapervin/OneDrive - Indiana University/Research Projects/Evaluation_TRENDY_GPP/Data/Dryland/AIp05.npy')


# In[10]:


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


# In[11]:


outdir = '/Users/rubayapervin/OneDrive - Indiana University/Research Projects/Evaluation_TRENDY_GPP/Figures/SWUS/PFTmaps/'


# In[12]:


#model_list = ['CABLE POP', 'CLASSIC', 'CLM5', 'JSBACH', 'JULES', 'LPJ GUESS', 'LPJwsl','LPX Bern', 'OCN', 'ORCHIDEE', 'SDGVM', 'VISIT','YIBs']
#print(len(model_list))
#crop mask and gpp data list 
mask_list = [ 'CLASSIC', 'CLM5','IBIS', 'ISBA CTRIP', 'JSBACH', 'JULES', 'LPJ GUESS', 'LPX Bern', 'OCN', 'ORCHIDEE', 'SDGVM', 'VISIT','YIBs']
model_list = ['CABLE POP', 'CLASSIC', 'CLM5', 'IBIS', 'ISBA CTRIP', 'JSBACH', 'JULES', 'LPJ GUESS', 'LPJwsl', 'LPX Bern', 'OCN', 'ORCHIDEE', 'SDGVM', 'VISIT', 'YIBs']


# In[13]:


crop_mask_files =glob.glob('/Users/rubayapervin/OneDrive - Indiana University/Research Projects/Evaluation_TRENDY_GPP/Data/Trendy/V11/PFT/TRENDY-v11_cropMasked/mod_masks/*')
for x  in crop_mask_files: 
    print(x)


# In[14]:


# Initialize list to store masks
all_masks = []

# Loop through each model in gpp_source
for mod in model_list:
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
    common_mask_p5 = np.any(np.stack(all_masks, axis=0), axis=0)
else:
    common_mask_p5 = np.zeros((360, 720), dtype=bool)  # No masking if no masks exist


# Initialize list to store masks
all_masks = []

# Loop through each model in gpp_source
for mod in model_list:
    if mod in mask_list:  # Check if the model has a mask
        mask_index = mask_list.index(mod)
        mask = np.load(crop_mask_files[mask_index])['mask']
        
        # Resize mask to (360, 720) if needed (nearest-neighbor interpolation)
        if mask.shape != (3600, 7200):
            zoom_factors = (3600 / mask.shape[0], 7200 / mask.shape[1])
            mask = zoom(mask, zoom_factors, order=0)  # order=0 for nearest-neighbor
        
        all_masks.append(mask)

# Create common_mask (True where ANY model's mask is True)
if all_masks:
    common_mask_p05 = np.any(np.stack(all_masks, axis=0), axis=0)
else:
    common_mask_p05 = np.zeros((360, 720), dtype=bool)  # No masking if no masks exist


# In[15]:


#functions to get regions 
def North_America (lat,lon,dt):
    m1 = np.ma.masked_where(lon<-125, dt )
    m2 = np.ma.masked_where(lon>-93, m1 )
    m3= np.ma.masked_where(lat<-53, m2 )
    m4= np.ma.masked_where(lat>-15, m3 )
    return m4

# function to create masked array
def masking(ar1, ar2):
    res_mask = np.ma.getmask(ar2)
    masked = np.ma.masked_array(ar1, mask=res_mask)
    return masked


# In[16]:


# Flip the mask for TRENDY data orientation
common_mask_flipped = np.flip(common_mask_p5, axis=0)
AI = np.ma.masked_where(common_mask_flipped, scl_AI)
#get AI data in north America
AI_N = North_America (lat,lon,AI)
AI = get_dryland(scl_AI, AI_N)


#getting lat lon
 # Global dataset parameters
global_num_rows = 3600
global_num_cols = 7200
global_resolution = 0.05
global_top_left_y = 90
global_top_left_x = -180

lon_global, lat_global = get_lon_lat(global_num_rows, global_num_cols, global_resolution, global_top_left_y, global_top_left_x)
lon_global, lat_global = np.meshgrid(lon_global, lat_global)
lat_global = np.flip(lat_global, axis=0)
# Flip the mask for RAP data orientation
rap_mask_flipped = np.flip(common_mask_p05, axis=0)
AI_rap = np.ma.masked_where(rap_mask_flipped, AI05)
#get AI data in north America
AI_N_rap = North_America (lat_global,lon_global,AI_rap)
AI_rap = get_dryland(AI05, AI_N_rap)


# In[17]:


#gpp_source_order = ['YIBs', 'CABLE POP','CLM5','JULES', 'CLASSIC', 'ORCHIDEE',  'JSBACH',  'VISIT', 'VISIT NIES', 'DLEM', 'IBIS', 'ISBA CTRIP','ISAM','','OCN','SDGVM' ,'LPJ GUESS',  'LPJwsl', 'LPX Bern']
model_list = ['CABLE POP', 'CLASSIC', 'CLM5','IBIS','ISBA CTRIP', 'JSBACH', 'JULES', 'LPJ GUESS', 'LPJwsl','LPX Bern', 'OCN', 'ORCHIDEE', 'SDGVM', 'VISIT','YIBs']
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


# In[18]:


# Define file paths and years of interest
years = np.arange(2001, 2017)  
base_path = "/Users/rubayapervin/OneDrive - Indiana University/Research Projects/Evaluation_TRENDY_GPP/Data/FractionalCover/RAP/np_files"
cover_types = {
    "woody": [],
    "nonwoody": [],
    "allveg": [],
    "barren": []
}

# Load, process, and sum cover types for each year
for year in years:
    # Load relevant bands for each year
    annual_grass = np.load(os.path.join(base_path, f"{year}Band 1_annual forb and grass.npy"))
    bare_ground = np.load(os.path.join(base_path, f"{year}Band 2_bare ground.npy"))
    litter = np.load(os.path.join(base_path, f"{year}Band 3_litter.npy"))
    perennial_grass = np.load(os.path.join(base_path, f"{year}Band 4_perennial forb and grass.npy"))
    shrub = np.load(os.path.join(base_path, f"{year}Band 5_shrub.npy"))
    tree = np.load(os.path.join(base_path, f"{year}Band 6_tree.npy"))

    # Calculate cover types
    woody = tree + shrub
    nonwoody = annual_grass + perennial_grass
    allveg = woody + nonwoody
    barren = bare_ground

    # Append the arrays to the corresponding lists for each cover type
    cover_types["woody"].append(woody)
    cover_types["nonwoody"].append(nonwoody)
    cover_types["allveg"].append(allveg)
    cover_types["barren"].append(barren)

# Convert lists to 3D arrays with shape (year, lat, lon)
for cover_type in cover_types:
    cover_types[cover_type] = np.stack(cover_types[cover_type], axis=0)
    
# Calculate the mean across years for each cover type
mean_woody = np.ma.mean(cover_types["woody"], axis=0)
mean_nonwoody = np.ma.mean(cover_types["nonwoody"], axis=0)
mean_allveg = np.ma.mean(cover_types["allveg"], axis=0)
mean_barren = np.ma.mean(cover_types["barren"], axis=0)

# Calculate the standard deviation across years for each cover type
std_woody = np.ma.std(cover_types["woody"], axis=0)
std_nonwoody = np.ma.std(cover_types["nonwoody"], axis=0)
std_allveg = np.ma.std(cover_types["allveg"], axis=0)
std_barren = np.ma.std(cover_types["barren"], axis=0)


# In[19]:


lon,lat = get_lon_lat(360,720,0.5,90,-180)


# In[20]:


#Creating lat lon mash grid
# Define the dimensions of the raster dataset
num_rows = 497  # Example number of rows
num_cols = 1158  # Example number of columns
resolution = 0.05
top_left_y = 49.40
top_left_x = -124.75

# Generate the row and column indices
row = np.arange(num_rows)
col = np.arange(num_cols)

# Calculate latitude and longitude for each pixel
latitude = top_left_y - resolution * row
longitude = top_left_x + resolution * col

# Create the mesh grid
lon_mesh, lat_mesh = np.meshgrid(longitude, latitude)

# Display the shape of the mesh grid
print("Latitude mesh grid shape:", lat_mesh.shape)
print("Longitude mesh grid shape:", lon_mesh.shape)


# In[21]:


def get_lon_lat(num_rows,num_cols,resolution,top_left_y,top_left_x):           # Define the dimensions of the raster dataset
    # Generate the row and column indices
    row = np.arange(num_rows)
    col = np.arange(num_cols)

    # Calculate latitude and longitude for each pixel
    lat = top_left_y - resolution * row
    lon = top_left_x + resolution * col
    return lon,lat


# In[22]:


lon,lat = get_lon_lat(3600,7200,0.05,90,-180)


# In[23]:


def mask_rap(rap_data, ai_data, lat_mesh, lon_mesh):
    num_rows, num_cols = rap_data.shape
    masked_rap_data= np.full_like(rap_data, np.nan, dtype=float)

    for i in range(num_rows):
        for j in range(num_cols):
            # Find the closest matching point in the global data
            global_row = np.abs(lat_global - lat_mesh[i, j]).argmin()
            global_col = np.abs(lon_global - lon_mesh[i, j]).argmin()

            # Get the corresponding AI value from the global dataset
            ai_value = ai_data[global_row, global_col]

            # Check if AI value is in the specified range
            if 0.05 <= ai_value <= 0.5:
                masked_rap_data[i, j] = rap_data[i, j]  # Keep the std_allveg value
            else:
                masked_rap_data[i, j] = np.nan  # Mask the value

    return masked_rap_data


# In[24]:


# Global dataset parameters
global_num_rows = 3600
global_num_cols = 7200
global_resolution = 0.05
global_top_left_y = 90
global_top_left_x = -180

lon_global, lat_global = get_lon_lat(global_num_rows, global_num_cols, global_resolution, global_top_left_y, global_top_left_x)

# Regional dataset parameters
num_rows = 497
num_cols = 1158
resolution = 0.05
top_left_y = 49.40
top_left_x = -124.75

longitude, latitude = get_lon_lat(num_rows, num_cols, resolution, top_left_y, top_left_x)
lon_mesh, lat_mesh = np.meshgrid(longitude, latitude)


# In[25]:


dry_mean_woody =  mask_rap(mean_woody, AI_rap, lat_mesh, lon_mesh)
dry_mean_nonwoody =  mask_rap(mean_nonwoody, AI_rap, lat_mesh, lon_mesh)
dry_mean_allveg =  mask_rap(mean_allveg, AI_rap, lat_mesh, lon_mesh)
dry_mean_barren =  mask_rap(mean_barren, AI_rap, lat_mesh, lon_mesh)

# Calculate the standard deviation across years for each cover type
dry_std_woody =  mask_rap(std_woody, AI_rap, lat_mesh, lon_mesh)
dry_std_nonwoody =  mask_rap(std_nonwoody, AI_rap, lat_mesh, lon_mesh)
dry_std_allveg =  mask_rap(std_allveg, AI_rap, lat_mesh, lon_mesh)
dry_std_barren =  mask_rap(std_barren, AI_rap, lat_mesh, lon_mesh)


# In[26]:


rap = [dry_std_allveg, dry_std_woody, dry_std_nonwoody, dry_std_barren]


# In[27]:


print(rap[1].shape)
plt.imshow(rap[1])


# In[28]:


modis_fn = glob.glob('C:/Users/rpervin/OneDrive - Indiana University/Research Projects/Evaluation_TRENDY_GPP/Data/FractionalCover/MODIS_VCF_Global_2001_2016/*')
for f in modis_fn: 
    print(f)


# In[29]:


# Define file paths and years of interest
years = np.arange(2001, 2017)  # Modify this range according to your dataset
base_path = "/Users/rubayapervin/OneDrive - Indiana University/Research Projects/Evaluation_TRENDY_GPP/Data/FractionalCover/MODIS_VCF_Global_2001_2016"
cover_types = {
    "woody": [],
    "nonwoody": [],
    "allveg": [],
    "barren": []
}

# Load, process, and sum cover types for each year
for year in years:
    # Load relevant bands for each year
    ds = gdal.Open(os.path.join(base_path, f"MODIS_VCF_Global_{year}.tif"))

    # Calculate cover types
    woody = ds.GetRasterBand(1).ReadAsArray()
    nonwoody =  ds.GetRasterBand(2).ReadAsArray()
    barren = ds.GetRasterBand(3).ReadAsArray()
    allveg = woody + nonwoody

    # Append the arrays to the corresponding lists for each cover type
    cover_types["woody"].append(woody)
    cover_types["nonwoody"].append(nonwoody)
    cover_types["allveg"].append(allveg)
    cover_types["barren"].append(barren)

# Convert lists to 3D arrays with shape (year, lat, lon)
for cover_type in cover_types:
    cover_types[cover_type] = np.stack(cover_types[cover_type], axis=0)

# Calculate the mean across years for each cover type
mean_woody = np.ma.mean(cover_types["woody"], axis=0)
mean_nonwoody = np.ma.mean(cover_types["nonwoody"], axis=0)
mean_allveg = np.ma.mean(cover_types["allveg"], axis=0)
mean_barren = np.ma.mean(cover_types["barren"], axis=0)

# Calculate the standard deviation across years for each cover type
std_woody = np.ma.std(cover_types["woody"], axis=0)
std_nonwoody = np.ma.std(cover_types["nonwoody"], axis=0)
std_allveg = np.ma.std(cover_types["allveg"], axis=0)
std_barren = np.ma.std(cover_types["barren"], axis=0)


# In[30]:


modis = [std_allveg, std_woody, std_nonwoody,std_barren]


# In[31]:


mod_std_allveg = get_dryland(AI,std_allveg)
#plt.imshow(mod_std_allveg)
#plt.colorbar()


# In[32]:


lon,lat = get_lon_lat(360,720,0.5,90,-180)


# In[33]:


#for single pft type
def pft_sd_plot(pft_files, pft_type, gpp_source_order, model_list, modis, reference_data, CB_label = 'PFT Standard Deviation'):
    fig = plt.figure(figsize=(60, 15))
    model_order = ['JULES', 'LPJ GUESS', 'LPJwsl', 'LPX Bern']

    ncols = 7
    nrows = 1
    spec = fig.add_gridspec(ncols=ncols, nrows=nrows, wspace=0.05, hspace=0.0)
    plt.rc('axes', titlesize=50)  # fontsize of the title
    plt.rc('legend', fontsize=40)  # fontsize of the legend
    plt.rc('font', size=40)  # controls default text size
    plt.rc('axes', labelsize=40)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=40)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=40)  # fontsize of the y tick labels
    labels = label_generator(case='lowercase letter', start='(', end=')')

    #cmap = colors.ListedColormap(sns.color_palette('YlGn', n_colors=10))
    #norm = colors.BoundaryNorm(bounds, cmap.N)
    cmap = plt.cm.Wistia
    norm = colors.Normalize(vmin=0, vmax=0.20)
    for i in range(nrows):
        for j in range(ncols):
            idx = i * ncols + j
            if idx<5: 
                ax = fig.add_subplot(spec[i, j], projection=ccrs.PlateCarree())
                ax.coastlines()
                ax.add_feature(cartopy.feature.STATES, edgecolor='black', linewidth=1)
                ax.set_global()
                ax.set_ylim(15, 55)
                ax.set_xlim(-125, -97)
                gl = ax.gridlines(draw_labels=True, xlocs=[-120, -110, -100, -90], ylocs=[15, 25, 35, 45, 55],
                                  linestyle='--')
                gl.xlabels_top = False
                gl.ylabels_right = False
                if j != 0:
                    gl.ylabels_left = False
                if i != nrows - 1:
                    gl.xlabels_bottom = False
                #ax.text(-125, 56, '%s' % (next(labels)), fontsize=50)
                ax.set_title(model_order[idx], fontsize=50)
                model_index = model_list.index(model_order[idx])
                print(pft_files[model_index])
                data = np.load(pft_files[model_index])
                try:
                    pft = data[pft_type]
                    pfty = st.resize(pft[0], (360, 720), order=0, preserve_range=True, anti_aliasing=False) if pft is not None else None
                    pfty = np.expand_dims(pfty, axis=0) if pfty is not None else None # Adding pft dimension at position 0 
                    for n,x in enumerate(pft[1:]):
                        pfy = st.resize(x, (360, 720), order=0, preserve_range=True, anti_aliasing=False) if x is not None else None
                        pfy= np.expand_dims(pfy, axis=0) if pfy is not None else None # Adding pft dimension at position 0
                        pfty = np.concatenate((pfty,pfy),axis=0) if pfty is not None else None

                    #print(pfty.shape)
                    sd = np.ma.std(pfty,axis = 0) if pfty is not None else None
                    s = ax.pcolormesh(lon, lat, get_dryland(scl_AI, np.flip(sd, 0)), cmap=cmap,
                                              norm=norm)
                    ax.set_ylabel('g C m^-2')
                                
                except KeyError:
                    pass
            if idx ==5:
                ax = fig.add_subplot(spec[i, j], projection=ccrs.PlateCarree())
                ax.coastlines()
                # Add state boundaries to plot
                ax.add_feature(cartopy.feature.STATES, edgecolor='black', linewidth=1)
                ax.set_global()
                ax.set_ylim(15,55)
                ax.set_xlim(-125,-97)
                gl = ax.gridlines(draw_labels=True, xlocs=[-120, -110, -100, -90], ylocs=[15, 25, 35, 45, 55], linestyle='--')
                gl.xlabels_top = False
                gl.ylabels_right = False
                if j != 0:
                    gl.ylabels_left = False
                if i != nrows-1:
                    gl.xlabels_bottom = False
                # Plot data
                s = ax.pcolormesh(lon, lat, get_dryland(scl_AI, modis)/100, cmap=cmap,
                                              norm=norm)
                plt.ylabel('g C m^-2')
                #ax.text(-125, 56, '%s' % (next(labels)), fontsize=50)
                ax.set_title('MODIS VCF', fontsize=50)
                
            if idx == 6:
                ax = fig.add_subplot(spec[i, j], projection=ccrs.PlateCarree())
                ax.coastlines()
                # Add state boundaries to plot
                ax.add_feature(cartopy.feature.STATES, edgecolor='black', linewidth=1)
                ax.set_global()
                ax.set_ylim(15,55)
                ax.set_xlim(-125,-97)
                gl = ax.gridlines(draw_labels=True, xlocs=[-120, -110, -100, -90], ylocs=[15, 25, 35, 45, 55], linestyle='--')
                gl.xlabels_top = False
                gl.ylabels_right = False
                if j != 0:
                    gl.ylabels_left = False
                if i != nrows-1:
                    gl.xlabels_bottom = False
                # Plot data
                # Define the dimensions of the raster dataset
                num_rows = 497  # Example number of rows
                num_cols = 1158  # Example number of columns
                resolution = 0.05
                top_left_y = 49.40
                top_left_x = -124.75

                # Generate the row and column indices
                row = np.arange(num_rows)
                col = np.arange(num_cols)

                # Calculate latitude and longitude for each pixel
                latitude = top_left_y - resolution * row
                longitude = top_left_x + resolution * col
                s = plt.pcolormesh(longitude, latitude, np.ma.masked_where(reference_data==0,reference_data)/100, cmap=cmap, norm=norm)
                plt.ylabel('g C m^-2')
                #ax.text(-125, 56, '%s' % (next(labels)), fontsize=50)
                ax.set_title('RAP', fontsize=50)
            
    # Add colorbar
    #fig.add_axes([left, bottom, width, height])
    cbar_ax = fig.add_axes([0.92, 0.22, 0.01, 0.55])
    cb = fig.colorbar(s, cax=cbar_ax, cmap=cmap, norm=norm, label=CB_label)
    cb.ax.tick_params(labelsize=30)  # Adjust colorbar tick font size

    plt.tight_layout()
    plt.show()


# In[34]:


#for multiple pft types
def pft_sd_plots(pft_files, gpp_source_order, model_list, modis, reference_data, pft_types = None, pft_type_titles = None, CB_label='PFT Standard Deviation'):
    fig = plt.figure(figsize=(60, 45))  # Adjust the figure size to accommodate 3 rows of subplots
    model_order = ['JULES', 'LPJ GUESS', 'LPJwsl', 'LPX Bern']

    ncols = 7
    nrows = 3  # Now 3 rows to accommodate allveg, woody_allveg, nonwoody_allveg

    spec = fig.add_gridspec(ncols=ncols, nrows=nrows, wspace=0.05, hspace=0.0)
    plt.rc('axes', titlesize=50)  # fontsize of the title
    plt.rc('legend', fontsize=40)  # fontsize of the legend
    plt.rc('font', size=40)  # controls default text size
    plt.rc('axes', labelsize=40)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=40)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=40)  # fontsize of the y tick labels

    cmap = plt.cm.Wistia
    norm = colors.Normalize(vmin=0, vmax=0.20)

    # Plot for each of the three PFT types
    if pft_types is None:
        pft_types = ['allveg', 'woody_allveg', 'nonwoody_allveg']
    else: pft_types = pft_types 
    if pft_type_titles is None:
        pft_type_titles = ['All Vegetation', 'Woody Vegetation', 'Non-Woody Vegetation']
    else: pft_type_titles = pft_type_titles
    
    # Loop through rows and columns, creating subplots
    for row_idx, pft_type in enumerate(pft_types):
        for col_idx in range(ncols):
            idx = row_idx * ncols + col_idx  # Compute the subplot index

            if idx < len(model_order):  # Ensure we only plot models in the correct range
                ax = fig.add_subplot(spec[row_idx, col_idx], projection=ccrs.PlateCarree())
                ax.coastlines()
                ax.add_feature(cartopy.feature.STATES, edgecolor='black', linewidth=1)
                ax.set_global()
                ax.set_ylim(15, 55)
                ax.set_xlim(-125, -97)

                gl = ax.gridlines(draw_labels=True, xlocs=[-120, -110, -100, -90], ylocs=[15, 25, 35, 45, 55], linestyle='--')
                gl.xlabels_top = False
                gl.ylabels_right = False
                if col_idx != 0:
                    gl.ylabels_left = False
                if row_idx != nrows - 1:
                    gl.xlabels_bottom = False

                # Title for each model in the row
                ax.set_title(f'{model_order[col_idx]}', fontsize=50)

                # Load the data for the current model and PFT type
                model_index = model_list.index(model_order[col_idx])
                print(f"Loading data for {model_order[col_idx]} and PFT {pft_type}")
                data = np.load(pft_files[model_index])
                
                try:
                    pft = data[pft_type]
                    # Resize and process PFT data
                    pfty = st.resize(pft[0], (360, 720), order=0, preserve_range=True, anti_aliasing=False) if pft is not None else None
                    pfty = np.expand_dims(pfty, axis=0) if pfty is not None else None  # Adding pft dimension at position 0
                    for n, x in enumerate(pft[1:]):
                        pfy = st.resize(x, (360, 720), order=0, preserve_range=True, anti_aliasing=False) if x is not None else None
                        pfy = np.expand_dims(pfy, axis=0) if pfy is not None else None  # Adding pft dimension at position 0
                        pfty = np.concatenate((pfty, pfy), axis=0) if pfty is not None else None
                    
                    # Compute standard deviation of PFT cover
                    sd = np.ma.std(pfty, axis=0) if pfty is not None else None
                    s = ax.pcolormesh(lon, lat, get_dryland(scl_AI, np.flip(sd, 0)), cmap=cmap, norm=norm)
                    ax.set_ylabel('g C m^-2')

                except KeyError:
                    pass

            # Plot for MODIS (fixed in column idx 5)
            if idx == 5:
                ax = fig.add_subplot(spec[row_idx, col_idx], projection=ccrs.PlateCarree())
                ax.coastlines()
                ax.add_feature(cartopy.feature.STATES, edgecolor='black', linewidth=1)
                ax.set_global()
                ax.set_ylim(15, 55)
                ax.set_xlim(-125, -97)
                gl = ax.gridlines(draw_labels=True, xlocs=[-120, -110, -100, -90], ylocs=[15, 25, 35, 45, 55], linestyle='--')
                gl.xlabels_top = False
                gl.ylabels_right = False
                if col_idx != 0:
                    gl.ylabels_left = False
                if row_idx != nrows - 1:
                    gl.xlabels_bottom = False

                # Plot MODIS data
                s = ax.pcolormesh(lon, lat, get_dryland(scl_AI, modis) / 100, cmap=cmap, norm=norm)
                ax.set_ylabel('g C m^-2')
                ax.set_title('MODIS VCF', fontsize=50)

            # Plot for RAP (fixed in column idx 6)
            if idx == 6:
                ax = fig.add_subplot(spec[row_idx, col_idx], projection=ccrs.PlateCarree())
                ax.coastlines()
                ax.add_feature(cartopy.feature.STATES, edgecolor='black', linewidth=1)
                ax.set_global()
                ax.set_ylim(15, 55)
                ax.set_xlim(-125, -97)
                gl = ax.gridlines(draw_labels=True, xlocs=[-120, -110, -100, -90], ylocs=[15, 25, 35, 45, 55], linestyle='--')
                gl.xlabels_top = False
                gl.ylabels_right = False
                if col_idx != 0:
                    gl.ylabels_left = False
                if row_idx != nrows - 1:
                    gl.xlabels_bottom = False

                # Plot RAP data
                num_rows = 497
                num_cols = 1158
                resolution = 0.05
                top_left_y = 49.40
                top_left_x = -124.75

                row = np.arange(num_rows)
                col = np.arange(num_cols)

                latitude = top_left_y - resolution * row
                longitude = top_left_x + resolution * col
                s = ax.pcolormesh(longitude, latitude, np.ma.masked_where(reference_data == 0, reference_data) / 100, cmap=cmap, norm=norm)
                ax.set_ylabel('g C m^-2')
                ax.set_title('RAP', fontsize=50)

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.22, 0.01, 0.55])
    cb = fig.colorbar(s, cax=cbar_ax, cmap=cmap, norm=norm, label=CB_label)
    cb.ax.tick_params(labelsize=30)  # Adjust colorbar tick font size

    plt.tight_layout()
    plt.show()


# In[35]:


lon, lat = np.meshgrid(lon, lat)


# In[37]:


def get_df(pft_type, pft_data, metric_name, metric_data, AI, gpp_source, model_list):
    df = pd.DataFrame(columns=['Models', pft_type, metric_name, 'Aridity'])   
    models = []
    pft_dt = []
    metric_dt = []
    aridity = []

    for idx, model in enumerate(model_list):
        gpp_source_index = gpp_source.index(model)

        # Check if pft_data[idx] or metric_data[gpp_source_index] is None
        if pft_data[idx] is None:
            print(f"Skipping pft_data[{idx}] because it is None")
            continue
        if metric_data[gpp_source_index] is None:
            print(f"Skipping metric_data[{gpp_source_index}] because it is None")
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
                if not test_mask[nlat, nlon] and 0.2 < AI[nlat, nlon] <= 0.5:
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


# In[38]:


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


# In[39]:


allveg_sd = get_sd_pft('allveg', sorted_pft_files, AI)
barren_sd = get_sd_pft('barren', sorted_pft_files, AI)


# In[41]:


def pft_sd_plots(pft_files, gpp_source_order, model_list, modis, reference_data, pft_types=None, pft_type_titles=None, CB_label='PFT Standard Deviation'):
    fig = plt.figure(figsize=(60, 45))  # Adjust the figure size to accommodate 3 rows of subplots
    model_order = ['JULES', 'LPJ GUESS', 'LPJwsl', 'LPX Bern']

    ncols = 6
    nrows = 4  # Now 3 rows to accommodate allveg, woody_allveg, nonwoody_allveg

    spec = fig.add_gridspec(ncols=ncols, nrows=nrows, wspace=0.05, hspace=0.0)
    plt.rc('axes', titlesize=50)  # fontsize of the title
    plt.rc('legend', fontsize=40)  # fontsize of the legend
    plt.rc('font', size=40)  # controls default text size
    plt.rc('axes', labelsize=40)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=40)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=40)  # fontsize of the y tick labels

    cmap = plt.cm.Wistia
    norm = colors.Normalize(vmin=0, vmax=0.20)

    # Define the PFT types and their titles if not provided
    if pft_types is None:
        pft_types = ['allveg', 'woody_allveg', 'nonwoody_allveg']
    if pft_type_titles is None:
        pft_type_titles = ['All Vegetation', 'Woody Vegetation', 'Non-Woody Vegetation']

    # Loop through rows (for different PFT types) and columns (for different models)
    for row_idx, (pft_type, pft_title) in enumerate(zip(pft_types, pft_type_titles)):
        for col_idx in range(ncols):
            idx = row_idx * ncols + col_idx  # Compute the subplot index

            if idx < len(model_order):  # Ensure we only plot models in the correct range
                ax = fig.add_subplot(spec[row_idx, col_idx], projection=ccrs.PlateCarree())
                ax.coastlines()
                ax.add_feature(cartopy.feature.STATES, edgecolor='black', linewidth=1)
                ax.set_global()
                ax.set_ylim(15, 55)
                ax.set_xlim(-125, -97)

                gl = ax.gridlines(draw_labels=True, xlocs=[-120, -110, -100, -90], ylocs=[15, 25, 35, 45, 55], linestyle='--')
                gl.xlabels_top = False
                gl.ylabels_right = False
                if col_idx != 0:
                    gl.ylabels_left = False
                if row_idx != nrows - 1:
                    gl.xlabels_bottom = False

                # Title for each model in the row
                ax.set_title(f'{model_order[col_idx]}', fontsize=50)

                # Load the data for the current model and PFT type
                model_index = model_list.index(model_order[col_idx])
                print(f"Loading data for {model_order[col_idx]} and PFT {pft_type}")
                data = np.load(pft_files[model_index])
                
                try:
                    pft = data[pft_type]
                    # Resize and process PFT data
                    pfty = st.resize(pft[0], (360, 720), order=0, preserve_range=True, anti_aliasing=False) if pft is not None else None
                    pfty = np.expand_dims(pfty, axis=0) if pfty is not None else None  # Adding pft dimension at position 0
                    for n, x in enumerate(pft[1:]):
                        pfy = st.resize(x, (360, 720), order=0, preserve_range=True, anti_aliasing=False) if x is not None else None
                        pfy = np.expand_dims(pfy, axis=0) if pfy is not None else None  # Adding pft dimension at position 0
                        pfty = np.concatenate((pfty, pfy), axis=0) if pfty is not None else None
                    
                    # Compute standard deviation of PFT cover
                    sd = np.ma.std(pfty, axis=0) if pfty is not None else None
                    s = ax.pcolormesh(lon, lat, get_dryland(scl_AI, np.flip(sd, 0)), cmap=cmap, norm=norm)
                    ax.set_ylabel('g C m^-2')

                except KeyError:
                    pass

            # Plot for MODIS (fixed in column idx 5)
            if idx == 4:
                ax = fig.add_subplot(spec[row_idx, col_idx], projection=ccrs.PlateCarree())
                ax.coastlines()
                ax.add_feature(cartopy.feature.STATES, edgecolor='black', linewidth=1)
                ax.set_global()
                ax.set_ylim(15, 55)
                ax.set_xlim(-125, -97)
                gl = ax.gridlines(draw_labels=True, xlocs=[-120, -110, -100, -90], ylocs=[15, 25, 35, 45, 55], linestyle='--')
                gl.xlabels_top = False
                gl.ylabels_right = False
                if col_idx != 0:
                    gl.ylabels_left = False
                if row_idx != nrows - 1:
                    gl.xlabels_bottom = False

                # Plot MODIS data
                s = ax.pcolormesh(lon, lat, get_dryland(scl_AI, modis) / 100, cmap=cmap, norm=norm)
                ax.set_ylabel('g C m^-2')
                ax.set_title('MODIS VCF', fontsize=50)

            # Plot for RAP (fixed in column idx 6)
            if idx == 5:
                ax = fig.add_subplot(spec[row_idx, col_idx], projection=ccrs.PlateCarree())
                ax.coastlines()
                ax.add_feature(cartopy.feature.STATES, edgecolor='black', linewidth=1)
                ax.set_global()
                ax.set_ylim(15, 55)
                ax.set_xlim(-125, -97)
                gl = ax.gridlines(draw_labels=True, xlocs=[-120, -110, -100, -90], ylocs=[15, 25, 35, 45, 55], linestyle='--')
                gl.xlabels_top = False
                gl.ylabels_right = False
                if col_idx != 0:
                    gl.ylabels_left = False
                if row_idx != nrows - 1:
                    gl.xlabels_bottom = False

                # Plot RAP data
                num_rows = 497
                num_cols = 1158
                resolution = 0.05
                top_left_y = 49.40
                top_left_x = -124.75

                row = np.arange(num_rows)
                col = np.arange(num_cols)

                latitude = top_left_y - resolution * row
                longitude = top_left_x + resolution * col
                s = ax.pcolormesh(longitude, latitude, np.ma.masked_where(reference_data == 0, reference_data) / 100, cmap=cmap, norm=norm)
                ax.set_ylabel('g C m^-2')
                ax.set_title('RAP', fontsize=50)

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.22, 0.01, 0.55])
    cb = fig.colorbar(s, cax=cbar_ax, cmap=cmap, norm=norm, label=CB_label)
    cb.ax.tick_params(labelsize=30)  # Adjust colorbar tick font size

    plt.tight_layout()
    plt.show()


# In[42]:


def pft_sd_combined_plot(pft_files, pft_types, pft_type_titles, model_list, modis, reference_data, CB_label='PFT Standard Deviation'):
    # Create a figure with 3 rows (one for each plot) and 7 columns (for the models + MODIS + RAP)
    fig, axs = plt.subplots(nrows=3, ncols=6, figsize=(55, 40),
                            subplot_kw={'projection': ccrs.PlateCarree()})
    model_order = [ 'JULES', 'LPJ GUESS', 'LPJwsl', 'LPX Bern']
    cmap = plt.cm.YlGnBu 
    norm = colors.Normalize(vmin=0, vmax=0.20)

    # Adjust the layout and remove excess whitespace
    plt.subplots_adjust(wspace=0.05, hspace=0.0)
    plt.rc('axes', titlesize=60)
    plt.rc('legend', fontsize=50)
    plt.rc('font', size=50)
    plt.rc('axes', labelsize=50)
    plt.rc('xtick', labelsize=50)
    plt.rc('ytick', labelsize=50)

    for i, pft_type in enumerate(pft_types):
        for j, model in enumerate(model_order):
            idx = i * len(model_order) + j
            ax = axs[i, j]  # Get the corresponding subplot for each PFT and model

            ax.coastlines()
            ax.add_feature(cartopy.feature.STATES, edgecolor='black', linewidth=1)
            ax.set_global()
            ax.set_ylim(15, 55)
            ax.set_xlim(-125, -97)
            gl = ax.gridlines(draw_labels=True, xlocs=[-125, -115, -105, -95], ylocs=[15, 25, 35, 45, 55], linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            if j != 0:
                gl.left_labels = False
            if i != 2:  # x labels on the bottom row
                gl.bottom_labels = False

            # Load the data for the current model and PFT type
            model_index = model_list.index(model)
            data = np.load(pft_files[model_index])
            try:
                pft = data[pft_type]
                pfty = st.resize(pft[0], (360, 720), order=0, preserve_range=True, anti_aliasing=False) if pft is not None else None
                pfty = np.expand_dims(pfty, axis=0) if pfty is not None else None
                for n, x in enumerate(pft[1:]):
                    pfy = st.resize(x, (360, 720), order=0, preserve_range=True, anti_aliasing=False) if x is not None else None
                    pfy = np.expand_dims(pfy, axis=0) if pfy is not None else None
                    pfty = np.concatenate((pfty, pfy), axis=0) if pfty is not None else None

                # Calculate the standard deviation
                sd = np.ma.std(pfty, axis=0) if pfty is not None else None
                s = ax.pcolormesh(lon, lat, get_dryland(AI, np.flip(sd, 0)), cmap=cmap, norm=norm)
                ax.set_ylabel('g C m^-2')
                if i==0:
                    ax.set_title(model, fontsize=60)
            except KeyError:
                pass

        # Add MODIS and RAP plots in the last two columns for each PFT type
        for j, pft_type in enumerate(pft_types):
            ax_modis = axs[j, 4]  
            ax_rap = axs[j, 5] 

            # MODIS plot
            ax_modis.coastlines()
            ax_modis.add_feature(cartopy.feature.STATES, edgecolor='black', linewidth=1)
            ax_modis.set_global()
            ax_modis.set_ylim(15, 55)
            ax_modis.set_xlim(-125, -97)
            gl = ax_modis.gridlines(draw_labels=True, xlocs=[-125, -115, -105, -95], ylocs=[15, 25, 35, 45, 55], linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            gl.left_labels = False
            if j != 2:
                gl.bottom_labels = False

            # Select the appropriate MODIS data for the PFT type (e.g., 'woody_allveg', 'allveg', 'allgrass')
            modis_data = modis[j]  # Assuming `modis` is a dictionary with pft_type keys
            s = ax_modis.pcolormesh(lon, lat, get_dryland(AI, modis_data) / 100, cmap=cmap, norm=norm)
            if j==0:
                ax_modis.set_title('MODIS VCF', fontsize=60)

            # RAP plot
            ax_rap.coastlines()
            ax_rap.add_feature(cartopy.feature.STATES, edgecolor='black', linewidth=1)
            ax_rap.set_global()
            ax_rap.set_ylim(15, 55)
            ax_rap.set_xlim(-125, -97)
            gl = ax_rap.gridlines(draw_labels=True, xlocs=[-125, -115, -105, -95], ylocs=[15, 25, 35, 45, 55], linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            gl.left_labels = False
            if j != 2:
                gl.bottom_labels = False

            # Select the appropriate RAP data for the PFT type (e.g., 'woody_allveg', 'allveg', 'allgrass')
            rap_data = reference_data[j]  # Assuming `reference_data` is a dictionary with pft_type keys
            s = ax_rap.pcolormesh(longitude, latitude, np.ma.masked_where(rap_data == 0, rap_data) / 100, cmap=cmap, norm=norm)
            if j==0:
                ax_rap.set_title('RAP', fontsize=60)
        # Add vertical text labels for each row (left side)
        fig.text(0, 0.8 - i * 0.32, pft_type_titles[i], rotation=90, fontsize=60, va='center', ha='center')
            
    # Add a single colorbar for all plots
    cbar_ax = fig.add_axes([1, 0.17, 0.01, 0.65])  # Position for colorbar
    cb = fig.colorbar(s, cax=cbar_ax, cmap=cmap, norm=norm, extend = 'max', label=CB_label)
    cb.ax.tick_params(labelsize=40)

    plt.tight_layout()
    # Save the plot
    outdir = '/Users/rubayapervin/OneDrive - Indiana University/Research Projects/Evaluation_TRENDY_GPP/Figures/Paper/'
    plt.savefig(outdir + 'fcoverSD.png')
    plt.show()


# In[43]:


def pft_sd_combined_2plot(pft_files, pft_types, pft_type_titles, model_list, modis, reference_data, CB_label='PFT Standard Deviation'):
    # Create a figure with 3 rows (one for each plot) and 7 columns (for the models + MODIS + RAP)
    fig, axs = plt.subplots(nrows=2, ncols=6, figsize=(60, 30),
                            subplot_kw={'projection': ccrs.PlateCarree()})
    model_order = ['JULES', 'LPJ GUESS', 'LPJwsl', 'LPX Bern']
    cmap = plt.cm.YlGnBu 
    norm = colors.Normalize(vmin=0, vmax=0.20)

    # Adjust the layout and remove excess whitespace
    plt.subplots_adjust(wspace=0.05, hspace=0.0)
    plt.rc('axes', titlesize=60)
    plt.rc('legend', fontsize=50)
    plt.rc('font', size=50)
    plt.rc('axes', labelsize=50)
    plt.rc('xtick', labelsize=50)
    plt.rc('ytick', labelsize=50)

    for i, pft_type in enumerate(pft_types):
        for j, model in enumerate(model_order):
            idx = i * len(model_order) + j
            ax = axs[i, j]  # Get the corresponding subplot for each PFT and model

            ax.coastlines()
            ax.add_feature(cartopy.feature.STATES, edgecolor='black', linewidth=1)
            ax.set_global()
            ax.set_ylim(15, 55)
            ax.set_xlim(-125, -97)
            gl = ax.gridlines(draw_labels=True, xlocs=[-125, -115, -105, -95], ylocs=[15, 25, 35, 45, 55], linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            if j != 0:
                gl.left_labels = False
            if i != 1:  # x labels on the bottom row
                gl.bottom_labels = False

            # Load the data for the current model and PFT type
            model_index = model_list.index(model)
            data = np.load(pft_files[model_index])
            try:
                pft = data[pft_type]
                pfty = st.resize(pft[0], (360, 720), order=0, preserve_range=True, anti_aliasing=False) if pft is not None else None
                pfty = np.expand_dims(pfty, axis=0) if pfty is not None else None
                for n, x in enumerate(pft[1:]):
                    pfy = st.resize(x, (360, 720), order=0, preserve_range=True, anti_aliasing=False) if x is not None else None
                    pfy = np.expand_dims(pfy, axis=0) if pfy is not None else None
                    pfty = np.concatenate((pfty, pfy), axis=0) if pfty is not None else None

                # Calculate the standard deviation
                sd = np.ma.std(pfty, axis=0) if pfty is not None else None
                s = ax.pcolormesh(lon, lat, get_dryland(AI, np.flip(sd, 0)), cmap=cmap, norm=norm)
                ax.set_ylabel('g C m^-2')
                if i ==0:
                    ax.set_title(model, fontsize=60)
            except KeyError:
                pass

        # Add MODIS and RAP plots in the last two columns for each PFT type
        for j, pft_type in enumerate(pft_types):
            ax_modis = axs[j, 4]  
            ax_rap = axs[j, 5]   

            # MODIS plot
            ax_modis.coastlines()
            ax_modis.add_feature(cartopy.feature.STATES, edgecolor='black', linewidth=1)
            ax_modis.set_global()
            ax_modis.set_ylim(15, 55)
            ax_modis.set_xlim(-125, -97)
            gl = ax_modis.gridlines(draw_labels=True, xlocs=[-125, -115, -105, -95], ylocs=[15, 25, 35, 45, 55], linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            gl.left_labels = False
            if j != 1:  # x labels on the bottom row
                gl.bottom_labels = False

            # Select the appropriate MODIS data for the PFT type (e.g., 'woody_allveg', 'allveg', 'allgrass')
            modis_data = modis[2]  # Assuming `modis` is a dictionary with pft_type keys
            s = ax_modis.pcolormesh(lon, lat, get_dryland(AI, modis_data) / 100, cmap=cmap, norm=norm)
            if j == 0:
                ax_modis.set_title('MODIS VCF', fontsize=60)

            # RAP plot
            ax_rap.coastlines()
            ax_rap.add_feature(cartopy.feature.STATES, edgecolor='black', linewidth=1)
            ax_rap.set_global()
            ax_rap.set_ylim(15, 55)
            ax_rap.set_xlim(-125, -97)
            gl = ax_rap.gridlines(draw_labels=True, xlocs=[-125, -115, -105, -95], ylocs=[15, 25, 35, 45, 55], linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            gl.left_labels = False
            if j != 1:  # x labels on the bottom row
                gl.bottom_labels = False

            # Select the appropriate RAP data for the PFT type (e.g., 'woody_allveg', 'allveg', 'allgrass')
            rap_data = reference_data[2]  # Assuming `reference_data` is a dictionary with pft_type keys
            s = ax_rap.pcolormesh(longitude, latitude, np.ma.masked_where(rap_data == 0, rap_data) / 100, cmap=cmap, norm=norm)
            if j==0: 
                ax_rap.set_title('RAP', fontsize=60)
        # Add vertical text labels for each row (left side)
        fig.text(0, 0.7 - i * 0.4, pft_type_titles[i], rotation=90, fontsize=60, va='center', ha='center')
            
    # Add a single colorbar for all plots
    cbar_ax = fig.add_axes([1, 0.17, 0.01, 0.65])  # Position for colorbar
    cb = fig.colorbar(s, cax=cbar_ax, cmap=cmap, norm=norm, extend = 'max', label=CB_label)
    cb.ax.tick_params(labelsize=40)

    plt.tight_layout()
    # Save the plot
    outdir = '/Users/rubayapervin/OneDrive - Indiana University/Research Projects/Evaluation_TRENDY_GPP/Figures/Supplementary/'
    plt.savefig(outdir + 'S_fcoverSD_2.png')
    plt.show()


# In[45]:


pft_types = ['allveg', 'woody_allveg', 'allgrass']
pft_type_titles = ['All Vegetation', 'Woody', 'Grass']
pft_sd_combined_plot(sorted_pft_files, pft_types, pft_type_titles, model_list, modis, rap, CB_label='SD (PFT, 2001-2016)')


# In[ ]:





# In[ ]:





# In[46]:


pft_types = ['nonwoody_allveg', 'allgrass']
pft_type_titles = ['Non-woody', 'Grass']


# In[47]:


pft_sd_combined_2plot(sorted_pft_files, pft_types, pft_type_titles, model_list, modis, rap, CB_label='SD (PFT, 2001-2016)')


# In[48]:


def pft_mean_plot_2ref(pft_files, pft_type, model_list, modis, reference_data, CB_label='PFT Standard Deviation'):
    fig = plt.figure(figsize=(45, 55))
    model_order = ['CABLE POP', 'CLASSIC', 'CLM5', 'IBIS', 'ISBA CTRIP', 'JSBACH', 'JULES', 'ORCHIDEE', 'VISIT', 'YIBs', 'LPJ GUESS', 'LPJwsl', 'LPX Bern', 'OCN', 'SDGVM']
    ncols = 5
    nrows = 4
    spec = fig.add_gridspec(ncols=ncols, nrows=nrows, wspace=0.05, hspace=0.0)
    plt.rc('axes', titlesize=50)
    plt.rc('legend', fontsize=40)
    plt.rc('font', size=40)
    plt.rc('axes', labelsize=40)
    plt.rc('xtick', labelsize=40)
    plt.rc('ytick', labelsize=40)
    labels = label_generator(case='lowercase letter', start='(', end=')')

    cmap = plt.cm.summer_r
    norm = colors.Normalize(vmin=0, vmax=1)

    for i in range(nrows):
        for j in range(ncols):
            idx = i * ncols + j
            if idx < 15:
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
                if i == nrows - 2 and j>1:
                    gl.bottom_labels = True
                ax.text(-125, 56, '%s' % (next(labels)), fontsize=50)
                ax.set_title(model_order[idx], fontsize=50)

                # Check if model is in the model_list
                if model_order[idx] in model_list:
                    model_index = model_list.index(model_order[idx])
                    print(pft_files[model_index])
                    data = np.load(pft_files[model_index])
                    try:
                        pft = data[pft_type]
                        pfty = st.resize(pft[0], (360, 720), order=0, preserve_range=True, anti_aliasing=False) if pft is not None else None
                        pfty = np.expand_dims(pfty, axis=0) if pfty is not None else None  # Adding pft dimension at position 0 
                        for n, x in enumerate(pft[1:]):
                            pfy = st.resize(x, (360, 720), order=0, preserve_range=True, anti_aliasing=False) if x is not None else None
                            pfy = np.expand_dims(pfy, axis=0) if pfy is not None else None  # Adding pft dimension at position 0
                            pfty = np.concatenate((pfty, pfy), axis=0) if pfty is not None else None

                        mean = np.ma.mean(pfty, axis=0) if pfty is not None else None
                        s = ax.pcolormesh(lon, lat, get_dryland(AI, np.flip(mean, 0)), cmap=cmap, norm=norm)
                        ax.set_ylabel('g C m^-2')
                    except KeyError:
                        pass
                else:
                    # Turn off the subplot if model is not found
                    ax.axis('off')

            if idx == 15:
                ax = fig.add_subplot(spec[i, j], projection=ccrs.PlateCarree())
                ax.coastlines()
                ax.add_feature(cartopy.feature.STATES, edgecolor='black', linewidth=1)
                ax.set_global()
                ax.set_ylim(15, 55)
                ax.set_xlim(-125, -97)
                gl = ax.gridlines(draw_labels=True, xlocs=[-120, -110, -100, -90], ylocs=[15, 25, 35, 45, 55], linestyle='--')
                gl.top_labels = False
                gl.right_labels = False
                if j != 0:
                    gl.left_labels = False
                if i != nrows - 1:
                    gl.bottom_labels = False
                
                s = ax.pcolormesh(lon, lat, get_dryland(AI, modis) / 100, cmap=cmap, norm=norm)
                plt.ylabel('g C m^-2')
                ax.text(-125, 56, '%s' % (next(labels)), fontsize=50)
                ax.set_title('MODIS VCF', fontsize=50)

            if idx == 16:
                ax = fig.add_subplot(spec[i, j], projection=ccrs.PlateCarree())
                ax.coastlines()
                ax.add_feature(cartopy.feature.STATES, edgecolor='black', linewidth=1)
                ax.set_global()
                ax.set_ylim(15, 55)
                ax.set_xlim(-125, -97)
                gl = ax.gridlines(draw_labels=True, xlocs=[-120, -110, -100, -90], ylocs=[15, 25, 35, 45, 55], linestyle='--')
                gl.top_labels = False
                gl.right_labels = False
                if j != 0:
                    gl.left_labels = False
                if i != nrows - 1:
                    gl.bottom_labels = False
                
                # Define the dimensions of the raster dataset
                num_rows = 497
                num_cols = 1158
                resolution = 0.05
                top_left_y = 49.40
                top_left_x = -124.75

                row = np.arange(num_rows)
                col = np.arange(num_cols)

                latitude = top_left_y - resolution * row
                longitude = top_left_x + resolution * col
                s = ax.pcolormesh(longitude, latitude, np.ma.masked_where(reference_data > 100, reference_data) / 100, cmap=cmap, norm=norm)
                plt.ylabel('g C m^-2')
                ax.text(-125, 56, '%s' % (next(labels)), fontsize=50)
                ax.set_title('RAP', fontsize=50)

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.22, 0.01, 0.55])
    cb = fig.colorbar(s, cax=cbar_ax, cmap=cmap, norm=norm, label=CB_label)
    cb.ax.tick_params(labelsize=30)

    plt.tight_layout()
    plt.show()


# In[203]:


pft_mean_plot_2ref(sorted_pft_files,'allveg', model_list,mean_allveg, dry_mean_allveg, CB_label='Mean All Vegetation (Total PFT, 2001-2016)')
pft_mean_plot_2ref(sorted_pft_files,'woody_allveg', model_list,mean_woody, dry_mean_woody, CB_label='Mean Woody (PFT, 2001-2016)')
pft_mean_plot_2ref(sorted_pft_files,'nonwoody_allveg', model_list,mean_nonwoody, dry_mean_nonwoody, CB_label='Mean Non-woody (PFT, 2001-2016)')
pft_mean_plot_2ref(sorted_pft_files,'barren', model_list,mean_barren, dry_mean_barren, CB_label='Mean Barren (PFT, 2001-2016)')


# In[218]:


def pft_mean_plot_modis(pft_files, pft_type, model_list, modis, CB_label='PFT Standard Deviation'):
    fig = plt.figure(figsize=(45, 55))
    model_order = ['CABLE POP', 'CLASSIC', 'CLM5', 'IBIS', 'ISBA CTRIP', 'JSBACH', 'JULES', 'ORCHIDEE', 'VISIT', 'YIBs', 'LPJ GUESS', 'LPJwsl', 'LPX Bern', 'OCN', 'SDGVM']
    ncols = 5
    nrows = 4
    spec = fig.add_gridspec(ncols=ncols, nrows=nrows, wspace=0.05, hspace=0.0)
    plt.rc('axes', titlesize=50)
    plt.rc('legend', fontsize=40)
    plt.rc('font', size=40)
    plt.rc('axes', labelsize=40)
    plt.rc('xtick', labelsize=40)
    plt.rc('ytick', labelsize=40)
    labels = label_generator(case='lowercase letter', start='(', end=')')

    cmap = plt.cm.summer_r
    norm = colors.Normalize(vmin=0, vmax=1)

    for i in range(nrows):
        for j in range(ncols):
            idx = i * ncols + j
            if idx < 15:
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
                if i == nrows - 2 and j>0:
                    gl.bottom_labels = True
          
                ax.text(-125, 56, '%s' % (next(labels)), fontsize=50)
                ax.set_title(model_order[idx], fontsize=50)

                # Check if model is in the model_list
                if model_order[idx] in model_list:
                    model_index = model_list.index(model_order[idx])
                    print(pft_files[model_index])
                    data = np.load(pft_files[model_index])
                    try:
                        pft = data[pft_type]
                        pfty = st.resize(pft[0], (360, 720), order=0, preserve_range=True, anti_aliasing=False) if pft is not None else None
                        pfty = np.expand_dims(pfty, axis=0) if pfty is not None else None  # Adding pft dimension at position 0 
                        for n, x in enumerate(pft[1:]):
                            pfy = st.resize(x, (360, 720), order=0, preserve_range=True, anti_aliasing=False) if x is not None else None
                            pfy = np.expand_dims(pfy, axis=0) if pfy is not None else None  # Adding pft dimension at position 0
                            pfty = np.concatenate((pfty, pfy), axis=0) if pfty is not None else None

                        mean = np.ma.mean(pfty, axis=0) if pfty is not None else None
                        s = ax.pcolormesh(lon, lat, get_dryland(AI, np.flip(mean, 0)), cmap=cmap, norm=norm)
                        ax.set_ylabel('g C m^-2')
                    except KeyError:
                        pass
                else:
                    # Turn off the subplot if model is not found
                    ax.axis('off')

            if idx == 15:
                ax = fig.add_subplot(spec[i, j], projection=ccrs.PlateCarree())
                ax.coastlines()
                ax.add_feature(cartopy.feature.STATES, edgecolor='black', linewidth=1)
                ax.set_global()
                ax.set_ylim(15, 55)
                ax.set_xlim(-125, -97)
                gl = ax.gridlines(draw_labels=True, xlocs=[-120, -110, -100, -90], ylocs=[15, 25, 35, 45, 55], linestyle='--')
                gl.top_labels = False
                gl.right_labels = False
                if j != 0:
                    gl.left_labels = False
                if i != nrows - 1:
                    gl.bottom_labels = False
                
                s = ax.pcolormesh(lon, lat, get_dryland(AI, modis) / 100, cmap=cmap, norm=norm)
                plt.ylabel('g C m^-2')
                ax.text(-125, 56, '%s' % (next(labels)), fontsize=50)
                ax.set_title('MODIS VCF', fontsize=50)

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.22, 0.01, 0.55])
    cb = fig.colorbar(s, cax=cbar_ax, cmap=cmap, norm=norm, label=CB_label)
    cb.ax.tick_params(labelsize=30)

    plt.tight_layout()
    plt.show()


# In[219]:


pft_mean_plot_modis(sorted_pft_files,'nonwoody_allveg', model_list,mean_nonwoody, CB_label='Mean Non-woody (PFT, 2001-2016)')


# In[229]:


def pft_mean_plot_rap(pft_files, pft_type, model_list, reference_data, CB_label='PFT Standard Deviation'):
    fig = plt.figure(figsize=(45, 55))
    model_order = ['CABLE POP', 'CLASSIC', 'CLM5', 'IBIS', 'ISBA CTRIP', 'JSBACH', 'JULES', 'ORCHIDEE', 'VISIT', 'YIBs', 'LPJ GUESS', 'LPJwsl', 'LPX Bern', 'OCN', 'SDGVM']
    ncols = 5
    nrows = 4
    spec = fig.add_gridspec(ncols=ncols, nrows=nrows, wspace=0.05, hspace=0.0)
    plt.rc('axes', titlesize=50)
    plt.rc('legend', fontsize=40)
    plt.rc('font', size=40)
    plt.rc('axes', labelsize=40)
    plt.rc('xtick', labelsize=40)
    plt.rc('ytick', labelsize=40)
    labels = label_generator(case='lowercase letter', start='(', end=')')

    cmap = plt.cm.summer_r
    norm = colors.Normalize(vmin=0, vmax=1)

    for i in range(nrows):
        for j in range(ncols):
            idx = i * ncols + j
            if idx < 15:
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
                if i == nrows - 2 and j>0:
                    gl.bottom_labels = True
                    
                ax.text(-125, 56, '%s' % (next(labels)), fontsize=50)
                ax.set_title(model_order[idx], fontsize=50)

                # Check if model is in the model_list
                if model_order[idx] in model_list:
                    model_index = model_list.index(model_order[idx])
                    print(pft_files[model_index])
                    data = np.load(pft_files[model_index])
                    try:
                        pft = data[pft_type]
                        pfty = st.resize(pft[0], (360, 720), order=0, preserve_range=True, anti_aliasing=False) if pft is not None else None
                        pfty = np.expand_dims(pfty, axis=0) if pfty is not None else None  # Adding pft dimension at position 0 
                        for n, x in enumerate(pft[1:]):
                            pfy = st.resize(x, (360, 720), order=0, preserve_range=True, anti_aliasing=False) if x is not None else None
                            pfy = np.expand_dims(pfy, axis=0) if pfy is not None else None  # Adding pft dimension at position 0
                            pfty = np.concatenate((pfty, pfy), axis=0) if pfty is not None else None

                        mean = np.ma.mean(pfty, axis=0) if pfty is not None else None
                        s = ax.pcolormesh(lon, lat, get_dryland(AI, np.flip(mean, 0)), cmap=cmap, norm=norm)
                        ax.set_ylabel('g C m^-2')
                    except KeyError:
                        pass
                else:
                    # Turn off the subplot if model is not found
                    ax.axis('off')

            if idx == 15:
                ax = fig.add_subplot(spec[i, j], projection=ccrs.PlateCarree())
                ax.coastlines()
                ax.add_feature(cartopy.feature.STATES, edgecolor='black', linewidth=1)
                ax.set_global()
                ax.set_ylim(15, 55)
                ax.set_xlim(-125, -97)
                gl = ax.gridlines(draw_labels=True, xlocs=[-120, -110, -100, -90], ylocs=[15, 25, 35, 45, 55], linestyle='--')
                gl.top_labels = False
                gl.right_labels = False
                if j != 0:
                    gl.left_labels = False
                if i != nrows - 1:
                    gl.bottom_labels = False
                
                # Define the dimensions of the raster dataset
                num_rows = 497
                num_cols = 1158
                resolution = 0.05
                top_left_y = 49.40
                top_left_x = -124.75

                row = np.arange(num_rows)
                col = np.arange(num_cols)

                latitude = top_left_y - resolution * row
                longitude = top_left_x + resolution * col
                s = ax.pcolormesh(longitude, latitude, np.ma.masked_where(reference_data > 100, reference_data) / 100, cmap=cmap, norm=norm)
                plt.ylabel('g C m^-2')
                ax.text(-125, 56, '%s' % (next(labels)), fontsize=50)
                ax.set_title('RAP', fontsize=50)


    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.22, 0.01, 0.55])
    cb = fig.colorbar(s, cax=cbar_ax, cmap=cmap, norm=norm, label=CB_label)
    cb.ax.tick_params(labelsize=30)

    plt.tight_layout()
    plt.show()


# In[230]:


pft_mean_plot_rap(sorted_pft_files,'allgrass', model_list, dry_mean_nonwoody, CB_label='Mean Grass (PFT, 2001-2016)')


# In[245]:


def pft_mean_plot(pft_files, pft_type, model_list, CB_label='PFT mean'):
    fig = plt.figure(figsize=(45, 45))
    model_order = ['CABLE POP', 'CLASSIC', 'CLM5', 'IBIS', 'ISBA CTRIP', 'JSBACH', 'JULES', 'ORCHIDEE', 'VISIT', 'YIBs', 'LPJ GUESS', 'LPJwsl', 'LPX Bern', 'OCN', 'SDGVM']
    ncols = 5
    nrows = 3
    spec = fig.add_gridspec(ncols=ncols, nrows=nrows, wspace=0.05, hspace=0.0)
    plt.rc('axes', titlesize=50)
    plt.rc('legend', fontsize=40)
    plt.rc('font', size=40)
    plt.rc('axes', labelsize=40)
    plt.rc('xtick', labelsize=40)
    plt.rc('ytick', labelsize=40)
    labels = label_generator(case='lowercase letter', start='(', end=')')

    cmap = plt.cm.summer_r
    norm = colors.Normalize(vmin=0, vmax=1)

    for i in range(nrows):
        for j in range(ncols):
            idx = i * ncols + j
            if idx < 15:
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
                ax.text(-125, 56, '%s' % (next(labels)), fontsize=50)
                ax.set_title(model_order[idx], fontsize=50)

                # Check if model is in the model_list
                if model_order[idx] in model_list:
                    model_index = model_list.index(model_order[idx])
                    print(pft_files[model_index])
                    data = np.load(pft_files[model_index], allow_pickle=True)  # allow_pickle=True to handle object arrays
                    
                    try:
                        pft = data.get(pft_type, None)
                        # Check if pft is a valid, iterable array with dimensions
                        if pft is None or not hasattr(pft, '__iter__') or pft.ndim == 0:
                            continue  # Skip this iteration if pft_type is not valid

                        # Resize and handle only non-None layers in pft
                        pfty = np.array([
                            st.resize(layer, (360, 720), order=0, preserve_range=True, anti_aliasing=False)
                            for layer in pft if layer is not None
                        ])
                        
                        if pfty.size > 0:
                            mean = np.ma.mean(pfty, axis=0)
                            s = ax.pcolormesh(lon, lat, get_dryland(AI, np.flip(mean, 0)), cmap=cmap, norm=norm)
                            ax.set_ylabel('g C m^-2')

                    except KeyError:
                        pass
                else:
                    # Turn off the subplot if model is not found
                    ax.axis('off')

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.22, 0.01, 0.55])
    cb = fig.colorbar(s, cax=cbar_ax, cmap=cmap, norm=norm, label=CB_label)
    cb.ax.tick_params(labelsize=30)

    plt.tight_layout()
    plt.show()


# In[246]:


pft_mean_plot(sorted_pft_files,'allgrass' ,model_list, CB_label='Mean Fractional Cover (Grass, 2001-2016)')
pft_mean_plot(sorted_pft_files,'c3grass' ,model_list, CB_label='Mean C3 Grass (PFT, 2001-2016)')
pft_mean_plot(sorted_pft_files,'c4grass' ,model_list, CB_label='Mean C4 Grass (PFT, 2001-2016)')
pft_mean_plot(sorted_pft_files,'allcrop' ,model_list, CB_label='Mean Crop (PFT, 2001-2016)')
pft_mean_plot(sorted_pft_files,'c3crop', model_list, CB_label='Mean Fractional Cover (C3 Crop, 2001-2016)')
pft_mean_plot(sorted_pft_files,'c4crop', model_list, CB_label='Mean Fractional Cover (C4 Crop, 2001-2016)')


# In[ ]:





# In[247]:


def pft_sd_plot_2ref(pft_files, pft_type, model_list, modis, reference_data, CB_label = 'PFT Standard Deviation'):
    fig = plt.figure(figsize=(45, 55))
    model_order = ['CABLE POP', 'CLASSIC', 'CLM5', 'IBIS', 'ISBA CTRIP', 'JSBACH', 'JULES', 'ORCHIDEE', 'VISIT', 'YIBs', 'LPJ GUESS', 'LPJwsl', 'LPX Bern', 'OCN', 'SDGVM']
    ncols = 5
    nrows = 4
    spec = fig.add_gridspec(ncols=ncols, nrows=nrows, wspace=0.05, hspace=0.0)
    plt.rc('axes', titlesize=50)  # fontsize of the title
    plt.rc('legend', fontsize=40)  # fontsize of the legend
    plt.rc('font', size=40)  # controls default text size
    plt.rc('axes', labelsize=40)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=40)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=40)  # fontsize of the y tick labels
    labels = label_generator(case='lowercase letter', start='(', end=')')

    #cmap = colors.ListedColormap(sns.color_palette('YlGn', n_colors=10))
    #norm = colors.BoundaryNorm(bounds, cmap.N)
    cmap = plt.cm.YlGnBu 
    norm = colors.Normalize(vmin=0, vmax=0.20)
    for i in range(nrows):
        for j in range(ncols):
            idx = i * ncols + j
            if idx<15: 
                ax = fig.add_subplot(spec[i, j], projection=ccrs.PlateCarree())
                ax.coastlines()
                ax.add_feature(cartopy.feature.STATES, edgecolor='black', linewidth=2)
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
                if i == nrows - 2 and j>1:
                    gl.bottom_labels = True
                    
                ax.text(-125, 56, '%s' % (next(labels)), fontsize=50)
                ax.set_title(model_order[idx], fontsize=50)
                
                # Check if model is in the model_list
                if model_order[idx] in model_list:
                    model_index = model_list.index(model_order[idx])
                    print(pft_files[model_index])
                    data = np.load(pft_files[model_index], allow_pickle=True)  # allow_pickle=True to handle object arrays
                    
                    try:
                        pft = data.get(pft_type, None)
                        # Check if pft is a valid, iterable array with dimensions
                        if pft is None or not hasattr(pft, '__iter__') or pft.ndim == 0:
                            continue  # Skip this iteration if pft_type is not valid

                        # Resize and handle only non-None layers in pft
                        pfty = np.array([
                            st.resize(layer, (360, 720), order=0, preserve_range=True, anti_aliasing=False)
                            for layer in pft if layer is not None
                        ])
                        
                        if pfty.size > 0:
                            sd = np.ma.std(pfty,axis = 0) if pfty is not None else None
                            s = ax.pcolormesh(lon, lat, get_dryland(scl_AI, np.flip(sd, 0)), cmap=cmap,
                                              norm=norm)
                            ax.set_ylabel('g C m^-2')
                                
                    except KeyError:
                        pass
            if idx ==15:
                ax = fig.add_subplot(spec[i, j], projection=ccrs.PlateCarree())
                ax.coastlines()
                # Add state boundaries to plot
                ax.add_feature(cartopy.feature.STATES, edgecolor='black', linewidth=2)
                ax.set_global()
                ax.set_ylim(15,55)
                ax.set_xlim(-125,-97)
                gl = ax.gridlines(draw_labels=True, xlocs=[-120, -110, -100, -90], ylocs=[15, 25, 35, 45, 55], linestyle='--')
                gl.top_labels = False
                gl.right_labels = False
                if j != 0:
                    gl.left_labels = False
                if i != nrows - 1:
                    gl.bottom_labels = False
                # Plot data
                s = ax.pcolormesh(lon, lat, get_dryland(scl_AI, modis)/100, cmap=cmap,
                                              norm=norm)
                plt.ylabel('g C m^-2')
                ax.text(-125, 56, '%s' % (next(labels)), fontsize=50)
                ax.set_title('MODIS VCF', fontsize=50)
                
            if idx == 16:
                ax = fig.add_subplot(spec[i, j], projection=ccrs.PlateCarree())
                ax.coastlines()
                # Add state boundaries to plot
                ax.add_feature(cartopy.feature.STATES, edgecolor='black', linewidth=2)
                ax.set_global()
                ax.set_ylim(15,55)
                ax.set_xlim(-125,-97)
                gl = ax.gridlines(draw_labels=True, xlocs=[-120, -110, -100, -90], ylocs=[15, 25, 35, 45, 55], linestyle='--')
                gl.top_labels = False
                gl.right_labels = False
                if j != 0:
                    gl.left_labels = False
                if i != nrows - 1:
                    gl.bottom_labels = False
                # Plot data
                # Define the dimensions of the raster dataset
                num_rows = 497  # Example number of rows
                num_cols = 1158  # Example number of columns
                resolution = 0.05
                top_left_y = 49.40
                top_left_x = -124.75

                # Generate the row and column indices
                row = np.arange(num_rows)
                col = np.arange(num_cols)

                # Calculate latitude and longitude for each pixel
                latitude = top_left_y - resolution * row
                longitude = top_left_x + resolution * col
                s = plt.pcolormesh(longitude, latitude, np.ma.masked_where(reference_data==0,reference_data)/100, cmap=cmap, norm=norm)
                plt.ylabel('g C m^-2')
                ax.text(-125, 56, '%s' % (next(labels)), fontsize=50)
                ax.set_title('RAP', fontsize=50)
            
    # Add colorbar
    #fig.add_axes([left, bottom, width, height])
    cbar_ax = fig.add_axes([0.92, 0.22, 0.01, 0.55])
    cb = fig.colorbar(s, cax=cbar_ax, cmap=cmap, norm=norm, label=CB_label, extend='max')
    cb.ax.tick_params(labelsize=30)  # Adjust colorbar tick font size

    plt.tight_layout()
    plt.show()


# In[248]:


pft_sd_plot_2ref(sorted_pft_files,'allveg', model_list,std_allveg, dry_std_allveg, CB_label='SD All vegetation (Total PFT, 2001-2016)')


# In[249]:


pft_sd_plot_2ref(sorted_pft_files,'woody_allveg', model_list,std_woody, dry_std_woody, CB_label='SD Woody (PFT, 2001-2016)')


# In[250]:


pft_sd_plot_2ref(sorted_pft_files,'barren', model_list,std_barren, dry_std_barren, CB_label='SD Barren (PFT, 2001-2016)')


# In[255]:


def pft_sd_plot_modis(pft_files, pft_type, model_list, modis, CB_label = 'PFT Standard Deviation'):
    fig = plt.figure(figsize=(45, 55))
    model_order = ['CABLE POP', 'CLASSIC', 'CLM5', 'IBIS', 'ISBA CTRIP', 'JSBACH', 'JULES', 'ORCHIDEE', 'VISIT', 'YIBs', 'LPJ GUESS', 'LPJwsl', 'LPX Bern', 'OCN', 'SDGVM']
    ncols = 5
    nrows = 4
    spec = fig.add_gridspec(ncols=ncols, nrows=nrows, wspace=0.05, hspace=0.0)
    plt.rc('axes', titlesize=50)  # fontsize of the title
    plt.rc('legend', fontsize=40)  # fontsize of the legend
    plt.rc('font', size=40)  # controls default text size
    plt.rc('axes', labelsize=40)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=40)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=40)  # fontsize of the y tick labels
    labels = label_generator(case='lowercase letter', start='(', end=')')

    #cmap = colors.ListedColormap(sns.color_palette('YlGn', n_colors=10))
    #norm = colors.BoundaryNorm(bounds, cmap.N)
    cmap = plt.cm.YlGnBu 
    norm = colors.Normalize(vmin=0, vmax=0.20)
    for i in range(nrows):
        for j in range(ncols):
            idx = i * ncols + j
            if idx<15: 
                ax = fig.add_subplot(spec[i, j], projection=ccrs.PlateCarree())
                ax.coastlines()
                ax.add_feature(cartopy.feature.STATES, edgecolor='black', linewidth=2)
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
                if i == nrows - 2 and j>0:
                    gl.bottom_labels = True
                ax.text(-125, 56, '%s' % (next(labels)), fontsize=50)
                ax.set_title(model_order[idx], fontsize=50)
                
                # Check if model is in the model_list
                if model_order[idx] in model_list:
                    model_index = model_list.index(model_order[idx])
                    print(pft_files[model_index])
                    data = np.load(pft_files[model_index], allow_pickle=True)  # allow_pickle=True to handle object arrays
                    
                    try:
                        pft = data.get(pft_type, None)
                        # Check if pft is a valid, iterable array with dimensions
                        if pft is None or not hasattr(pft, '__iter__') or pft.ndim == 0:
                            continue  # Skip this iteration if pft_type is not valid

                        # Resize and handle only non-None layers in pft
                        pfty = np.array([
                            st.resize(layer, (360, 720), order=0, preserve_range=True, anti_aliasing=False)
                            for layer in pft if layer is not None
                        ])
                        
                        if pfty.size > 0:
                            sd = np.ma.std(pfty,axis = 0) if pfty is not None else None
                            s = ax.pcolormesh(lon, lat, get_dryland(AI, np.flip(sd, 0)), cmap=cmap,
                                              norm=norm)
                            ax.set_ylabel('g C m^-2')
                                
                    except KeyError:
                        pass
            if idx ==15:
                ax = fig.add_subplot(spec[i, j], projection=ccrs.PlateCarree())
                ax.coastlines()
                # Add state boundaries to plot
                ax.add_feature(cartopy.feature.STATES, edgecolor='black', linewidth=2)
                ax.set_global()
                ax.set_ylim(15,55)
                ax.set_xlim(-125,-97)
                gl = ax.gridlines(draw_labels=True, xlocs=[-120, -110, -100, -90], ylocs=[15, 25, 35, 45, 55], linestyle='--')
                gl.top_labels = False
                gl.right_labels = False
                if j != 0:
                    gl.left_labels = False
                if i != nrows - 1:
                    gl.bottom_labels = False
                # Plot data
                s = ax.pcolormesh(lon, lat, get_dryland(AI, modis)/100, cmap=cmap,
                                              norm=norm)
                plt.ylabel('g C m^-2')
                ax.text(-125, 56, '%s' % (next(labels)), fontsize=50)
                ax.set_title('MODIS VCF', fontsize=50)
                
    # Add colorbar
    #fig.add_axes([left, bottom, width, height])
    cbar_ax = fig.add_axes([0.92, 0.22, 0.01, 0.55])
    cb = fig.colorbar(s, cax=cbar_ax, cmap=cmap, norm=norm, label=CB_label, extend='max')
    cb.ax.tick_params(labelsize=30)  # Adjust colorbar tick font size

    plt.tight_layout()
    plt.show()


# In[256]:


pft_sd_plot_modis(sorted_pft_files,'nonwoody_allveg', model_list,std_nonwoody, CB_label='SD Non-woody (PFT, 2001-2016)')


# In[257]:


def pft_sd_plot_rap(pft_files, pft_type, model_list, reference_data, CB_label = 'PFT Standard Deviation'):
    fig = plt.figure(figsize=(45, 55))
    model_order = ['CABLE POP', 'CLASSIC', 'CLM5', 'IBIS', 'ISBA CTRIP', 'JSBACH', 'JULES', 'ORCHIDEE', 'VISIT', 'YIBs', 'LPJ GUESS', 'LPJwsl', 'LPX Bern', 'OCN', 'SDGVM']
    ncols = 5
    nrows = 4
    spec = fig.add_gridspec(ncols=ncols, nrows=nrows, wspace=0.05, hspace=0.0)
    plt.rc('axes', titlesize=50)  # fontsize of the title
    plt.rc('legend', fontsize=40)  # fontsize of the legend
    plt.rc('font', size=40)  # controls default text size
    plt.rc('axes', labelsize=40)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=40)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=40)  # fontsize of the y tick labels
    labels = label_generator(case='lowercase letter', start='(', end=')')

    #cmap = colors.ListedColormap(sns.color_palette('YlGn', n_colors=10))
    #norm = colors.BoundaryNorm(bounds, cmap.N)
    cmap = plt.cm.YlGnBu 
    norm = colors.Normalize(vmin=0, vmax=0.20)
    for i in range(nrows):
        for j in range(ncols):
            idx = i * ncols + j
            if idx<15: 
                ax = fig.add_subplot(spec[i, j], projection=ccrs.PlateCarree())
                ax.coastlines()
                ax.add_feature(cartopy.feature.STATES, edgecolor='black', linewidth=2)
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
                if i == nrows - 2 and j>0:
                    gl.bottom_labels = True
                ax.text(-125, 56, '%s' % (next(labels)), fontsize=50)
                ax.set_title(model_order[idx], fontsize=50)
                
                # Check if model is in the model_list
                if model_order[idx] in model_list:
                    model_index = model_list.index(model_order[idx])
                    print(pft_files[model_index])
                    data = np.load(pft_files[model_index], allow_pickle=True)  # allow_pickle=True to handle object arrays
                    
                    try:
                        pft = data.get(pft_type, None)
                        # Check if pft is a valid, iterable array with dimensions
                        if pft is None or not hasattr(pft, '__iter__') or pft.ndim == 0:
                            continue  # Skip this iteration if pft_type is not valid

                        # Resize and handle only non-None layers in pft
                        pfty = np.array([
                            st.resize(layer, (360, 720), order=0, preserve_range=True, anti_aliasing=False)
                            for layer in pft if layer is not None
                        ])
                        
                        if pfty.size > 0:
                            sd = np.ma.std(pfty,axis = 0) if pfty is not None else None
                            s = ax.pcolormesh(lon, lat, get_dryland(AI, np.flip(sd, 0)), cmap=cmap,
                                              norm=norm)
                            ax.set_ylabel('g C m^-2')
                                
                    except KeyError:
                        pass
            if idx ==15:
                ax = fig.add_subplot(spec[i, j], projection=ccrs.PlateCarree())
                ax.coastlines()
                # Add state boundaries to plot
                ax.add_feature(cartopy.feature.STATES, edgecolor='black', linewidth=2)
                ax.set_global()
                ax.set_ylim(15,55)
                ax.set_xlim(-125,-97)
                gl = ax.gridlines(draw_labels=True, xlocs=[-120, -110, -100, -90], ylocs=[15, 25, 35, 45, 55], linestyle='--')
                gl.top_labels = False
                gl.right_labels = False
                if j != 0:
                    gl.left_labels = False
                if i != nrows - 1:
                    gl.bottom_labels = False
                # Plot data
                # Define the dimensions of the raster dataset
                num_rows = 497  # Example number of rows
                num_cols = 1158  # Example number of columns
                resolution = 0.05
                top_left_y = 49.40
                top_left_x = -124.75

                # Generate the row and column indices
                row = np.arange(num_rows)
                col = np.arange(num_cols)

                # Calculate latitude and longitude for each pixel
                latitude = top_left_y - resolution * row
                longitude = top_left_x + resolution * col
                s = plt.pcolormesh(longitude, latitude, np.ma.masked_where(reference_data==0,reference_data)/100, cmap=cmap, norm=norm)
                plt.ylabel('g C m^-2')
                ax.text(-125, 56, '%s' % (next(labels)), fontsize=50)
                ax.set_title('RAP', fontsize=50)
            
    # Add colorbar
    #fig.add_axes([left, bottom, width, height])
    cbar_ax = fig.add_axes([0.92, 0.22, 0.01, 0.55])
    cb = fig.colorbar(s, cax=cbar_ax, cmap=cmap, norm=norm, label=CB_label, extend='max')
    cb.ax.tick_params(labelsize=30)  # Adjust colorbar tick font size

    plt.tight_layout()
    plt.show()


# In[258]:


pft_sd_plot_rap(sorted_pft_files,'allgrass', model_list, dry_std_nonwoody, CB_label='SD Grass (PFT, 2001-2016)')


# In[260]:


def pft_sd_plot(pft_files, pft_type, model_list, CB_label = 'PFT Standard Deviation'):
    fig = plt.figure(figsize=(45, 45))
    model_order = ['CABLE POP', 'CLASSIC', 'CLM5', 'IBIS', 'ISBA CTRIP', 'JSBACH', 'JULES', 'ORCHIDEE', 'VISIT', 'YIBs', 'LPJ GUESS', 'LPJwsl', 'LPX Bern', 'OCN', 'SDGVM']
    ncols = 5
    nrows = 3
    spec = fig.add_gridspec(ncols=ncols, nrows=nrows, wspace=0.05, hspace=0.0)
    plt.rc('axes', titlesize=50)  # fontsize of the title
    plt.rc('legend', fontsize=40)  # fontsize of the legend
    plt.rc('font', size=40)  # controls default text size
    plt.rc('axes', labelsize=40)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=40)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=40)  # fontsize of the y tick labels
    labels = label_generator(case='lowercase letter', start='(', end=')')

    #cmap = colors.ListedColormap(sns.color_palette('YlGn', n_colors=10))
    #norm = colors.BoundaryNorm(bounds, cmap.N)
    cmap = plt.cm.YlGnBu 
    norm = colors.Normalize(vmin=0, vmax=0.20)
    for i in range(nrows):
        for j in range(ncols):
            idx = i * ncols + j
            if idx<15: 
                ax = fig.add_subplot(spec[i, j], projection=ccrs.PlateCarree())
                ax.coastlines()
                ax.add_feature(cartopy.feature.STATES, edgecolor='black', linewidth=2)
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

                ax.text(-125, 56, '%s' % (next(labels)), fontsize=50)
                ax.set_title(model_order[idx], fontsize=50)
                
                # Check if model is in the model_list
                if model_order[idx] in model_list:
                    model_index = model_list.index(model_order[idx])
                    print(pft_files[model_index])
                    data = np.load(pft_files[model_index], allow_pickle=True)  # allow_pickle=True to handle object arrays
                    
                    try:
                        pft = data.get(pft_type, None)
                        # Check if pft is a valid, iterable array with dimensions
                        if pft is None or not hasattr(pft, '__iter__') or pft.ndim == 0:
                            continue  # Skip this iteration if pft_type is not valid

                        # Resize and handle only non-None layers in pft
                        pfty = np.array([
                            st.resize(layer, (360, 720), order=0, preserve_range=True, anti_aliasing=False)
                            for layer in pft if layer is not None
                        ])
                        
                        if pfty.size > 0:
                            sd = np.ma.std(pfty,axis = 0) if pfty is not None else None
                            s = ax.pcolormesh(lon, lat, get_dryland(AI, np.flip(sd, 0)), cmap=cmap,
                                              norm=norm)
                            ax.set_ylabel('g C m^-2')
                                
                    except KeyError:
                        pass

    # Add colorbar
    #fig.add_axes([left, bottom, width, height])
    cbar_ax = fig.add_axes([0.92, 0.22, 0.01, 0.55])
    cb = fig.colorbar(s, cax=cbar_ax, cmap=cmap, norm=norm, label=CB_label, extend='max')
    cb.ax.tick_params(labelsize=30)  # Adjust colorbar tick font size

    plt.tight_layout()
    plt.show()


# In[261]:


pft_sd_plot(sorted_pft_files,'allgrass', model_list, CB_label='SD (Grass, 2001-2016)')
pft_sd_plot(sorted_pft_files,'c3grass', model_list, CB_label='SD C3 Grass (PFT, 2001-2016)')
pft_sd_plot(sorted_pft_files,'c4grass', model_list, CB_label='SD C4 Grass (PFT, 2001-2016)')


# In[263]:


def pft_sd_plot(pft_files, pft_type, model_list, CB_label = 'PFT Standard Deviation'):
    fig = plt.figure(figsize=(45, 45))
    model_order = ['CABLE POP', 'CLASSIC', 'CLM5', 'IBIS', 'ISBA CTRIP', 'JSBACH', 'JULES', 'ORCHIDEE', 'VISIT', 'YIBs', 'LPJ GUESS', 'LPJwsl', 'LPX Bern', 'OCN', 'SDGVM']
    ncols = 5
    nrows = 3
    spec = fig.add_gridspec(ncols=ncols, nrows=nrows, wspace=0.05, hspace=0.0)
    plt.rc('axes', titlesize=50)  # fontsize of the title
    plt.rc('legend', fontsize=40)  # fontsize of the legend
    plt.rc('font', size=40)  # controls default text size
    plt.rc('axes', labelsize=40)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=40)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=40)  # fontsize of the y tick labels
    labels = label_generator(case='lowercase letter', start='(', end=')')

    #cmap = colors.ListedColormap(sns.color_palette('YlGn', n_colors=10))
    #norm = colors.BoundaryNorm(bounds, cmap.N)
    cmap = plt.cm.YlGnBu 
    norm = colors.Normalize(vmin=0, vmax=0.20)
    for i in range(nrows):
        for j in range(ncols):
            idx = i * ncols + j
            if idx<15: 
                ax = fig.add_subplot(spec[i, j], projection=ccrs.PlateCarree())
                ax.coastlines()
                ax.add_feature(cartopy.feature.STATES, edgecolor='black', linewidth=2)
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

                ax.text(-125, 56, '%s' % (next(labels)), fontsize=50)
                ax.set_title(model_order[idx], fontsize=50)
                
                # Check if model is in the model_list
                if model_order[idx] in model_list:
                    model_index = model_list.index(model_order[idx])
                    print(pft_files[model_index])
                    data = np.load(pft_files[model_index], allow_pickle=True)  # allow_pickle=True to handle object arrays
                    
                    try:
                        pft = data.get(pft_type, None)
                        # Check if pft is a valid, iterable array with dimensions
                        if pft is None or not hasattr(pft, '__iter__') or pft.ndim == 0:
                            continue  # Skip this iteration if pft_type is not valid

                        # Resize and handle only non-None layers in pft
                        pfty = np.array([
                            st.resize(layer, (360, 720), order=0, preserve_range=True, anti_aliasing=False)
                            for layer in pft if layer is not None
                        ])
                        
                        if pfty.size > 0:
                            sd = np.ma.std(pfty,axis = 0) if pfty is not None else None
                            s = ax.pcolormesh(lon, lat, get_dryland(scl_AI, np.flip(sd, 0)), cmap=cmap,
                                              norm=norm)
                            ax.set_ylabel('g C m^-2')
                                
                    except KeyError:
                        pass

    # Add colorbar
    #fig.add_axes([left, bottom, width, height])
    cbar_ax = fig.add_axes([0.92, 0.22, 0.01, 0.55])
    cb = fig.colorbar(s, cax=cbar_ax, cmap=cmap, norm=norm, label=CB_label, extend='max')
    cb.ax.tick_params(labelsize=30)  # Adjust colorbar tick font size

    plt.tight_layout()
    plt.show()


# In[264]:


pft_sd_plot(sorted_pft_files,'allcrop', model_list, CB_label='SD (Crop, 2001-2016)')
pft_sd_plot(sorted_pft_files,'c3crop',model_list, CB_label='SD (C3 Crop, 2001-2016)')
pft_sd_plot(sorted_pft_files,'c4crop', model_list, CB_label='SD (C4 Crop, 2001-2016)')


# In[ ]:




