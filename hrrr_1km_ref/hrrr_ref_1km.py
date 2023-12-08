from herbie import Herbie, Herbie_latest, FastHerbie
import pandas as pd
from toolbox import EasyMap, pc
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
from datetime import datetime, timezone, timedelta
import os
import requests
import geopandas as gpd

r = requests.get('http://worldclockapi.com/api/json/utc/now').json()
input_time_string=r['currentDateTime']
# Convert the string to a datetime object
input_datetime = datetime.strptime(input_time_string, "%Y-%m-%dT%H:%MZ")

# Subtract 2 hours from the datetime object
adjusted_datetime = input_datetime - timedelta(hours=2)

# Format the adjusted datetime object as "YYYY-MM-DD-HH"
adjusted_formatted_time = adjusted_datetime.strftime('%Y-%m-%d-%H')

####Read in airport lat/lons####
file_path = 'blue_cities.csv'
icao_data = pd.read_csv(file_path)

####Set map region####

####Southeast####
se_extent=[-100, -72,23,40]
####Northeast####
ne_extent=[-95, -70,35,48]
####Southwest####
sw_extent=[-126.5, -90,25,40]
####Northwest####
nw_extent=[-129, -95,35,51]
extents=[se_extent,ne_extent,sw_extent,nw_extent]

def radar_colormap():
    nws_reflectivity_colors = [
    "#646464", # 0
    "#04e9e7", # 5
    "#019ff4", # 10
    "#0300f4", # 15
    "#02fd02", # 20
    "#01c501", # 25
    "#008e00", # 30
    "#fdf802", # 35
    "#e5bc00", # 40
    "#fd9500", # 45
    "#fd0000", # 50
    "#d40000", # 55
    "#bc0000", # 60
    "#f800fd", # 65
    "#9854c6", # 70
    "#fdfdfd" # 75
    ]

    return mpl.colors.ListedColormap(nws_reflectivity_colors)
    
def plot_airports(ax, icao_data):
    for i, row in icao_data.iterrows():
        if extent[0] <= row['Longitude'] <= extent[1] and extent[2] <= row['Latitude'] <= extent[3]:
           ax.plot(row['Longitude'], row['Latitude'], 'ro', markersize=5, transform=ccrs.PlateCarree())
           ax.text(row['Longitude'], row['Latitude'], row['ICAO'], transform=ccrs.PlateCarree(), fontsize=8)


fname=r'/home/shawn/Airspace_Boundary.shp'
data=gpd.read_file(fname)
artcc=data[(data.TYPE_CODE == 'ARTCC')]

hour=adjusted_datetime.strftime('%H')
if (int(hour) == 18 or int(hour) == 0 or int(hour) == 6 or int(hour) == 12):
 fhours=49
else:
 fhours=19

for fxx in range(fhours):
 H = Herbie(adjusted_formatted_time, searchString=":REFD:1000 m:", save_dir='/home/shawn/hrrr_grib', model="hrrr",product="sfc",freq="1H",member=1,fxx=fxx,verbose=True,ovewrite=True)
 for extent in extents:
  #########Create 1km AGL reflectivity plots###########
  ref=H.xarray("REFD:1000 m", remove_grib=False)
  ax = EasyMap("50m", crs=ref.herbie.crs, figsize=[16, 12], dpi=200).STATES().ax
  cmapref = radar_colormap()
  norm = mpl.colors.Normalize(vmin=0, vmax=80)
  cmapref.set_under('white')
  p = ax.pcolormesh(ref.longitude, ref.latitude, ref.refd,transform=pc, cmap=cmapref, norm=norm)
  ax.add_geometries(artcc.geometry, crs=ccrs.PlateCarree(), linewidth=0.50, facecolor='none', edgecolor='gray')
  plot_airports(ax, icao_data)
  ax.set_extent(extent)
  if extent == ne_extent:
   extent_name="Northeast"
  elif extent == se_extent:
   extent_name="Southeast"
  elif extent == sw_extent:
   extent_name="Southwest" 
  elif extent == nw_extent:
   extent_name="Northwest"
  plt.colorbar(p,ax=ax,orientation="horizontal",pad=0.01,shrink=0.8,)
  ax.set_title(
      f"{ref.model.upper()}\nInit: {ref.time.dt.strftime('%H:%M UTC %d %b %Y').item()}   Valid: {ref.valid_time.dt.strftime('%H:%M UTC %d %b %Y').item()}",
      loc="left",
  )
  ax.set_title("1km AGL simulated reflectivity (dBZ)", loc="right")
  #ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
  plt.savefig(extent_name+'/1kmref_' + str(adjusted_formatted_time) + '_' + str(fxx) + '.png',bbox_inches='tight', pad_inches=0)
  plt.figure().clear()
  plt.close()
  plt.cla()
  plt.clf()
   
 
