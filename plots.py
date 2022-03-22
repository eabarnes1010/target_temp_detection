"""Metrics for generic plotting.

Functions
---------
plot_metrics(history,metric)
plot_metrics_panels(history, settings)
plot_map(x, clim=None, title=None, text=None, cmap='RdGy')
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy as ct
import numpy.ma as ma
import cartopy.crs as ccrs
import cartopy.feature as cfeature

mpl.rcParams["figure.facecolor"] = "white"
mpl.rcParams["figure.dpi"] = 150


def plot_metrics(history,metric):
    
    imin = np.argmin(history.history['val_loss'])
    
    plt.plot(history.history[metric], label='training')
    plt.plot(history.history['val_' + metric], label='validation')
    plt.title(metric)
    plt.axvline(x=imin, linewidth=.5, color='gray',alpha=.5)
    plt.legend()
    
def plot_metrics_panels(history, settings):
    
    if(settings["network_type"]=="reg"):
        error_name = "mae"
    elif settings['network_type'] == 'shash2':
        error_name = "custom_mae"
    else: 
        raise NotImplementedError('no such network_type')
    
    imin = len(history.history[error_name])
    plt.subplots(figsize=(20,4))
    
    plt.subplot(1,4,1)
    plot_metrics(history,'loss')
    plt.ylim(0,10.)

    plt.subplot(1,4,2)
    plot_metrics(history,error_name)
    plt.ylim(0,10)

    try:
        plt.subplot(1,4,3)
        plot_metrics(history,'interquartile_capture')

        plt.subplot(1,4,4)
        plot_metrics(history,'sign_test')
    except:
        pass
    
    
def plot_map(x, clim=None, title=None, text=None, cmap='RdGy'):
    plt.pcolor(x,
               cmap=cmap,
              )
    plt.clim(clim)
    plt.colorbar()
    plt.title(title,fontsize=15,loc='right')    
    plt.yticks([])
    plt.xticks([])
    
    plt.text(0.01, 1.0, text, fontfamily='monospace', fontsize='small', va='bottom',transform=plt.gca().transAxes)    
    
def drawOnGlobe(ax, map_proj, data, lats, lons, cmap='coolwarm', vmin=None, vmax=None, inc=None, cbarBool=True, contourMap=[], contourVals = [], fastBool=False, extent='both'):

    data_crs = ct.crs.PlateCarree()
    data_cyc, lons_cyc = add_cyclic_point(data, coord=lons) #fixes white line by adding point#data,lons#ct.util.add_cyclic_point(data, coord=lons) #fixes white line by adding point
    data_cyc = data
    lons_cyc = lons
    
    
#     ax.set_global()
#     ax.coastlines(linewidth = 1.2, color='black')
#     ax.add_feature(cartopy.feature.LAND, zorder=0, scale = '50m', edgecolor='black', facecolor='black')    
    land_feature = cfeature.NaturalEarthFeature(
        category='physical',
        name='land',
        scale='50m',
        facecolor='None',
        edgecolor = 'k'
    )
    ax.add_feature(land_feature)
#     ax.GeoAxes.patch.set_facecolor('black')
    
    if(fastBool):
        image = ax.pcolormesh(lons_cyc, lats, data_cyc, transform=data_crs, cmap=cmap)
#         image = ax.contourf(lons_cyc, lats, data_cyc, np.linspace(0,vmax,20),transform=data_crs, cmap=cmap)
    else:
        image = ax.pcolor(lons_cyc, lats, data_cyc, transform=data_crs, cmap=cmap,shading='auto')
    
    if(np.size(contourMap) !=0 ):
        contourMap_cyc, __ = add_cyclic_point(contourMap, coord=lons) #fixes white line by adding point
        ax.contour(lons_cyc,lats,contourMap_cyc,contourVals, transform=data_crs, colors='fuchsia')
    
    if(cbarBool):
        cb = plt.colorbar(image, shrink=.45, orientation="horizontal", pad=.02, extend=extent)
        cb.ax.tick_params(labelsize=6) 
    else:
        cb = None

    image.set_clim(vmin,vmax)
    
    return cb, image   

def add_cyclic_point(data, coord=None, axis=-1):

    # had issues with cartopy finding utils so copied for myself
    
    if coord is not None:
        if coord.ndim != 1:
            raise ValueError('The coordinate must be 1-dimensional.')
        if len(coord) != data.shape[axis]:
            raise ValueError('The length of the coordinate does not match '
                             'the size of the corresponding dimension of '
                             'the data array: len(coord) = {}, '
                             'data.shape[{}] = {}.'.format(
                                 len(coord), axis, data.shape[axis]))
        delta_coord = np.diff(coord)
        if not np.allclose(delta_coord, delta_coord[0]):
            raise ValueError('The coordinate must be equally spaced.')
        new_coord = ma.concatenate((coord, coord[-1:] + delta_coord[0]))
    slicer = [slice(None)] * data.ndim
    try:
        slicer[axis] = slice(0, 1)
    except IndexError:
        raise ValueError('The specified axis does not correspond to an '
                         'array dimension.')
    new_data = ma.concatenate((data, data[tuple(slicer)]), axis=axis)
    if coord is None:
        return_value = new_data
    else:
        return_value = new_data, new_coord
    return return_value    