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
    imin = len(history.history['custom_mae'])
    plt.subplots(figsize=(20,4))

    plt.subplot(1,4,1)
    plot_metrics(history,'loss')
    plt.ylim(0,10.)

    plt.subplot(1,4,2)
    plot_metrics(history,'custom_mae')
    plt.ylim(0,10)

    plt.subplot(1,4,3)
    plot_metrics(history,'interquartile_capture')

    plt.subplot(1,4,4)
    plot_metrics(history,'sign_test')
       
    
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