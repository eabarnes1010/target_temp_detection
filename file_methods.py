"""Functions for working with generic files.

Functions
---------
get_model_name(settings)
get_netcdf_da(filename)
save_pred_obs(pred_vector, filename)
save_tf_model(model, model_name, directory, settings)
get_cmip_filenames(settings, verbose=0)
"""

import xarray as xr
import json
import pickle
import tensorflow as tf

__author__ = "Elizabeth A. Barnes and Noah Diffenbaugh"
__version__ = "20 March 2022"

def get_model_name(settings):
    # model_name = (settings["exp_name"] + '_' +
    #               'ssp' + settings["ssp"] + '_' +
    #               str(settings["target_temp"]) + '_' +
    #               'gcmsub' + settings["gcmsub"] + '_' +
    #               settings["network_type"] + 
    #               '_rng' + str(settings["rng_seed"]) + 
    #               '_seed' + str(settings["seed"])
    #              )
    model_name = (settings["exp_name"] +
                  '_seed' + str(settings["seed"])
                 )
    
    return model_name


def get_netcdf_da(filename):
    da = xr.open_dataarray(filename)
    return da    


def save_pred_obs(pred_vector, filename):
    with open(filename + '.pickle', 'wb') as f:
        pickle.dump(pred_vector, f)
        

def save_tf_model(model, model_name, directory, settings):
    
    # save the tf model
    tf.keras.models.save_model(model, directory + model_name + "_model", overwrite=True)

    # example code for loading a tf model
    # model = tf.keras.models.load_model(directory + model_name + "_model", compile = False)
    # predictions = model.predict(x_data)

    # save the meta data
    with open(directory + model_name + '_metadata.json', 'w') as json_file:
        json_file.write(json.dumps(settings))
        
def get_cmip_filenames(settings, verbose=0):
    if settings["ssp"] == '370' and settings["gcmsub"] == 'ALL':
        filenames = ('tas_Amon_historical_ssp370_CanESM5_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp370_ACCESS-ESM1-5_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp370_UKESM1-0-LL_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp370_MIROC-ES2L_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp370_GISS-E2-1-G_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp370_IPSL-CM6A-LR_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp370_CESM2-LE2-smbb_r1-10_ncecat_ann_mean_2pt5degree.nc',
                    )
    elif settings["ssp"] == '370' and settings["gcmsub"] == 'UNIFORM' and settings["target_temp"] == 2.0:
        filenames = ('tas_Amon_historical_ssp370_CanESM5_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp370_ACCESS-ESM1-5_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp370_UKESM1-0-LL_r1-10_ncecat_ann_mean_2pt5degree.nc',
                    )
    elif settings["ssp"] == '370' and settings["gcmsub"] == 'UNIFORM':
        filenames = ('tas_Amon_historical_ssp370_CanESM5_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp370_ACCESS-ESM1-5_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp370_UKESM1-0-LL_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp370_MIROC-ES2L_r1-10_ncecat_ann_mean_2pt5degree.nc',
                    )
    elif settings["ssp"] == '126' and settings["gcmsub"] == 'ALL':
        filenames = ('tas_Amon_historical_ssp126_CanESM5_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp126_ACCESS-ESM1-5_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp126_UKESM1-0-LL_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp126_MIROC-ES2L_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp126_MIROC6_r1-10_ncecat_ann_mean_2pt5degree.nc',
                    )
    elif settings["ssp"] == '126' and settings["gcmsub"] == 'ALL' and settings["target_temp"] == 2.0:
        filenames = ('tas_Amon_historical_ssp126_CanESM5_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp126_ACCESS-ESM1-5_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp126_UKESM1-0-LL_r1-10_ncecat_ann_mean_2pt5degree.nc',
                    )
    elif settings["ssp"] == '126' and settings["gcmsub"] == 'UNIFORM':
        filenames = ('tas_Amon_historical_ssp126_CanESM5_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp126_ACCESS-ESM1-5_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp126_UKESM1-0-LL_r1-10_ncecat_ann_mean_2pt5degree.nc',
                     'tas_Amon_historical_ssp126_MIROC-ES2L_r1-10_ncecat_ann_mean_2pt5degree.nc',
                    )
    else:
        raise NotImplementedError('no such SSP')

    if verbose!=0:
        print(filenames)
    
    return filenames