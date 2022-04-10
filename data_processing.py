"""Build the split and scaled training, validation and testing data.

Functions
---------
get_members(settings)
get_observations(directory, settings)
get_cmip_data(directory, rng, settings)
get_labels(da, settings, plot=False)
preprocess_data(da, MEMBERS, settings)
make_data_split(da, data, f_labels, f_years, labels, years, MEMBERS, settings)
"""
import numpy as np
import pandas as pd
import file_methods


__author__ = "Elizabeth A. Barnes and Noah Diffenbaugh"
__version__ = "20 March 2022"


def get_members(settings):
    n_train = settings["n_train_val_test"][0]
    n_val   = settings["n_train_val_test"][1]
    n_test  = settings["n_train_val_test"][2]
    all_members = np.arange(0,n_train+n_val+n_test)

    return n_train, n_val, n_test, all_members

def get_observations(directory, settings):
    if settings["obsdata"] == "BEST":
        nc_filename_obs = 'Land_and_Ocean_LatLong1_185001_202112_ann_mean_2pt5degree.nc'
    elif settings["obsdata"] == 'GISS': 
        nc_filename_obs = 'gistemp1200_GHCNv4_ERSSTv5_188001_202112_ann_mean_2pt5degree.nc'
    else:
        raise NotImplementedError('no such obs data')

    da_obs = file_methods.get_netcdf_da(directory + nc_filename_obs)
    global_mean_obs = compute_global_mean(da_obs)

    data_obs = preprocess_data(da_obs, MEMBERS=None, settings=settings) 
    x_obs = data_obs.values.reshape((data_obs.shape[0],data_obs.shape[1]*data_obs.shape[2]))
    if settings["anomalies"]:
        print('observations: filling NaNs with zeros')
        x_obs = np.nan_to_num(x_obs,0.)

    print('np.shape(x_obs) = ' + str(np.shape(x_obs)))
    
    return data_obs, x_obs, global_mean_obs

def compute_global_mean(da):
    weights = np.cos(np.deg2rad(da.lat))
    weights.name = "weights"
    temp_weighted = da.weighted(weights)
    global_mean = temp_weighted.mean(("lon", "lat"), skipna=False)
    
    return global_mean

def get_cmip_data(directory, settings, verbose=1):
    data_train, data_val, data_test = None, None, None
    labels_train, labels_val, labels_test = None, None, None
    years_train, years_val, years_test = None, None, None
    target_years = []
    
    N_TRAIN, N_VAL, N_TEST, ALL_MEMBERS = get_members(settings)
    
    rng_cmip = np.random.default_rng(settings["seed"])
    train_members = rng_cmip.choice(ALL_MEMBERS, size=N_TRAIN, replace=False)
    val_members   = rng_cmip.choice(np.setdiff1d(ALL_MEMBERS,train_members), size=N_VAL, replace=False)
    test_members  = rng_cmip.choice(np.setdiff1d(ALL_MEMBERS,np.append(train_members[:],val_members)), size=N_TEST, replace=False)
    if verbose == 1:
        print(train_members, val_members, test_members)
    
    # save the meta data
    settings['train_members'] = train_members.tolist()
    settings['val_members'] = val_members.tolist()
    settings['test_members'] = test_members.tolist()
    
    # loop through and get the data
    filenames = file_methods.get_cmip_filenames(settings, verbose=0)
    for f in filenames:
        if verbose == 1:
            print(f)
        da = file_methods.get_netcdf_da(directory + f)
        f_labels, f_years, f_target_year = get_labels(da, settings,verbose=verbose)

        # create sets of train / validaton / test
        target_years = np.append(target_years,f_target_year)
        data_train, labels_train, years_train = make_data_split(da, 
                                                                data_train, 
                                                                f_labels, 
                                                                f_years, 
                                                                labels_train,
                                                                years_train,
                                                                train_members,
                                                                settings,
                                                               )
        data_val, labels_val, years_val       = make_data_split(da, 
                                                                data_val, 
                                                                f_labels, 
                                                                f_years, 
                                                                labels_val,
                                                                years_val,
                                                                val_members,
                                                                settings,
                                                               )
        data_test, labels_test, years_test    = make_data_split(da, 
                                                                data_test, 
                                                                f_labels, 
                                                                f_years, 
                                                                labels_test,
                                                                years_test,
                                                                test_members,
                                                                settings,
                                                               )

    YEARS_UNIQUE = np.unique(years_train)
    if verbose == 1:
        print('---------------------------')                
        print('data_train.shape = ' + str(np.shape(data_train)))
        print('data_val.shape = ' + str(np.shape(data_val)))
        print('data_test.shape = ' + str(np.shape(data_test)))
    
    x_train = data_train.reshape((data_train.shape[0]*data_train.shape[1],data_train.shape[2]*data_train.shape[3]))
    x_val   = data_val.reshape((data_val.shape[0]*data_val.shape[1],data_val.shape[2]*data_val.shape[3]))
    x_test  = data_test.reshape((data_test.shape[0]*data_test.shape[1],data_test.shape[2]*data_test.shape[3]))

    y_train = labels_train.reshape((data_train.shape[0]*data_train.shape[1],))
    y_val   = labels_val.reshape((data_val.shape[0]*data_val.shape[1],))
    y_test  = labels_test.reshape((data_test.shape[0]*data_test.shape[1],))

    y_yrs_train = years_train.reshape((data_train.shape[0]*data_train.shape[1],))
    y_yrs_val   = years_val.reshape((data_val.shape[0]*data_val.shape[1],))
    y_yrs_test  = years_test.reshape((data_test.shape[0]*data_test.shape[1],))
    if verbose == 1:
        print(x_train.shape, y_train.shape, y_yrs_train.shape)
        print(x_val.shape, y_val.shape, y_yrs_val.shape)
        print(x_test.shape, y_test.shape, y_yrs_test.shape)  
    
    # make onehot vectors for training
    if settings["network_type"] == 'shash2':
        onehot_train = np.zeros((x_train.shape[0],2))
        onehot_train[:,0] = y_train.astype('float32')
        onehot_val = np.zeros((x_val.shape[0],2))    
        onehot_val[:,0] = y_val.astype('float32')
        onehot_test = np.zeros((x_test.shape[0],2))    
        onehot_test[:,0] = y_test.astype('float32')
    else:
        onehot_train = np.copy(y_train)
        onehot_val = np.copy(y_val)
        onehot_test = np.copy(y_test)    
    
    map_shape = np.shape(data_train)[2:]
    
    return x_train, x_val, x_test, y_train, y_val, y_test, onehot_train, onehot_val, onehot_test, y_yrs_train, y_yrs_val, y_yrs_test, target_years, map_shape, settings


def get_labels(da, settings, plot=False, verbose=1):
    # compute the ensemble mean, global mean temperature
    # these computations should be based on the training set only
    da_ens = da.mean(axis=0)
    weights = np.cos(np.deg2rad(da_ens.lat))
    weights.name = "weights"
    temp_weighted = da_ens.weighted(weights)
    global_mean = temp_weighted.mean(("lon", "lat"))
    
    global_mean_ens = da.weighted(weights)
    global_mean_ens = global_mean_ens.mean(("lon","lat"))
    
    # compute the target year 
    if settings["gcmsub"] == 'MAX':
        
        baseline_mean = global_mean.sel(time=slice(str(settings["baseline_yr_bounds"][0]),str(settings["baseline_yr_bounds"][1]))).mean('time')       
        imax = np.argmax(global_mean.values)
        target_year = global_mean["time"].values[imax].year
        temp_reached = np.round(global_mean.values[imax]-baseline_mean.values,2)
    else: 
        temp_reached = settings["target_temp"]
        try:
            baseline_mean = global_mean.sel(time=slice(str(settings["baseline_yr_bounds"][0]),str(settings["baseline_yr_bounds"][1]))).mean('time')
            iwarmer = np.where(global_mean.values > baseline_mean.values+settings["target_temp"])[0]
            target_year = global_mean["time"].values[iwarmer[0]].year
        except:
            if settings["gcmsub"] == 'FORCE' or settings["gcmsub"] == 'MIROC':
                target_year = global_mean["time"].values[-1].year
            elif settings["gcmsub"] == 'EXTEND' or settings["gcmsub"] == 'MIROC':    
                target_year = 2150
            else:
                raise ValueError('****no such target****')

    # plot the calculation to make sure things make sense
    if plot == True:
        for ens in np.arange(0,global_mean_ens.shape[0]):
            global_mean_ens[ens,:].plot(linewidth=1.0,color="gray",alpha=.5)
        global_mean.plot(linewidth=2,label='data',color="aqua")
        plt.axhline(y=baseline_mean, color='k', linestyle='-', label='baseline temp')
        plt.axhline(y=baseline_mean+settings["target_temp"], color='tab:blue',linewidth=1., linestyle='--', label='target temp')
        plt.axvline(x=target_year,color='tab:blue',linewidth=1., linestyle='--', label='target year')
        global_mean_obs.plot(linewidth=2,label='data',color="tab:orange")        
        plt.xlabel('year')
        plt.ylabel('temp (K)')
        plt.title(f + '\ntargets [' + str(target_year.year) + ', ' + str(settings["target_temp"]) + 'C]',
                  fontsize = 8,
                 )
        plt.show()
    
    # define the labels
    if verbose == 1:
        print('TARGET_YEAR = ' + str(target_year) + ', TARGET_TEMP = ' + str(temp_reached))
    labels = target_year - da['time.year'].values
    
    return labels, da['time.year'].values, target_year

def preprocess_data(da, MEMBERS, settings):

    if MEMBERS is None:
        new_data = da
    else:
        new_data = da[MEMBERS,:,:,:]

    if settings["anomalies"] is True:
        new_data = new_data - new_data.sel(time=slice(str(settings["anomaly_yr_bounds"][0]),str(settings["anomaly_yr_bounds"][1]))).mean('time')
        
    if settings["remove_map_mean"]  == 'raw':
        new_data = new_data - new_data.mean(("lon","lat"))
    elif settings["remove_map_mean"] == 'weighted':
        weights = np.cos(np.deg2rad(new_data.lat))
        weights.name = "weights"
        new_data_weighted = new_data.weighted(weights)
        new_data = new_data - new_data_weighted.mean(("lon","lat"))

    return new_data

def make_data_split(da, data, f_labels, f_years, labels, years, MEMBERS, settings):

    # process the data, i.e. compute anomalies, subtract the mean, etc.
    new_data = preprocess_data(da, MEMBERS, settings)    
    
    # only train on certain samples
    iyears = np.where((f_years >= settings["training_yr_bounds"][0]) & (f_years <= settings["training_yr_bounds"][1]))[0]    
    f_years = f_years[iyears]
    f_labels = f_labels[iyears]            
    new_data = new_data[:,iyears,:,:]
    
    if data is None:
        data = new_data.values
        labels = np.tile(f_labels,(len(MEMBERS),1))        
        years = np.tile(f_years,(len(MEMBERS),1))
    else:
        data = np.concatenate((data,new_data.values),axis=0)        
        labels = np.concatenate((labels,np.tile(f_labels,(len(MEMBERS),1))),axis=0)        
        years = np.concatenate((years,np.tile(f_years,(len(MEMBERS),1))),axis=0)
    
    return data, labels, years