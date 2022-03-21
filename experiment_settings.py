"""Experimental settings

Functions
---------
get_settings(experiment_name)
"""

__author__ = "Elizabeth A. Barnes and Noah Diffenbaugh"
__date__   = "20 March 2022"


def get_settings(experiment_name):
    experiments = {  
        
        "exp0": {
            "save_model": True,
            "n_models": 10,            # the number of networks to train
            "ssp" : "370",             #[options: '126' or '370']
            "gcmsub" : "ALL",          #[options: 'ALL' or 'UNIFORM'
            "obsdata" : "BEST",        #[options: 'BEST' or 'GISTEMP'
            "target_temp": 1.5,
            "n_train_val_test" : (7,2,1),
            "baseline_yr_bounds": (1850,1899),
            "training_yr_bounds": (1970,2100),
            "anomaly_yr_bounds": (1951,1980),
            "anomalies": True,         #[options: True or False]
            "remove_map_mean": False,  #[options: False or "weighted" or "raw"]

            "network_type": 'shash2',  #[options: "reg" or "shash2"]
            "hiddens": [10,10],
            "dropout_rate": 0.,
            "ridge_param": [1.0,0.0], 
            "learning_rate": 0.00001,  # reg->0.0001, shash2->.00005 or .00001
            "batch_size": 64,
            "rng_seed": 8889,
            "seed": None,
            "act_fun": ["relu","relu"],
            "n_epochs": 25_000,
            "patience": 50,
        },        

    }
    
    exp_dict = experiments[experiment_name]
    exp_dict['exp_name'] = experiment_name

    return exp_dict

