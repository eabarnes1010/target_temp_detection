"""Experimental settings

Functions
---------
get_settings(experiment_name)
"""

__author__ = "Elizabeth A. Barnes and Noah Diffenbaugh"
__date__   = "25 March 2022"


def get_settings(experiment_name):
    experiments = {  
        #---------------------- MAIN SIMULATIONS ---------------------------
        "exp15C_370": { 
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
            "hiddens": [25,25],
            "dropout_rate": 0.,
            "ridge_param": [10.0,0.0], 
            "learning_rate": 0.00001,  # reg->0.0001, shash2->.00005 or .00001
            "batch_size": 64,
            "rng_seed": 8889,
            "seed": None,
            "act_fun": ["relu","relu"],
            "n_epochs": 25_000,
            "patience": 50,
        },    
        
        "exp11C_370": { 
            "save_model": True,
            "n_models": 10,            # the number of networks to train
            "ssp" : "370",             #[options: '126' or '370']
            "gcmsub" : "ALL",          #[options: 'ALL' or 'UNIFORM'
            "obsdata" : "BEST",        #[options: 'BEST' or 'GISTEMP'
            "target_temp": 1.1,
            "n_train_val_test" : (7,2,1),
            "baseline_yr_bounds": (1850,1899),
            "training_yr_bounds": (1970,2100),
            "anomaly_yr_bounds": (1951,1980),
            "anomalies": True,         #[options: True or False]
            "remove_map_mean": False,  #[options: False or "weighted" or "raw"]

            "network_type": 'shash2',  #[options: "reg" or "shash2"]
            "hiddens": [25,25],
            "dropout_rate": 0.,
            "ridge_param": [10.0,0.0], 
            "learning_rate": 0.00001,  # reg->0.0001, shash2->.00005 or .00001
            "batch_size": 64,
            "rng_seed": 8889,
            "seed": None,
            "act_fun": ["relu","relu"],
            "n_epochs": 25_000,
            "patience": 50,
        },  
        "exp20C_370": {
            "save_model": True,
            "n_models": 10,            # the number of networks to train
            "ssp" : "370",             #[options: '126' or '370']
            "gcmsub" : "ALL",          #[options: 'ALL' or 'UNIFORM'
            "obsdata" : "BEST",        #[options: 'BEST' or 'GISTEMP'
            "target_temp": 2.0,
            "n_train_val_test" : (7,2,1),
            "baseline_yr_bounds": (1850,1899),
            "training_yr_bounds": (1970,2100),
            "anomaly_yr_bounds": (1951,1980),
            "anomalies": True,         #[options: True or False]
            "remove_map_mean": False,  #[options: False or "weighted" or "raw"]

            "network_type": 'shash2',  #[options: "reg" or "shash2"]
            "hiddens": [25,25],
            "dropout_rate": 0.,
            "ridge_param": [10.0,0.0], 
            "learning_rate": 0.00001,  # reg->0.0001, shash2->.00005 or .00001
            "batch_size": 64,
            "rng_seed": 8889,
            "seed": None,
            "act_fun": ["relu","relu"],
            "n_epochs": 25_000,
            "patience": 50,
        },        
        "exp15C_126": { 
            "save_model": True,
            "n_models": 10,            # the number of networks to train
            "ssp" : "126",             #[options: '126' or '370']
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
            "hiddens": [25,25],
            "dropout_rate": 0.,
            "ridge_param": [10.0,0.0], 
            "learning_rate": 0.00001,  # reg->0.0001, shash2->.00005 or .00001
            "batch_size": 64,
            "rng_seed": 8889,
            "seed": None,
            "act_fun": ["relu","relu"],
            "n_epochs": 25_000,
            "patience": 50,
        },        
        "exp20C_126": { 
            "save_model": True,
            "n_models": 10,            # the number of networks to train
            "ssp" : "126",             #[options: '126' or '370']
            "gcmsub" : "ALL",          #[options: 'ALL' or 'UNIFORM'
            "obsdata" : "BEST",        #[options: 'BEST' or 'GISTEMP'
            "target_temp": 2.0,
            "n_train_val_test" : (7,2,1),
            "baseline_yr_bounds": (1850,1899),
            "training_yr_bounds": (1970,2100),
            "anomaly_yr_bounds": (1951,1980),
            "anomalies": True,         #[options: True or False]
            "remove_map_mean": False,  #[options: False or "weighted" or "raw"]

            "network_type": 'shash2',  #[options: "reg" or "shash2"]
            "hiddens": [25,25],
            "dropout_rate": 0.,
            "ridge_param": [10.0,0.0], 
            "learning_rate": 0.00001,  # reg->0.0001, shash2->.00005 or .00001
            "batch_size": 64,
            "rng_seed": 8889,
            "seed": None,
            "act_fun": ["relu","relu"],
            "n_epochs": 25_000,
            "patience": 50,
        },           
        
        #---------------------- HYPERPARAMETER TUNING ---------------------------
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
            "ridge_param": [0.,0.0], 
            "learning_rate": 0.00001,  # reg->0.0001, shash2->.00005 or .00001
            "batch_size": 64,
            "rng_seed": 8889,
            "seed": None,
            "act_fun": ["relu","relu"],
            "n_epochs": 25_000,
            "patience": 50,
        },           
        "exp1": {
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
            "ridge_param": [.1,0.0], 
            "learning_rate": 0.00001,  # reg->0.0001, shash2->.00005 or .00001
            "batch_size": 64,
            "rng_seed": 8889,
            "seed": None,
            "act_fun": ["relu","relu"],
            "n_epochs": 25_000,
            "patience": 50,
        },    
        "exp2": {
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
            "ridge_param": [1.,0.0], 
            "learning_rate": 0.00001,  # reg->0.0001, shash2->.00005 or .00001
            "batch_size": 64,
            "rng_seed": 8889,
            "seed": None,
            "act_fun": ["relu","relu"],
            "n_epochs": 25_000,
            "patience": 50,
        },    
        "exp3": {
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
            "ridge_param": [100.,0.0], 
            "learning_rate": 0.00001,  # reg->0.0001, shash2->.00005 or .00001
            "batch_size": 64,
            "rng_seed": 8889,
            "seed": None,
            "act_fun": ["relu","relu"],
            "n_epochs": 25_000,
            "patience": 50,
        },     
        "exp4": {
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
            "ridge_param": [5.,0.0], 
            "learning_rate": 0.00001,  # reg->0.0001, shash2->.00005 or .00001
            "batch_size": 64,
            "rng_seed": 8889,
            "seed": None,
            "act_fun": ["relu","relu"],
            "n_epochs": 25_000,
            "patience": 50,
        },       
        "exp5": {
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
            "ridge_param": [10.,0.0], 
            "learning_rate": 0.00001,  # reg->0.0001, shash2->.00005 or .00001
            "batch_size": 64,
            "rng_seed": 8889,
            "seed": None,
            "act_fun": ["relu","relu"],
            "n_epochs": 25_000,
            "patience": 50,
        },         
        "exp10": {
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
            "hiddens": [10],
            "dropout_rate": 0.,
            "ridge_param": [10.,0.0], 
            "learning_rate": 0.00001,  # reg->0.0001, shash2->.00005 or .00001
            "batch_size": 64,
            "rng_seed": 8889,
            "seed": None,
            "act_fun": ["relu"],
            "n_epochs": 25_000,
            "patience": 50,
        },   
        "exp11": {
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
            "hiddens": [5,5],
            "dropout_rate": 0.,
            "ridge_param": [10.,0.0], 
            "learning_rate": 0.00001,  # reg->0.0001, shash2->.00005 or .00001
            "batch_size": 64,
            "rng_seed": 8889,
            "seed": None,
            "act_fun": ["relu","relu"],
            "n_epochs": 25_000,
            "patience": 50,
        },  
        "exp12": {
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
            "hiddens": [25,25],
            "dropout_rate": 0.,
            "ridge_param": [10.,0.0], 
            "learning_rate": 0.00001,  # reg->0.0001, shash2->.00005 or .00001
            "batch_size": 64,
            "rng_seed": 8889,
            "seed": None,
            "act_fun": ["relu","relu"],
            "n_epochs": 25_000,
            "patience": 50,
        },        
        "exp13": {
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
            "hiddens": [2],
            "dropout_rate": 0.,
            "ridge_param": [10.,0.0], 
            "learning_rate": 0.00001,  # reg->0.0001, shash2->.00005 or .00001
            "batch_size": 64,
            "rng_seed": 8889,
            "seed": None,
            "act_fun": ["linear"],
            "n_epochs": 25_000,
            "patience": 50,
        },    
        "exp14": {
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
            "hiddens": [100,100],
            "dropout_rate": 0.,
            "ridge_param": [10.,0.0], 
            "learning_rate": 0.00001,  # reg->0.0001, shash2->.00005 or .00001
            "batch_size": 64,
            "rng_seed": 8889,
            "seed": None,
            "act_fun": ["relu","relu"],
            "n_epochs": 25_000,
            "patience": 50,
        },        
        
        "exp20": {
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
            "hiddens": [10],
            "dropout_rate": 0.,
            "ridge_param": [5.,0.0], 
            "learning_rate": 0.00001,  # reg->0.0001, shash2->.00005 or .00001
            "batch_size": 64,
            "rng_seed": 8889,
            "seed": None,
            "act_fun": ["relu"],
            "n_epochs": 25_000,
            "patience": 50,
        },   
        "exp21": {
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
            "hiddens": [5,5],
            "dropout_rate": 0.,
            "ridge_param": [5.,0.0], 
            "learning_rate": 0.00001,  # reg->0.0001, shash2->.00005 or .00001
            "batch_size": 64,
            "rng_seed": 8889,
            "seed": None,
            "act_fun": ["relu","relu"],
            "n_epochs": 25_000,
            "patience": 50,
        },  
        "exp22": {
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
            "hiddens": [25,25],
            "dropout_rate": 0.,
            "ridge_param": [5.,0.0], 
            "learning_rate": 0.00001,  # reg->0.0001, shash2->.00005 or .00001
            "batch_size": 64,
            "rng_seed": 8889,
            "seed": None,
            "act_fun": ["relu","relu"],
            "n_epochs": 25_000,
            "patience": 50,
        },        
        "exp23": {
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
            "hiddens": [2],
            "dropout_rate": 0.,
            "ridge_param": [5.,0.0], 
            "learning_rate": 0.00001,  # reg->0.0001, shash2->.00005 or .00001
            "batch_size": 64,
            "rng_seed": 8889,
            "seed": None,
            "act_fun": ["linear"],
            "n_epochs": 25_000,
            "patience": 50,
        },
        
        "exp30": {
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
            "hiddens": [25,25],
            "dropout_rate": 0.,
            "ridge_param": [0.,0.0], 
            "learning_rate": 0.00001,  # reg->0.0001, shash2->.00005 or .00001
            "batch_size": 64,
            "rng_seed": 8889,
            "seed": None,
            "act_fun": ["relu","relu"],
            "n_epochs": 25_000,
            "patience": 50,
        },        
        "exp31": {
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
            "hiddens": [25,25],
            "dropout_rate": 0.0,
            "ridge_param": [0.1,0.0], 
            "learning_rate": 0.00001,  # reg->0.0001, shash2->.00005 or .00001
            "batch_size": 64,
            "rng_seed": 8889,
            "seed": None,
            "act_fun": ["relu","relu"],
            "n_epochs": 25_000,
            "patience": 50,
        },        
        "exp32": {
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
            "hiddens": [25,25],
            "dropout_rate": 0.,
            "ridge_param": [1.,0.0], 
            "learning_rate": 0.00001,  # reg->0.0001, shash2->.00005 or .00001
            "batch_size": 64,
            "rng_seed": 8889,
            "seed": None,
            "act_fun": ["relu","relu"],
            "n_epochs": 25_000,
            "patience": 50,
        },  
        "exp33": {
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
            "hiddens": [25,25],
            "dropout_rate": 0.,
            "ridge_param": [100.,0.0], 
            "learning_rate": 0.00001,  # reg->0.0001, shash2->.00005 or .00001
            "batch_size": 64,
            "rng_seed": 8889,
            "seed": None,
            "act_fun": ["relu","relu"],
            "n_epochs": 25_000,
            "patience": 50,
        }, 
        
        "exp15C_370_uniform": { 
            "save_model": True,
            "n_models": 10,            # the number of networks to train
            "ssp" : "370",             #[options: '126' or '370']
            "gcmsub" : "UNIFORM",          #[options: 'ALL' or 'UNIFORM'
            "obsdata" : "BEST",        #[options: 'BEST' or 'GISTEMP'
            "target_temp": 1.5,
            "n_train_val_test" : (7,2,1),
            "baseline_yr_bounds": (1850,1899),
            "training_yr_bounds": (1970,2100),
            "anomaly_yr_bounds": (1951,1980),
            "anomalies": True,         #[options: True or False]
            "remove_map_mean": False,  #[options: False or "weighted" or "raw"]

            "network_type": 'shash2',  #[options: "reg" or "shash2"]
            "hiddens": [25,25],
            "dropout_rate": 0.,
            "ridge_param": [10.0,0.0], 
            "learning_rate": 0.00001,  # reg->0.0001, shash2->.00005 or .00001
            "batch_size": 64,
            "rng_seed": 8889,
            "seed": None,
            "act_fun": ["relu","relu"],
            "n_epochs": 25_000,
            "patience": 50,
        },    
          
        "exp20C_370_uniform": {
            "save_model": True,
            "n_models": 10,            # the number of networks to train
            "ssp" : "370",             #[options: '126' or '370']
            "gcmsub" : "UNIFORM",          #[options: 'ALL' or 'UNIFORM'
            "obsdata" : "BEST",        #[options: 'BEST' or 'GISTEMP'
            "target_temp": 2.0,
            "n_train_val_test" : (7,2,1),
            "baseline_yr_bounds": (1850,1899),
            "training_yr_bounds": (1970,2100),
            "anomaly_yr_bounds": (1951,1980),
            "anomalies": True,         #[options: True or False]
            "remove_map_mean": False,  #[options: False or "weighted" or "raw"]

            "network_type": 'shash2',  #[options: "reg" or "shash2"]
            "hiddens": [25,25],
            "dropout_rate": 0.,
            "ridge_param": [10.0,0.0], 
            "learning_rate": 0.00001,  # reg->0.0001, shash2->.00005 or .00001
            "batch_size": 64,
            "rng_seed": 8889,
            "seed": None,
            "act_fun": ["relu","relu"],
            "n_epochs": 25_000,
            "patience": 50,
        },        
        "exp15C_126_uniform": { 
            "save_model": True,
            "n_models": 10,            # the number of networks to train
            "ssp" : "126",             #[options: '126' or '370']
            "gcmsub" : "UNIFORM",          #[options: 'ALL' or 'UNIFORM'
            "obsdata" : "BEST",        #[options: 'BEST' or 'GISTEMP'
            "target_temp": 1.5,
            "n_train_val_test" : (7,2,1),
            "baseline_yr_bounds": (1850,1899),
            "training_yr_bounds": (1970,2100),
            "anomaly_yr_bounds": (1951,1980),
            "anomalies": True,         #[options: True or False]
            "remove_map_mean": False,  #[options: False or "weighted" or "raw"]

            "network_type": 'shash2',  #[options: "reg" or "shash2"]
            "hiddens": [25,25],
            "dropout_rate": 0.,
            "ridge_param": [10.0,0.0], 
            "learning_rate": 0.00001,  # reg->0.0001, shash2->.00005 or .00001
            "batch_size": 64,
            "rng_seed": 8889,
            "seed": None,
            "act_fun": ["relu","relu"],
            "n_epochs": 25_000,
            "patience": 50,
        },        
        "exp20C_126_uniform": { 
            "save_model": True,
            "n_models": 10,            # the number of networks to train
            "ssp" : "126",             #[options: '126' or '370']
            "gcmsub" : "UNIFORM",          #[options: 'ALL' or 'UNIFORM'
            "obsdata" : "BEST",        #[options: 'BEST' or 'GISTEMP'
            "target_temp": 2.0,
            "n_train_val_test" : (7,2,1),
            "baseline_yr_bounds": (1850,1899),
            "training_yr_bounds": (1970,2100),
            "anomaly_yr_bounds": (1951,1980),
            "anomalies": True,         #[options: True or False]
            "remove_map_mean": False,  #[options: False or "weighted" or "raw"]

            "network_type": 'shash2',  #[options: "reg" or "shash2"]
            "hiddens": [25,25],
            "dropout_rate": 0.,
            "ridge_param": [10.0,0.0], 
            "learning_rate": 0.00001,  # reg->0.0001, shash2->.00005 or .00001
            "batch_size": 64,
            "rng_seed": 8889,
            "seed": None,
            "act_fun": ["relu","relu"],
            "n_epochs": 25_000,
            "patience": 50,
        },         
        "exp20C_126_force": { 
            "save_model": True,
            "n_models": 10,            # the number of networks to train
            "ssp" : "126",             #[options: '126' or '370']
            "gcmsub" : "FORCE",          #[options: 'ALL' or 'UNIFORM'
            "obsdata" : "BEST",        #[options: 'BEST' or 'GISTEMP'
            "target_temp": 2.0,
            "n_train_val_test" : (7,2,1),
            "baseline_yr_bounds": (1850,1899),
            "training_yr_bounds": (1970,2100),
            "anomaly_yr_bounds": (1951,1980),
            "anomalies": True,         #[options: True or False]
            "remove_map_mean": False,  #[options: False or "weighted" or "raw"]

            "network_type": 'shash2',  #[options: "reg" or "shash2"]
            "hiddens": [25,25],
            "dropout_rate": 0.,
            "ridge_param": [10.0,0.0], 
            "learning_rate": 0.00001,  # reg->0.0001, shash2->.00005 or .00001
            "batch_size": 64,
            "rng_seed": 8889,
            "seed": None,
            "act_fun": ["relu","relu"],
            "n_epochs": 25_000,
            "patience": 50,
        },          
        "exp20C_126_extended": { 
            "save_model": True,
            "n_models": 10,            # the number of networks to train
            "ssp" : "126",             #[options: '126' or '370']
            "gcmsub" : "EXTEND",          #[options: 'ALL' or 'UNIFORM'
            "obsdata" : "BEST",        #[options: 'BEST' or 'GISTEMP'
            "target_temp": 2.0,
            "n_train_val_test" : (7,2,1),
            "baseline_yr_bounds": (1850,1899),
            "training_yr_bounds": (1970,2100),
            "anomaly_yr_bounds": (1951,1980),
            "anomalies": True,         #[options: True or False]
            "remove_map_mean": False,  #[options: False or "weighted" or "raw"]

            "network_type": 'shash2',  #[options: "reg" or "shash2"]
            "hiddens": [25,25],
            "dropout_rate": 0.,
            "ridge_param": [10.0,0.0], 
            "learning_rate": 0.00001,  # reg->0.0001, shash2->.00005 or .00001
            "batch_size": 64,
            "rng_seed": 8889,
            "seed": None,
            "act_fun": ["relu","relu"],
            "n_epochs": 25_000,
            "patience": 50,
        },     
        "exp20C_126_max": { 
            "save_model": True,
            "n_models": 10,            # the number of networks to train
            "ssp" : "126",             #[options: '126' or '370']
            "gcmsub" : "MAX",          #[options: 'ALL' or 'UNIFORM'
            "obsdata" : "BEST",        #[options: 'BEST' or 'GISTEMP'
            "target_temp": 2.0,
            "n_train_val_test" : (7,2,1),
            "baseline_yr_bounds": (1850,1899),
            "training_yr_bounds": (1970,2100),
            "anomaly_yr_bounds": (1951,1980),
            "anomalies": True,         #[options: True or False]
            "remove_map_mean": False,  #[options: False or "weighted" or "raw"]

            "network_type": 'shash2',  #[options: "reg" or "shash2"]
            "hiddens": [25,25],
            "dropout_rate": 0.,
            "ridge_param": [10.0,0.0], 
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

