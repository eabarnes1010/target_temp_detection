"""Network functions.

Classes
---------
Exponentiate(keras.layers.Layer)


Functions
---------
RegressLossExpSigma(y_true, y_pred)
compile_model(x_train, y_train, settings)


"""
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Dropout, Softmax
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow import keras
import numpy as np

import custom_metrics


class Exponentiate(keras.layers.Layer):
    """Custom layer to exp the sigma and tau estimates inline."""

    def __init__(self, **kwargs):
        super(Exponentiate, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.math.exp(inputs)

def RegressLossExpSigma(y_true, y_pred):    
    # network predictions
    mu = y_pred[:,0]
    sigma = y_pred[:,1]
    
    # normal distribution defined by N(mu,sigma)
    norm_dist = tfp.distributions.Normal(mu,sigma)

    # compute the log as the -log(p)
    loss = -norm_dist.log_prob(y_true[:,0])    

    return tf.reduce_mean(loss, axis=-1)    

def compile_model(x_train, y_train, settings):

    # First we start with an input layer
    inputs = Input(shape=x_train.shape[1:]) 

    normalizer = tf.keras.layers.Normalization()
    normalizer.adapt(x_train)
    layers = normalizer(inputs)

    layers = Dropout(rate=settings["dropout_rate"],
                     seed=settings["seed"])(layers) 
    
    for hidden, activation, ridge in zip(settings["hiddens"], settings["act_fun"], settings["ridge_param"]):
        layers = Dense(hidden, activation=activation,
                       kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.00, l2=ridge),
                       bias_initializer=tf.keras.initializers.RandomNormal(seed=settings["seed"]),
                       kernel_initializer=tf.keras.initializers.RandomNormal(seed=settings["seed"]))(layers)


    if settings['network_type'] == 'reg':
        LOSS = 'mae'
        metrics = ['mse',]
        
        output_layer = Dense(1, activation='linear',
                          bias_initializer=tf.keras.initializers.RandomNormal(seed=settings["seed"]),
                          kernel_initializer=tf.keras.initializers.RandomNormal(seed=settings["seed"]))(layers)
        
    elif settings['network_type'] == 'shash2':
        LOSS = RegressLossExpSigma
        metrics = [
                    custom_metrics.CustomMAE(name="custom_mae"),
                    custom_metrics.InterquartileCapture(name="interquartile_capture"),
                    custom_metrics.SignTest(name="sign_test"),
                  ]

        y_avg = np.mean(y_train)
        y_std = np.std(y_train)

        mu_z_unit = tf.keras.layers.Dense(
            units=1,
            activation="linear",
            use_bias=True,
            bias_initializer=tf.keras.initializers.RandomNormal(seed=settings["seed"]+100),
            kernel_initializer=tf.keras.initializers.RandomNormal(seed=settings["seed"]+100),
            name="mu_z_unit",
        )(layers)
        
        mu_unit = tf.keras.layers.Rescaling(
            scale=y_std,
            offset=y_avg,
            name="mu_unit",
        )(mu_z_unit)
        
        # sigma_unit. The network predicts the log of the scaled sigma_z, then
        # the resclaing layer scales it up to log of sigma y, and the custom
        # Exponentiate layer converts it to sigma_y.
        log_sigma_z_unit = tf.keras.layers.Dense(
            units=1,
            activation="linear",
            use_bias=True,
            bias_initializer=tf.keras.initializers.Zeros(),
            kernel_initializer=tf.keras.initializers.Zeros(),
            name="log_sigma_z_unit",
        )(layers)

        log_sigma_unit = tf.keras.layers.Rescaling(
            scale=1.0,
            offset=np.log(y_std),
            name="log_sigma_unit",
        )(log_sigma_z_unit)

        sigma_unit = Exponentiate(
            name="sigma_unit",
        )(log_sigma_unit)
        
        output_layer = tf.keras.layers.concatenate([mu_unit, sigma_unit], axis=1)
        
    else:
        raise NotImpletementedError('no such network_type')
        
    # Constructing the model
    model = Model(inputs, output_layer)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=settings["learning_rate"]), 
                  loss=LOSS, 
                  metrics=metrics,
                 )
        
        
    model.summary()
    
    return model
