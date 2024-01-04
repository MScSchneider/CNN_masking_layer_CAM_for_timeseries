# Code snippets for CNN with masking layer

import numpy as np
import pandas as pd
import datetime
import time
import sys
import matplotlib.pyplot as plt
import os
from pathlib import Path

from sklearn.utils import class_weight
from mpl_toolkits.axes_grid1 import make_axes_locatable

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.ops import disable_eager_execution

from matplotlib.font_manager import FontProperties
import csv

physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
i = 0
print('Appliing Memory Groth Hack to selected GPUs ....')
for i in range(len(physical_devices)):
    #if(i in useGPUs):
        tf.config.experimental.set_memory_growth(physical_devices[i], True)
        print('   GPU number {} : {}'.format(i, tf.config.experimental.get_memory_growth(physical_devices[i])))
    #i = i+1
print('success')

logical_devices = tf.config.list_logical_devices('GPU')
for dev in logical_devices:
    print('   Found GPU: ', dev.name)
print('success')

tf.__version__
#2.6.0

outPath = Path.cwd() / 'cnn_masking'
maskVal = -1000000
nb_classes = 2

def build_model(X_train1, maskVal_, nb_classes_, CNN_path, model_png_exp):
    print("----------Generiere CNN Model-------")
    #print( X_train.reshape(-1).shape )
    l_1 = tf.keras.layers.Input( shape= ( X_train1.shape[1], X_train1.shape[2] ) ) #,name='L_1'
    l_2 = tf.keras.layers.Masking( mask_value=maskVal_ )( l_1 )
    #l_3 = tfa.layers.ESN(units=5, return_sequences=True)(l_2)
    l_3 = tf.keras.layers.Conv1D(filters= 64, kernel_size= 8192, padding='same', kernel_initializer='he_uniform')( l_2 )
    l_4 = tf.keras.layers.Activation(activation='relu')(l_3)
    #l_5 = tf.keras.layers.GlobalAveragePooling1D()(l_4)
    l_5 = tf.keras.layers.Conv1D(filters= 128, kernel_size= 8, padding='same', kernel_initializer='he_uniform')(l_4)
    l_6 = tf.keras.layers.Activation(activation='relu')(l_5)
    l_7 = tf.keras.layers.GlobalAveragePooling1D()(l_6)
    l_8 = tf.keras.layers.Dense(units = nb_classes_, activation='softmax')(l_7)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model_ = tf.keras.models.Model( inputs = l_1, outputs = [l_8] )
    model_.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model_.summary()
    if model_png_exp == True:
        tf.keras.utils.plot_model(model_, to_file=CNN_path + os.sep + 'Modelstruktur_CNN.png', dpi=400, show_shapes = True, show_layer_names = True)
    model_.save(CNN_path + os.sep + str(model_.name))
    return model_

model_def = build_model(X_train_unbal, maskVal, nb_classes, str(outPath), True)
Modelname = model_def.name
Modelname

def model_copy_func(Modelname_, outPath_):
    model_bal_ = keras.models.load_model( str(outPath_) + os.sep + Modelname_ )
    model_bal_._name = Modelname_ + "_bal"
    Modelname_bal = model_bal_.name
    print("modelname: ",Modelname_bal)

    model_bal_S_ = keras.models.load_model( str(outPath_) + os.sep + Modelname_ )
    model_bal_S_._name = Modelname_ + "_bal_S"
    Modelname_bal_S = model_bal_S_.name
    print("modelname: ",Modelname_bal_S)

    model_unbal_ = keras.models.load_model( str(outPath_) + os.sep + Modelname_ )
    model_unbal_._name = Modelname_ + "_unbal"
    Modelname_unbal = model_unbal_.name
    print("modelname: ",Modelname_unbal)

    model_unbal_S_ = keras.models.load_model( str(outPath_) + os.sep + Modelname_ )
    model_unbal_S_._name = Modelname_ + "_unbal_S"
    Modelname_unbal_S = model_unbal_S_.name
    print("modelname: ",Modelname_unbal_S)
    return model_bal_, model_bal_S_, model_unbal_, model_unbal_S_

def dir_generator(CNN_path,mo_name,mo_name1,mo_name2,mo_name3,mo_name4):
    dir_models = CNN_path + os.sep + mo_name + os.sep + 'models'
    dir_figures = CNN_path + os.sep + mo_name + os.sep + 'CNN_figures'
    dir_CAM_figures = CNN_path + os.sep + mo_name + os.sep + 'CAM_figures'
    
    dir_best1 = CNN_path + os.sep + mo_name + os.sep + 'best_' + mo_name1
    dir_Log1 = CNN_path + os.sep + mo_name + os.sep + 'log_' + mo_name1
    dir_best2 = CNN_path + os.sep + mo_name + os.sep + 'best_' + mo_name2
    dir_Log2 = CNN_path + os.sep + mo_name + os.sep + 'log_' + mo_name2
    dir_best3 = CNN_path + os.sep + mo_name + os.sep + 'best_' + mo_name3
    dir_Log3 = CNN_path + os.sep + mo_name + os.sep + 'log_' + mo_name3
    dir_best4 = CNN_path + os.sep + mo_name + os.sep + 'best_' + mo_name4
    dir_Log4 = CNN_path + os.sep + mo_name + os.sep + 'log_' + mo_name4
    
    Path(dir_models).mkdir(parents=True, exist_ok=True)
    Path(dir_figures).mkdir(parents=True, exist_ok=True)
    Path(dir_CAM_figures).mkdir(parents=True, exist_ok=True)
    
    Path(dir_best1).mkdir(parents=True, exist_ok=True)
    Path(dir_Log1).mkdir(parents=True, exist_ok=True)
    Path(dir_best2).mkdir(parents=True, exist_ok=True)
    Path(dir_Log2).mkdir(parents=True, exist_ok=True)
    Path(dir_best3).mkdir(parents=True, exist_ok=True)
    Path(dir_Log3).mkdir(parents=True, exist_ok=True)
    Path(dir_best4).mkdir(parents=True, exist_ok=True)
    Path(dir_Log4).mkdir(parents=True, exist_ok=True)

    print("dir: " + dir_models + " created!")
    print("dir: " + dir_figures + " created!")
    print("dir: " + dir_CAM_figures + " created!")
    print("dir: " + dir_best1 + " created!")
    print("dir: " + dir_Log1 + " created!")
    print("dir: " + dir_best2 + " created!")
    print("dir: " + dir_Log2 + " created!")
    print("dir: " + dir_best3 + " created!")
    print("dir: " + dir_Log3 + " created!")
    print("dir: " + dir_best4 + " created!")
    print("dir: " + dir_Log4 + " created!")
    return dir_models,dir_figures,dir_CAM_figures,dir_best1,dir_Log1,dir_best2,dir_Log2,dir_best3,dir_Log3,dir_best4,dir_Log4

dir,dir_figures,dir_CAM_figures,dir_best_bal,dir_Log_bal,dir_best_bal_S,dir_Log_bal_S,dir_best_unbal,dir_Log_unbal,dir_best_unbal_S,dir_Log_unbal_S = dir_generator(str(outPath),Modelname,model_bal._name,model_bal_S._name,model_unbal._name,model_unbal_S._name)

def CNN_fit_func(model_, X_train_, Y_train_, dir_, dir_best_, dir_Log_, Modelname_):
    class TimeHistory(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.times = []

        def on_epoch_begin(self, epoch, logs={}):
            self.epoch_time_start = time.time()

        def on_epoch_end(self, epoch, logs={}):
            self.times.append(time.time() - self.epoch_time_start)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model_.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    batch_size = 32
    nb_epochs = 100
    
    c_mcp_save_all = tf.keras.callbacks.ModelCheckpoint( str(dir_) + os.sep + 'model_{epoch:03d}.hdf5', 
                                                                save_best_only = False, 
                                                                save_weights_only = False, 
                                                                monitor = 'val_bin_classification_layer_cohen_kappa', 
                                                                mode = 'max',
                                                                verbose=1)
    model_best = 'CNN_Klas_Manuel_best' + "_" + str(Modelname_) +'.hdf5'
    c_mcp_save_best = tf.keras.callbacks.ModelCheckpoint( str(dir_best_) + os.sep + model_best, save_best_only=True, monitor='loss', mode='min')

    c_csvLogger = tf.keras.callbacks.CSVLogger(str(dir_Log_) + os.sep + str(Modelname_) +'log.csv', separator=",", append=True)

    time_callback = TimeHistory()
    c_tensorboard = tf.keras.callbacks.TensorBoard(log_dir = dir_Log_,histogram_freq=1)
    
    hist = model_.fit(
            x=X_train_, # / all_max normierung auf 1
            y=Y_train_,
            batch_size=batch_size,
            epochs=nb_epochs,
            verbose='auto',
            callbacks=[c_mcp_save_best, c_mcp_save_all,c_csvLogger,time_callback,c_tensorboard],
            validation_split=0.3,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_batch_size=None, #20
            #validation_freq=1,
            #max_queue_size=10,
            #workers=1,
            use_multiprocessing=True
    )

    print(len(time_callback.times))
    timelist = np.array(time_callback.times)

    with open(str(dir_Log_) +os.sep + str(Modelname_) + 'timelog.csv', 'w', newline='\n') as f:
            writer = csv.writer(f)#, delimiter=','
            writer.writerow(timelist)
            
    return hist,model_

hist_bal, model_bal = CNN_fit_func(model_bal, X_train_bal, Y_train_bal, dir, dir_best_bal, dir_Log_bal, model_bal.name)

def partial_func(model_):
    partial_model_ = tf.keras.Model(model_.inputs, [model_.layers[-4].output,
                                                      model_.layers[-3].output,
                                                      model_.layers[-2].output,
                                                      model_.layers[-1].output])
    return partial_model_

val_cam_output = partial_model( X_train_bal[0:1,:,:], training=False) #x_val_inputs
print( np.array(val_cam_output[0]) )
print( val_cam_output )

[cnn2, relu2, gap, dense] = val_cam_output
dense.shape, gap.shape, relu2.shape, relu2.shape

def relu_plot(relu2_, name, best_model, dir_figures_,best_model_):
    plt.close(7)
    plt.figure(7, figsize=(6, 7), dpi= 100, facecolor='w', edgecolor='k')
    plt.plot(relu2_[0,:,0])
    plt.plot(relu2_[0,:,1])
    plt.title("relu2 good and bad (" + str(name) +")", fontsize = 16)
    fname = "CAM_aktivation_test_" + name
    if best_model:
        fname = fname + "_best"
        plt.savefig(str(dir_figures_) +os.sep+ str(fname) + ".pdf", dpi='figure', format='pdf')
    else:
        plt.savefig(str(dir_figures_) +os.sep+ str(fname) + ".pdf", dpi='figure', format='pdf')

    if not plt.rcParams["text.usetex"]:
        plt.savefig(str(dir_figures_) +os.sep+ str(fname) + ".svg", dpi=150)

def get_wb(model_):
    class_weights_ = model_.layers[-1].get_weights()[0]
    class_bias_ = model_.layers[-1].get_weights()[1]
    print("class_weights_.shape, class_bias_.shape: ", class_weights_.shape, class_bias_.shape)
    return class_weights_,class_bias_

def get_sep(model_, relu2_):
    class_weights,class_bias = get_wb(model_)
    act_good_sep_ = relu2_ * class_weights[:,0] + class_bias[0]
    act_bad_sep_ = relu2_ * class_weights[:,1] + class_bias[1]
    print("act_good_sep_.shape, act_bad_sep_.shape: ", act_good_sep_.shape, act_bad_sep_.shape )
    return act_good_sep_,act_bad_sep_

def get_sep_sum(model_, relu2_):
    act_good_sep_, act_bad_sep_ = get_sep(model_, relu2_)
    act_good_series_ = np.sum(act_good_sep_, axis=-1)
    act_bad_series_ = np.sum(act_bad_sep_, axis=-1)
    print("act_good_series_.shape, act_bad_series_.shape: ", act_good_series_.shape, act_bad_series_.shape )
    return act_good_series_,act_bad_series_

def get_CAM(model_, val_cam_output_):
    last_conv = val_cam_output_
    gap_weight = model_.layers[-1].get_weights()
    print(last_conv[0].shape)
    print(gap_weight[0].shape)
    CAM_ = np.dot(last_conv[0], gap_weight[0])
    return CAM_

def CAM_plot3(model_, X_test_, Y_test_bi_, Y_test_keys_, val_cam_output_, best_model, cam_signal, dir_figures_):
    plt.rc('text', usetex=False)
    plt.rc('font', family='serif')
    
    CAM = get_CAM(model_, val_cam_output_)
    print(len(X_test_))
    CAM = (CAM - CAM.min(axis=1, keepdims=True)) / (CAM.max(axis=1, keepdims=True) - CAM.min(axis=1, keepdims=True))
    print(CAM.shape)
    print("best status: ",best_model)
    k = cam_signal # Nr. von Test 40,33

    if k == k:
        maske = np.argmin(X_test_[k,200:])
        maske = maske + 200
        print("maske: ",maske)
        max1 = np.nanmax(X_test_[k,:]) + np.nanmax(X_test_[k,:])/10
        print("max1: ",max1 )
        
        min1 = -1000
        print("min1: ",min1 )
        print(np.arange(0, len(X_test_[k,0:maske])))
        print(X_test_[k,0:maske].squeeze())
        plt.close(8)
        fig, ax = plt.subplots(3, 1, figsize=(9, 9), sharex = True, gridspec_kw={'hspace': 0}, num=8)
        
        cam_pk1 = ax[0].scatter(np.arange(0, len(X_test_[k,200:maske])), 
                                (np.cumsum(X_test_[k,200:maske].squeeze() - np.mean(X_test_[k,200:maske])) )/np.mean(X_test_[k,200:maske]), 
                                c = CAM[0, :, int(Y_test_bi_[k])][:len(X_test_[k,200:maske])].squeeze(), 
                                cmap='rainbow', 
                                s=10, 
                                linewidths = 0.5, 
                                edgecolors = None)
        
        cam_pk1 = ax[1].scatter(np.arange(0, len(X_test_[k,200:maske])), 
                                X_test_[k,200:maske].squeeze(), 
                                c = CAM[0, :, int(Y_test_bi_[k])][:len(X_test_[k,200:maske])].squeeze(), 
                                cmap='rainbow', 
                                s=10, 
                                linewidths = 0.5, 
                                edgecolors = None)
        
        cam_cnn = ax[2].scatter(np.arange(0, len(X_test_[k,200:maske])), 
                                CAM[0, :, int(Y_test_bi_[k])][:len(X_test_[k,200:maske])].squeeze(), 
                                c = CAM[0, :, int(Y_test_bi_[k])][:len(X_test_[k,200:maske])].squeeze(), 
                                cmap='rainbow', 
                                s = 8, 
                                linewidths = 0.5, 
                                edgecolors = None)

        shotKey_kwargs = dict(fontsize='large', color='black', backgroundcolor='white', weight='normal') #fontsize='xx-large'
        ax[0].tick_params(labelsize = 14)
        ax[0].set_ylabel('pressure (bar)', fontsize = 14)
        ax[0].set_title("CAM (" + str(model_.name) +")" + ' shotKey: {}'.format(Y_test_keys_[k])+ ', class: {}'.format(Y_test[k]), fontsize = 16)
        ax[0].set_xlim(0,maske)
        
        ax[1].set_xlim(0,maske)
        ax[1].set_ylim(bottom = min1, top = max1)
            
        ax[1].tick_params(labelsize = 14)
        
        ax[2].set_xlim(0,maske)
        ax[2].set_xticks(range(0, maske, 2500))
        ax[2].set_xticklabels(np.arange(0,maske,2500.0)/1000.0)
        ax[2].set_xlabel('time (s)', fontsize = 14)
        ax[2].set_ylabel("Activation CAM", fontsize = 14)

        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.subplots_adjust(bottom = 0.1, right = 0.8, top = 0.95)

        cax = plt.axes([0.815, 0.125, 0.015, 0.8])
        cbar = plt.colorbar(cam_pk1, cax=cax)
        cbar.ax.tick_params(labelsize = 14)
        cbar.ax.get_yaxis().labelpad = 25
        cbar.set_label("Activation CAM", rotation = 90, fontsize = 14)

        fname = "CAM_act_series_" + str(Y_test_keys_[k]) + "_" + str(model_.name)
        if k != k:
            plt.savefig(str(dir_figures_) +os.sep+ str(fname) +".pdf", dpi='figure', format='pdf')
        else:
            if best_model:
                fname = fname + "_best"
                plt.savefig(str(dir_figures_) +os.sep+ str(fname) +".pdf", dpi='figure', format='pdf')
            else:
                plt.savefig(str(dir_figures_) +os.sep+ str(fname) +".pdf", dpi='figure', format='pdf')
    return CAM

cam_signal = 21 # test timeseries from input dataset
CAM = CAM_plot3(model, X_test, Y_test_bi, Y_test_keys, val_cam_output, best_model, cam_signal, dir_CAM_figures)
