from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import os
import pandas as pd
import CroissantModel as cm
from sklearn.preprocessing import MinMaxScaler
import matplotlib as mpl
import matplotlib.pyplot as plt

def create_time_steps(length):
  return list(range(-length, 0))

def show_plot(plot_data, latent, title):
  labels = ['History', 'True Future', 'Model Prediction']
  marker = ['.-', 'rx', 'go']
 # time_steps = create_time_steps(plot_data[0].shape[0])
  
  # data= plot_data.reshape((1,-1))
  # latent = latent.reshape((1,256))

  plt.title(title)
  t= np.arange(0,256)
  plt.subplot(211)
  plt.plot( t , np.asarray(data).flatten(), 'o')
  plt.subplot(212)
  plt.plot(t, np.asarray(latent).flatten(), 'bs', label="2" )

  # , markersize=int(latent * 10),)
    # else:
    #   plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
  plt.legend()
  
  
  return plt

def univariate_data(dataset, start_index, end_index, history_size, target_size):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i)
    # Reshape data from (history_size,) to (history_size, 1)
    data.append(np.reshape(dataset[indices], (1, history_size)))
    # labels.append(dataset[i+target_size])

  return np.array(data), np.array(data)

def create_time_steps(length):
  return list(range(-length, 0))

def transform_data_to_windows_multivariate(data, window_size=30,val_perc=0,window_shift=1):
    scaler_dict = {}
    train_x = []
    val_x = []

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler_name="scaler"
    min_ts_cuif = len(data)
    data = data.reshape(-1, 1)
    scaler = scaler.fit(data)
    normalized = scaler.transform(data)
    if len(normalized) < window_size:
        raise Exception("Interface {} has a time series shorter than the windows size".format(
            self.data.theTSs.values()[if_idx]))
    threshold = np.floor(((len(normalized) - 1) / window_shift) * val_perc)
    
    for i in range(0, min_ts_cuif - window_size - 1, window_shift):
        x = normalized[i:(i + window_size), :]
        if (i < threshold):
            val_x.append(x)
        else:
            train_x.append(x)
    scaler_dict[scaler_name] = scaler
    return np.array(train_x), np.array(val_x), scaler_dict

if __name__ == "__main__":
    
    zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)

    csv_path, _ = os.path.splitext(zip_path)

    df = pd.read_csv(csv_path)

    uni_data = df['T (degC)']
    uni_data.index = df['Date Time']    

    
    TRAIN_SPLIT = 300000
    EPOCHS=15
    EVALUATION_INTERVAL = 200
    WINDOW_SIZE = 30

    uni_data = uni_data.values
   

    train_data, val_data, scaler_dict = transform_data_to_windows_multivariate(uni_data,WINDOW_SIZE,0.15)

    BATCH_SIZE = 256
    BUFFER_SIZE = 10000

    train_univariate = tf.data.Dataset.from_tensor_slices((train_data, train_data))
    train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_univariate = tf.data.Dataset.from_tensor_slices((val_data, val_data))
    val_univariate = val_univariate.batch(BATCH_SIZE).repeat()

    croissant = cm.CroissantModel(
            input_dim= WINDOW_SIZE,
            timesteps = 1,
            batch_size = BATCH_SIZE,
            intermediate_dim = 64,
            latent_dim= 32,
            epsilon_std= 1.,
            learning_rate= 0.001
    )

    vae,enc,dec = croissant.generate_model()
    print("Summary")
    print(vae.summary())


    history = vae.fit( train_univariate,
                        epochs=EPOCHS,
                        steps_per_epoch=EVALUATION_INTERVAL,
                        validation_data=val_univariate, 
                        validation_steps=50)
    




    data = [] 
    latent =[]
    for x, y in val_univariate.take(1):
        for value in x:
            data.append(value[15].numpy())
            latent.append(np.mean(enc.predict(np.expand_dims(value,0))))
    plot = show_plot(np.asarray(np.concatenate(data)), np.asarray(latent), 'Simple LSTM model')
    plot.show()
