import os

import flwr as fl
import tensorflow as tf
import zipfile
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

class FlowerClient(fl.client.NumPyClient):
   
    def __init__(self, client_id, model, training_dataset, testing_dataset):
        self.client_id = client_id
        self.model = model
        self.x_train, self.y_train = training_dataset
        self.x_test, self.y_test = testing_dataset
        self.model.compile(optimizer='Adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics= tf.keras.metrics.BinaryAccuracy())
    
    def get_parameters(self):  # type: ignore
            return self.model.get_weights()

    def fit(self, parameters, config):  # type: ignore
            self.model.set_weights(parameters)
            self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=32)
            return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):  # type: ignore
            self.model.set_weights(parameters)
            loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
            
            print(f"Client {self.client_id} - Evaluate on {len(self.x_train)} samples: Average loss: {loss:.4f}, Accuracy: {100*accuracy:.2f}%\n")
            
            return loss, len(self.x_test), {"accuracy": accuracy}
        
def ids_iot_2020_datasets():
    with zipfile.ZipFile("combinedcsvs.zip", "r") as zip_ref:
        zip_ref.extractall()
        
    dataset = pd.read_csv('combinedcsvs.csv')
    protocol = {'CLDAP':1,'DATA':2,'DHCP':3,'DNS':4,'DTLS':5,'DVB_SDT':6,'ECHO':7,'ISAKMP':8,'MDNS':9,'MP2T':0,'MPEG_PAT':11,'MPEG_PMT':12,'MQTT':13,'NAT-PMP':14,'NBNS':15,'NTP':16,'PORTMAP':17,'RADIUS':18,'RIP':19,'SNMP':20,'SRVLOC':21,'SSH':22,'TCP':23,'UDP':24,'XDMCP':25,'NFS':26}
    dataset.protocol = [protocol[item] for item in dataset.protocol]
    features = pd.DataFrame(dataset.iloc[:, 3:30].values)
    M = len(features.index)
    N = len(features.columns)
    ran = pd.DataFrame(np.random.randn(M,N), columns=features.columns, index=features.index)
    features.update(ran, overwrite = False)
    scaler = preprocessing.MinMaxScaler()
    X = scaler.fit_transform(features)
    my_imputer = SimpleImputer(missing_values = np.nan,
                            strategy ='mean')
    X = pd.DataFrame(my_imputer.fit_transform(X))
    y = dataset.iloc[:, 30].values
    X_training, X_testing, y_training, y_testing = train_test_split(X, y, test_size = 0.05)
    return (X_training, y_training), (X_testing, y_testing), X.shape[1]