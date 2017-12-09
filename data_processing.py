import numpy as np
import pandas as pd

def read_csv(filename):
    """Reads the data from the csv file and returns the latitude, longitude and fault"""
    data = pd.read_csv(filename)
    latitude = data.iloc[:,2]
    longitude = data.iloc[:,3]
    fault = data.iloc[:,-1]
    return latitude, longitude, fault

def transform_coordinates(latitude, longitude):
    """Transforms the latitude and longitude values into Earth-centered, Earth-fixed coordinates (x,y,z)"""
    radius = 6371
    x = radius * np.cos(latitude * np.pi/180) * np.cos(longitude * np.pi/180)
    y = radius * np.cos(latitude * np.pi/180) * np.sin(longitude * np.pi/180)
    z = radius * np.sin(latitude * np.pi/180)
    return x, y, z	

def preprocess_data(x, y, z):
    """Create a matrix with num_row rows and 3 columns for x, y, z"""
    num_row= x.shape[0]
    X= np.empty((num_row,3))
    for i in range(num_row):
        X[i,]= [x[i], y[i], z[i]]
    return X