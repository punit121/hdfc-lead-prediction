import os,sys
import numpy as np
import pandas as pd
import json
from constants.constant import *
from constants.all_dict import *

#function to get the value from a dict corresponding to a key traffic_source
def get_utm_traffic_source(traffic_source):
    
    traffic_source = traffic_source.lower()
    for key, value in utm_traffic_source_dict.items():
        if key == traffic_source:
            return value

#function to get the value from a dict corresponding to a key city_name
def get_city_name(city_name):
    
    city_name = city_name.lower()
    for key, value in city_name_dict.items():
        if key == city_name:
            return value

#Function to get the response dict
def get_response_dict(response):
    for key, value in response_dict.items():
        inner_dict = value
        for inner_key, inner_value in inner_dict.items():
            inner_dict[inner_key] = response
    return response_dict

#Utility to get a dataframe from a given dict
def get_dataframe_from_dict(json_load):

    for key, value in processed_data_default_dict:
        if key not in json_load:
            json_load[key] = processed_data_default_dict[key]
        else:
            json_load[key] = [np.asarray(value)]

    data = pd.DataFrame.from_dict(json_dict, orient='columns')
    
    return data

def get_hyperparameters():
    return hyperparameter_dict

def fill_null_values_dataframe(df_hdfc):
    
    df_hdfc[ALL_EVENTS_MAX_PROPERTY].fillna(EMPTY_EVENTS_VALUE, inplace=True)
    df_hdfc[ALL_EVENTS_MAX_OCCURED_CITY_NAME].fillna(EMPTY_EVENTS_VALUE, inplace=True)
    df_hdfc[ALL_EVENTS_MAX_OCCURED_SUBCAT_NAME].fillna(EMPTY_EVENTS_VALUE, inplace=True)
    df_hdfc[ALL_EVENTS_MAX_SOURCE].fillna(EMPTY_EVENTS_VALUE, inplace=True)
    df_hdfc[ALL_EVENTS_UTM_TRAFFIC_SOURCE].fillna(EMPTY_EVENTS_VALUE, inplace=True)
    df_hdfc[RESPONSE_EVENTS_MAX_PROPERTY].fillna(EMPTY_EVENTS_VALUE, inplace=True)
    df_hdfc[RESPONSE_EVENTS_MAX_OCCURED_CITY_NAME].fillna(EMPTY_EVENTS_VALUE, inplace=True)
    df_hdfc[RESPONSE_EVENTS_MAX_OCCURED_SUBCAT_NAME].fillna(EMPTY_EVENTS_VALUE, inplace=True)
    df_hdfc[RESPONSE_EVENTS_MAX_SOURCE].fillna(EMPTY_EVENTS_VALUE, inplace=True)
    df_hdfc[RESPONSE_EVENTS_UTM_TRAFFIC_SOURCE].fillna(EMPTY_EVENTS_VALUE, inplace=True)
    
    return df_hdfc