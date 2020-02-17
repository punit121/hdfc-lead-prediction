import os
import boto3
import re
import pandas as pd
import numpy as np
from sklearn import preprocessing


#function for encoding the following columns of dataframe hdfc using label encoders, Alternate for One-Hot-Encoding
def label_encoder_hdfc(df_hdfc):
    label_encoder = preprocessing.LabelEncoder()

    df_hdfc[ALL_EVENTS_MAX_PROPERTY] = label_encoder.fit_transform((df_hdfc[ALL_EVENTS_MAX_PROPERTY]))

    label_encoder = preprocessing.LabelEncoder()
    df_hdfc[ALL_EVENTS_MAX_OCCURED_CITY_NAME] = label_encoder.fit_transform(
        (df_hdfc[ALL_EVENTS_MAX_OCCURED_CITY_NAME].astype('str')))

    label_encoder = preprocessing.LabelEncoder()
    df_hdfc[ALL_EVENTS_MAX_OCCURED_SUBCAT_NAME] = label_encoder.fit_transform(
        (df_hdfc[ALL_EVENTS_MAX_OCCURED_SUBCAT_NAME].astype('str')))

    label_encoder = preprocessing.LabelEncoder()
    df_hdfc[ALL_EVENTS_MAX_SOURCE] = label_encoder.fit_transform((df_hdfc[ALL_EVENTS_MAX_SOURCE].astype('str')))

    label_encoder = preprocessing.LabelEncoder()
    df_hdfc[ALL_EVENTS_UTM_TRAFFIC_SOURCE] = label_encoder.fit_transform(
        (df_hdfc[ALL_EVENTS_UTM_TRAFFIC_SOURCE].astype('str')))

    label_encoder = preprocessing.LabelEncoder()
    df_hdfc[RESPONSE_EVENTS_MAX_PROPERTY] = label_encoder.fit_transform(
        (df_hdfc[RESPONSE_EVENTS_MAX_PROPERTY].astype('str')))

    label_encoder = preprocessing.LabelEncoder()
    df_hdfc[RESPONSE_EVENTS_MAX_OCCURED_CITY_NAME] = label_encoder.fit_transform(
        (df_hdfc[RESPONSE_EVENTS_MAX_OCCURED_CITY_NAME].astype('str')))

    label_encoder = preprocessing.LabelEncoder()
    df_hdfc[RESPONSE_EVENTS_MAX_OCCURED_SUBCAT_NAME] = label_encoder.fit_transform(
        (df_hdfc[RESPONSE_EVENTS_MAX_OCCURED_SUBCAT_NAME].astype('str')))

    label_encoder = preprocessing.LabelEncoder()
    df_hdfc[RESPONSE_EVENTS_MAX_SOURCE] = label_encoder.fit_transform(
        (df_hdfc[RESPONSE_EVENTS_MAX_SOURCE].astype('str')))

    label_encoder = preprocessing.LabelEncoder()
    df_hdfc[RESPONSE_EVENTS_UTM_TRAFFIC_SOURCE] = label_encoder.fit_transform(
        (df_hdfc[RESPONSE_EVENTS_UTM_TRAFFIC_SOURCE].astype('str')))
    
    return df_hdfc
