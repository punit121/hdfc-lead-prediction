#!/usr/bin/env python
# coding: utf-8

import os
import boto3
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sagemaker
from sagemaker import get_execution_role
from sagemaker.predictor import csv_serializer
from constants.constant import *
from encoders.label_encoder import *
from helpers.helper import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sagemaker.xgboost.estimator import XGBoost


def train_and_deploy_hdfc():
    
    role = get_execution_role()
    
    region = boto3.Session().region_name
    
    bucket=S3_BUCKET_PATH
    
    prefix = S3_PREFIX
    
    df_hdfc=pd.read_csv(HDFC_DATA_PATH)
    
    df_hdfc=fill_null_values_dataframe(df_hdfc)

    df_hdfc=label_encoder_hdfc(df_hdfc)
    
    np.where(pd.isnull(df_hdfc))
    
    df_hdfc.fillna(df_hdfc.mean(), inplace=True)
    
    df_hdfc_final=df_hdfc[[TRAIN_OUTPUT_VALUE,RESPONSE_EVENTS_PROPERTIES_RESPONDED,RESPONSE_EVENTS_SALE_SERIOUSNESS_FACTOR,RESPONSE_EVENTS_MAX_PRICE,RESPONSE_EVENTS_UTM_TRAFFIC_SOURCE,RESPONSE_EVENTS_SALE_RESPONSES,RESPONSE_EVENTS_MAX_OCCURED_CITY_NAME,ALL_EVENTS_SERIOUSNESS_FACTOR,ALL_EVENTS_LOCALITIES_COUNT,ALL_EVENTS_MAX_OCCURED_CITY_NAME,ALL_EVENTS_PROPERTIES_VIEWED,ALL_EVENTS_MAX_AREA_SQ_FEET,ALL_EVENTS_MAX_PRICE]]
    
    train_data, validation_data, test_data = np.split(df_hdfc_final.sample(frac=1, random_state=1729), [int(0.7 * len(df_hdfc_final)), int(0.9 * len(df_hdfc_final))])
    train_data.to_csv(HDFC_TRAIN_DATA, header=False, index=False)
    validation_data.to_csv(HDFC_VALIDATION_DATA, header=False, index=False)

    s3_input_train = boto3.Session().resource(S3_RESOURCE).Bucket(bucket).Object(os.path.join(prefix,HDFC_TRAIN_UPLOAD_PATH)).upload_file(HDFC_TRAIN_DATA)
    s3_input_validation = boto3.Session().resource(S3_RESOURCE).Bucket(bucket).Object(os.path.join(prefix,HDFC_VALIDATION_UPLOAD_PATH)).upload_file(HDFC_VALIDATION_DATA)
    

    sess = sagemaker.Session()
    script_path= SCRIPT_PATH_FOR_TRAINING_HDFC
    xgboost_estimator = XGBoost(entry_point=script_path,
                                train_instance_type=TRAIN_INSTANCE_TYPE,
                                train_instance_count=1,
                                role=role,
                                sagemaker_session=sess,
                                output_path=XGB_OUTPUT_PATH.format(bucket, prefix),framework_version = XGB_FRAMEWORK_VERSION, 
                        py_version=PY_VERSION)
    
    xgboost_estimator.fit({'train': s3_input_train, 'validation': s3_input_validation})
    
    xgb_predictor = xgboost_estimator.deploy(initial_instance_count=1, instance_type=DEPLOY_INSTANCE_TYPE)

if __name__ == '__main__':
    train_and_deploy_hdfc()
    sys.exit(0)

