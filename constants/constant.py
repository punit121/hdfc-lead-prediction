
import os,sys

# Big brother event processed data names
ALL_EVENTS_MAX_PROPERTY="SMax_property"
ALL_EVENTS_MAX_OCCURED_CITY_NAME="SMax_city_name"
ALL_EVENTS_MAX_OCCURED_SUBCAT_NAME="SMax_subcategory_name"
ALL_EVENTS_MAX_SOURCE="SMax_source"
ALL_EVENTS_UTM_TRAFFIC_SOURCE="SMax_utm_traffic_source"

RESPONSE_EVENTS_MAX_PROPERTY="RMax_property"
RESPONSE_EVENTS_MAX_OCCURED_CITY_NAME="RMax_city_name"
RESPONSE_EVENTS_MAX_OCCURED_SUBCAT_NAME="RMax_subcategory_name"
RESPONSE_EVENTS_MAX_SOURCE="RMax_source"
RESPONSE_EVENTS_UTM_TRAFFIC_SOURCE="RMax_utm_traffic_source"

EMPTY_EVENTS_VALUE="zzxa"
TRAIN_OUTPUT_VALUE="dvJun13"
RESPONSE_EVENTS_PROPERTIES_RESPONDED="R30d_distinctpropertiesresponded"
RESPONSE_EVENTS_SALE_SERIOUSNESS_FACTOR="R30d_Saleseriousness"
RESPONSE_EVENTS_MAX_PRICE="RMax_price"
RESPONSE_EVENTS_SALE_RESPONSES="R30d_Saleresponses"
ALL_EVENTS_SERIOUSNESS_FACTOR="S30d_Saleseriousness"
ALL_EVENTS_LOCALITIES_COUNT="S30d_distinctcitiesviewed"
ALL_EVENTS_PROPERTIES_VIEWED="S30d_distinctpropertiesviewed"
ALL_EVENTS_MAX_AREA_SQ_FEET="SMax_Area_Sq_Feet"
ALL_EVENTS_MAX_PRICE="SMax_price"

#Path to upload model
S3_BUCKET_PATH="hdfc-model-v2"
S3_PREFIX="sagemaker/hdfc-model"

#Script path for training hdfc model
SCRIPT_PATH_FOR_TRAINING_HDFC="sklearn_train_hdfc.py"

#Path for uploading hdfc data
HDFC_DATA_PATH="hdfc_data.csv"

HDFC_TRAIN_DATA="train.csv"

HDFC_VALIDATION_DATA="validation.csv"

HDFC_TRAIN_UPLOAD_PATH="train/train.csv"

HDFC_VALIDATION_UPLOAD_PATH="validation/validation.csv"

S3_RESOURCE="s3"

TRAIN_INSTANCE_TYPE="ml.m4.xlarge"

DEPLOY_INSTANCE_TYPE="ml.t2.medium"

#Version  for Xgboost Framework
XGB_FRAMEWORK_VERSION="0.90-1"

XGB_OUTPUT_PATH="s3://{}/{}/output"

PY_VERSION="3"