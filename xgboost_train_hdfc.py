import argparse
import pandas as pd
import os
import numpy as np
from sklearn import tree
from sklearn.externals import joblib
import csv
import json
import subprocess
import sys
import xgboost
from io import StringIO
from helpers.helper import *  
from sagemaker_containers.beta.framework import (
    encoders, env, modules, transformer, worker)


def train_xgboost_hdfc_model(args,hyperparameters):

    # Take the set of files and read them all into a single pandas dataframe
    input_files = [ os.path.join(args.train, file) for file in os.listdir(args.train) ]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))
    raw_data = [ pd.read_csv(file, header=None, engine="python") for file in input_files ]
    train_data = pd.concat(raw_data)

    # labels are in the first column
    train_y = train_data.iloc[:,0]
    train_X = train_data.iloc[:,1:]
    train_X = train_X.values
    #X_test = X_test.as_matrix()
    
    clf = xgboost.XGBClassifier(**hyperparameters)
    clf = clf.fit(train_X, train_y)

    joblib.dump(clf, os.path.join(args.model_dir, "model.joblib"))


def model_fn(model_dir):

    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf


def input_fn(request_body, request_content_type):
    """An input_fn that loads a json data    """
    if request_content_type == "application/json":
        json_load = json.loads(request_body)
        data=get_dataframe_from_dict(json_load)
        csv_data=data.to_csv(index=False,header=None)
        data=csv_data.replace("\n","")
        s = StringIO(data)
        data = pd.read_csv(s, header=None)

        return data
    else:
        # Handle other content-types here or raise an Exception
        # if the content type is not supported.
        pass


def predict_fn(input_data, model):

    input_data=input_data.values
    output = model.predict_proba(input_data)
    return output

def output_fn(prediction, accept):

    response = get_response_dict(prediction.tolist()[0][1])
    return worker.Response(json.dumps(response), mimetype=accept)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    args = parser.parse_args()

    hyperparameters=get_hyperparameters()

    train_xgboost_hdfc_model(args,hyperparameters)