import os,sys
import numpy as np
import json


#using dict constructers


utm_traffic_source_dict = dict(alerts=0,
                               cf=1,
                               direct=2,
                               facebook=3,
                               internal=4,
                               organic=5,
                               otherpaid=6,
                               referral=7,
                               sem=8,
                               sms=9,
                               utm_not_available_in_dict=10)

city_name_dict = dict(bangalore=0,
                      chennai=1,
                      delhi_ncr=2,
                      hyderabad=3,
                      kolkata=4,
                      mumbai_mmr=5,
                      pune=6,
                      roi=7,
                      city_not_available_in_dict=8)

processed_data_default_dict = dict(response_events_properties_responded=[np.asarray(0)],
                                   response_events_sale_seriousness_factor=[np.asarray(0)],
                                   response_events_max_price=[np.asarray(1)],
                                   response_events_utm_traffic_source=[np.asarray(10)],
                                   response_events_sale_responses=[np.asarray(0)],
                                   response_events_max_occured_city_name=[np.asarray(8)],
                                   all_events_seriousness_factor=[np.asarray(0)],
                                   all_events_localities_count=[np.asarray(0)],
                                   all_events_max_occured_city_name=[np.asarray(0)],
                                   all_events_properties_viewed=[np.asarray(0)],
                                   all_events_max_area_sq_feet=[np.asarray(0)],
                                   all_events_max_price=[np.asarray(0)])

response_dict = dict(payload=dict(confidence_score=None))

hyperparameter_dict=h_params= dict(learning_rate=0.1,
                                   colsample_bytree=0.4,
                                   subsample=0.4,
                                   objective='binary:logistic',
                                   n_estimators=1000,
                                   reg_alpha=0.3,
                                   max_depth=5,
                                   gamma=10)