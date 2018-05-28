
#Importing the needed packages
import pandas as pd
from pandas import Series, DataFrame
import numpy as np


from collections import defaultdict
import random
import time
import os
import math
import copy
import pickle


def create_train_file(data, name, features,negsamp_rate):
    # This method converts the dataset to a txt format on which 
    # LambdaMART can be trained according to LEMUR file specification
    # data: dataset
    # name: name of train file
    # features: list of feature names.


    def get_negative_count(data, srch_id):
        d = data.loc[data['srch_id'] == srch_id]
        n_negative = sum(d['click_bool'] == 0)
        #target_n = sum(d['click_bool'] == 1)
        return n_negative
    
    def is_numerical(val):
        if type(val) == float or type(val) == int:
            return True
        else:
            for el in val:
                if el != '.':
                    if not el.isdigit():
                        return False
        return True
    s = time.time()
    if negsamp_rate:
        print('building neg dict') 
        srch_ids = data.srch_id.unique()
        neg_dict = {i:get_negative_count(data,i) for i in srch_ids}
        neg_count_dict= defaultdict(lambda:0)
        print('building neg dict done')

    k2n = {k+1:name for k,name in enumerate(features)}
    n2k = {v:k for (k,v) in k2n.items()}
    prop_key = n2k['prop_id']

    f = open('model_input/'+name+'_train.txt','w')
    n = len(data)
    i_part = n/10
    for idx,(i, row) in enumerate(data.iterrows()):
        srch_id = row['srch_id']
        #print(idx)
        if idx%i_part == 0:
           print('{}/{}'.format(i, n))
        #negative sampling
        if negsamp_rate:
            is_neg = (row['click_bool'] == 0)
            if is_neg:
                n_neg = neg_dict[srch_id]
                if np.random.rand(1)[0] > (negsamp_rate/n_neg):
                   continue
            if is_neg:
               neg_count_dict[srch_id] += 1
        target = max(np.array((row['booking_bool'],row['click_bool']))*[5,1])
        feat_dict = {k+1:row[name] for k,name in enumerate(features)}
        line = feat_to_line(target,srch_id, feat_dict)
        line+='\n'
        f.write(line)
    f.close()
    print('Writing to file took {} seconds'.format(time.time()-s))
    return 1

def create_test_file(data, name, features):
    # This method converts the dataset to a txt format on which 
    # LambdaMART can be trained according to LEMUR file specification
    # data: dataset
    # name: name of train file
    # features: list of feature names.

    s = time.time()

    def is_numerical(val):
        if type(val) == float or type(val) == int:
            return True
        else:
            for el in val:
                if el != '.':
                    if not el.isdigit():
                        return False
        return True
    k2n = {k+1:name for k,name in enumerate(features)}
    n2k = {v:k for (k,v) in k2n.items()}
    prop_key = n2k['prop_id']

    f = open('model_input/'+name+'_test.txt','w')
    n = len(data)
    i_part = n/10
    for idx,(i, row) in enumerate(data.iterrows()):
        srch_id = row['srch_id']
        #print(idx)
        if idx%i_part == 0:
           print('{}/{}'.format(i, n))
        target = 0#max(np.array((row['booking_bool'],row['click_bool']))*[5,1])
        feat_dict = {k+1:row[name] for k,name in enumerate(features)}
        line = feat_to_line(target,srch_id, feat_dict,prop_key)
        line+='\n'
        f.write(line)
    f.close()
    print('Writing to file took {} seconds'.format(time.time()-s))
    
    return 1





def feat_to_line(target, srch_id, feat_dict,prop_key):
        line = '{} qid:{} '.format(target, srch_id)
        prop_val = -1
        for (k,v) in list(feat_dict.items()):
            assert is_numerical(v), 'non numerical value detected: {}'.format(v)
            if math.isnan(v):
                v = 0

            if k == prop_key:
                prop_val = v
            line += '{}:{} '.format(k,v)
        line += '#prop_id:{}'.format(prop_val)
        return line
        
def sample_by_query(data, N):
    s = time.time()
    ids = data.srch_id.unique()
    selection = np.random.choice(ids,N,replace=False)
    result = data.loc[data['srch_id'].isin(selection)]
    print('{0:.2f}% of total data sampled (n = {1:})'.format(result.shape[0]*100/data.shape[0], result.shape[0]))
    print('{0:.2f}% of queries sampled'.format(N*100/len(ids)))
    print('Sampling took {} seconds'.format(time.time()-s))
    return result

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-name')
    parser.add_argument('-type')
    parser.add_argument('-negsamp',type=int)
    parser.add_argument('-sample')
    args = parser.parse_args()
    
    #file_id = str(args.type)
    #log_file = open("log_preprocess_{}.txt".format(file_id),"a")
    if args.type == 'test':
        assert args.sample == 0, 'No sampling allowed on test file'
        assert args.negsamp == 0, 'No negative sampling allowed on test file'
    s = time.time()
    #print('Opening dataset without negative samling')
    if args.type == 'train':
        with open('./sets/train_dataset.pickle','rb') as f:
            data = pickle.load(f)
    elif args.type == 'test':
        with open('./sets/test_dataset.pickle','rb') as f:
            data = pickle.load(f)

    if not arg.type == 'test' and args.sample:     
        print('Sampling data')
        data = sample_by_query(data, 10000)
    print("Starting train generation")
    feature_list = ["srch_id","site_id","visitor_location_country_id","visitor_hist_starrating","visitor_hist_adr_usd","prop_country_id","prop_id","prop_starrating","prop_review_score","prop_brand_bool","prop_location_score1","prop_location_score2","prop_log_historical_price","price_usd","promotion_flag","srch_destination_id","srch_length_of_stay","srch_booking_window","srch_adults_count","srch_children_count","srch_room_count","srch_saturday_night_bool","srch_query_affinity_score","orig_destination_distance","year","month","day","hour","comp_rate","comp_inv","comp_diff","price_diff","price_rank","srch_norm_distance","dest_norm_distance","window_norm_price","prop_norm_price","srch_norm_price","dest_norm_price","srch_norm_score1","dest_norm_score1","srch_norm_score2","dest_norm_score2","srch_norm_review","dest_norm_review","srch_norm_star","dest_norm_star"]
    if args.type == 'train':
        a = create_train_file(data,args.name,feature_list,args.negsamp)
    elif args.type == 'test':
        a = create_test_file(data,args.name,feature_list)
    #print(a)
    print('Done creating file {} in {} minutes'.format(args.name,(time.time()-s)/60))
