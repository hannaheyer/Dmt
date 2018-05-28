# Does all pre-processing of features. 
# Look at preprocess_test/train.sh for use: 
#  - Null filling, Feature combination & engineering is done first
#  - Normalization is split in 6 processes that can be run in parallel
#  - Final (-i 10) joins all files together.

import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from sklearn import linear_model

from scipy.stats import rankdata
from time import gmtime, strftime

import random
import time
import os
import math
import copy
import sys
import pickle

#loading the data
def save_data(data, filename):
    print("Saving {}...".format(filename),flush=True)
    with open(filename,'wb') as f:
        pickle.dump(data,f)
    print("--> {} succesfully saved!".format(filename))

def load_data(filename):
    with open(filename,'rb') as f:
        data = pickle.load(f)
    return data

def sample_by_query(data, N):
    s = time.time()
    ids = data.srch_id.unique()
    selection = np.random.choice(ids,N,replace=False)
    result = data.loc[data['srch_id'].isin(selection)]
    print('{0:.2f}% of total data sampled (n = {1:})'.format(result.shape[0]*100/data.shape[0], result.shape[0]),flush=True)
    print('{0:.2f}% of queries sampled'.format(N*100/len(ids)),flush=True)
    print('Sampling took {} seconds'.format(time.time()-s),flush=True)
    return result

def pp_time(input_data):
    data = copy.deepcopy(input_data)
    data["date_time"] = pd.to_datetime(data["date_time"])
    data["year"] = data["date_time"].dt.year
    data["month"] = data["date_time"].dt.month
    data["day"] = data["date_time"].dt.day
    data["hour"]= data["date_time"].dt.hour
    return data

def join_comps(input_data):
    #data = copy.deepcopy(input_data)
    data = input_data
    s = time.time()
    def remove_comp_outliers(data, *variables):
        # Removes outliers from the percent_diff set. Only high-end outliers are removed
        for var in variables:
            median = data[var].median()
            quantile = data[var].quantile(0.9)
            thresh = median + (1.5*(quantile-median))
            removed_outliers = []
            for i,point in enumerate(data[var].values):
                if not math.isnan(point):
                    if point > thresh:
                        removed_outliers.append(median)
                    else:
                        removed_outliers.append(point)
                else:
                    removed_outliers.append(median)
            data.drop(labels=[var],axis="columns",inplace=True)
            data[var] = removed_outliers
        return data
    
    def combine_comps(data, comp_vars):
        def non_null(array):
            # Returns all values in array that aren't NaN. 
            # If all NaN then return 0.
            r = [x for x in array if not math.isnan(x)]
            if len(r) == 0:
                return [0]
            else:
                return r

        rate_vars = [x for x in comp_vars if 'rate' in x and 'percent' not in x]
        inv_vars  = [x for x in comp_vars if 'inv'  in x]
        percent_vars = [x for x in comp_vars if 'percent' in x]
        comp_rate = np.zeros((len(data)))
        comp_inv = np.zeros((len(data)))
        comp_diff = np.zeros((len(data)))
        print('Combining rate and inv features,',flush=True)
        s = time.time()
        ts = []
        part_i = int(len(data)/5)
        for i, (idx,row) in enumerate(data.iterrows()):
            rates = row[rate_vars].values
            invs = row[inv_vars].values
            percents = row[percent_vars].values
            comp_rate[i] =  np.mean(non_null(rates))
            comp_inv[i] = np.mean(non_null(invs))
            comp_diff[i] = np.mean(non_null(rates*percents))
            #print(comp_diff[i])
            
            if math.isnan(np.mean(non_null(rates*percents))):
                print(rates)
                print(percents)
                print(rates*percents)
                print(non_null(rates*percents))
                print(np.mean(non_null(rates*percents)))
            if i != 0 and i%part_i == 0:
                print('{}/{}.'.format(i, len(data)),flush=True)
        
        data['comp_rate'] = comp_rate
        data['comp_inv'] = comp_inv
        data['comp_diff'] = comp_diff
       
        return data
    comp_vars = ['comp1_rate','comp1_inv','comp1_rate_percent_diff',
                'comp2_rate','comp2_inv','comp2_rate_percent_diff',
                'comp3_rate','comp3_inv','comp3_rate_percent_diff',
                'comp4_rate','comp4_inv','comp4_rate_percent_diff',
                'comp5_rate','comp5_inv','comp5_rate_percent_diff',
                'comp6_rate','comp6_inv','comp6_rate_percent_diff',
                'comp7_rate','comp7_inv','comp7_rate_percent_diff',
                'comp8_rate','comp8_inv','comp8_rate_percent_diff']
    data = combine_comps(data,comp_vars)
    data = remove_comp_outliers(data, 'comp_diff')
    #data.drop(comp_vars,axis=1)
    print('Joining Comps took {} minutes'.format((time.time()-s)/60),flush=True)
    return data

def fill_orig_dest(data_in):
    #data = copy.deepcopy(data_in)
    data = data_in
    # Fills orig_destination_distance with a replacement in the following order:
    # - first try the average over other prop_country_id (DEST) -> visitor_location_country_id (SRC) pairs
    # - if this is nan: take average over inverse orig_destination_distance (where prop_id is SRC and vice versa)
    # - if this is also nan, simply replace with mean value over distances to target destination.
    s = time.time()
    # Build distance matrix
    print('Filling nan in orig_destination_distance',flush=True)
    dest_ids = data.prop_country_id.unique()
    src_ids = data.visitor_location_country_id.unique()
    M = np.zeros((len(dest_ids),len(src_ids)))
    d_d = {v:i for i,v in enumerate(dest_ids)}
    s_d = {v:i for i,v in enumerate(src_ids)}
    for i,dest_id in enumerate(dest_ids):
        if i%10 == 0:
            print('{}/{}'.format(i, len(dest_ids)),flush=True)
        for j,src_id in enumerate(src_ids):
            bla = data.loc[(data['prop_country_id'] == dest_id) & (data['visitor_location_country_id'] == src_id)]
            bla2 = data.loc[(data['prop_country_id'] == src_id) & (data['visitor_location_country_id'] == dest_id)]
            mean_dist = np.mean(bla['orig_destination_distance'])
            if math.isnan(mean_dist):
                mean_dist = np.mean(bla2['orig_destination_distance'])
            M[i,j] = mean_dist
    print('Done building dist matrix in {} seconds'.format(time.time()-s),flush=True)
    
    # Building dict of average distance over property countries
    def mean_dist(data, prop_country_id):
        return np.mean(data.loc[data['prop_country_id'] == prop_country_id]['orig_destination_distance'])
    st = time.time()
    last_mean_dict = {d:mean_dist(data,d) for d in dest_ids}
    print('Building mean dict took {} seconds'.format(time.time()-st))
    for i,row in data.iterrows():
        src = row['visitor_location_country_id']
        dest = row['prop_country_id']
        if math.isnan(row['orig_destination_distance']):
            inferred = M[d_d[dest],s_d[src]]
            if math.isnan(inferred):
                data.at[i,'orig_destination_distance'] = last_mean_dict[dest]
            else:
                data.at[i,'orig_destination_distance'] = inferred
    t = time.time()-s
    print('Done filling distance nans in {} secs. aka {} mins'.format(t, t/60),flush=True)
    return data

def fill_nan_with(input_data, variable, filling):
    data = copy.deepcopy(input_data)
    data.loc[pd.isnull(data[variable]),variable] = filling
    return data

def add_price_order(input_data):
    # Sorts price per query from 1 (lowest price) to n. Average ranking used
    s = time.time()
    #data = copy.deepcopy(input_data)
    data = input_data
    data['price_rank'] = np.zeros((len(data)))
    srch_ids = data.srch_id.unique()
    part_i = int(len(srch_ids)/5)
    for i,srch_id in enumerate(srch_ids):
        srch_data = data.loc[data['srch_id'] == srch_id]
        price_sort = rankdata(np.array(srch_data['price_usd']))
        idxs = srch_data.index
        data.loc[idxs,'price_rank'] = price_sort
        if i%part_i == 0:
            print('{}/{}'.format(i, len(srch_ids)),flush=True)
    print('Adding price rank took {} seconds'.format(time.time()-s),flush=True)
    return data

def add_price_difference(input_data):
    # Converts prop_log_hist to actual price (inverse log) and divides current
    # price by that result. A value under 1 signifies how much lower the price is
    # with respect to earlier prices.
    s = time.time()
    data = copy.deepcopy(input_data)
    data['price_diff'] = data['price_usd']/(np.e**data['prop_log_historical_price'])
    print('Adding price diff took {} seconds'.format(time.time()-s),flush=True)
    return data

def normalize_by(data, target, by, name):
    s = time.time()
    #data = copy.deepcopy(input_data)
    data[name] = np.zeros((len(data)))
    ids = data[by].unique()
    part_i = int(len(ids)/5)
    for i, id_ in enumerate(ids):
        by_data = data.loc[data[by] == id_]
        idxs = by_data.index
        mean = np.mean(by_data[target])
        if mean ==0:
            mean = 1e-5
        data.loc[idxs,name] = by_data[target]/mean
        if i%part_i == 0:
            print('{}/{}'.format(i, len(ids)),flush=True)
    print('Adding {} took {} seconds'.format(name,time.time()-s),flush=True)
    return data

def fill_location_score2(input_data):
    def train_loc(data,train_vars):
        # Trains a simple linear regression model on the three variables below
        # in order to predict prop_location_score2, it its missing.
        train = data.loc[pd.notnull(data['prop_location_score2'])]
        #test = data.loc[pd.isnull(data['prop_location_score2'])]
        X = train[train_vars].values
        Y = train['prop_location_score2'].values
        regression_model = linear_model.LinearRegression()
        regression_model.fit(X,Y)
        return regression_model
    s = time.time()
    data = copy.deepcopy(input_data)
    # Features to predict score2 on
    X_vars = ['prop_location_score1','prop_starrating','prop_review_score']
    loc2_model = train_loc(data,X_vars)
    nan_scores = data.loc[pd.isnull(data['prop_location_score2'])]
    part_i = len(nan_scores)/5
    data.loc[pd.isnull(data['prop_location_score2']),'prop_location_score2'] = loc2_model.predict(nan_scores[X_vars].values)
    print('Predicting missing location score 2 values took {} seconds'.format(time.time()-s),flush=True)
    return data

def normalize(data, i,filename):

    if i == 1:
        data = normalize_by(data,'orig_destination_distance','srch_id','srch_norm_distance')
        save_data(data, filename)
        data = normalize_by(data,'orig_destination_distance','srch_destination_id','dest_norm_distance')
        #save_data(data, filename)
        #data = normalize_by(data,'orig_destination_distance','prop_id','prop_norm_distance')
        #save_data(data, filename)
    elif i == 2:
        #data = load_data(filename)
        data = normalize_by(data,'price_usd','srch_booking_window','window_norm_price')
        save_data(data, filename)
        data = normalize_by(data, 'price_usd','prop_id','prop_norm_price')
        save_data(data, filename)
        data = normalize_by(data, 'price_usd','srch_id','srch_norm_price')
        save_data(data, filename)
        data = normalize_by(data, 'price_usd', 'srch_destination_id','dest_norm_price')
        #save_data(data, filename)
    elif i == 3:
        data = normalize_by(data, 'prop_location_score1','srch_id','srch_norm_score1')
        save_data(data, filename)
        data = normalize_by(data, 'prop_location_score1','srch_destination_id','dest_norm_score1')
        #save_data(data, filename)
    elif i == 4:
        data = normalize_by(data, 'prop_location_score2','srch_id','srch_norm_score2')
        save_data(data, filename)
        data = normalize_by(data, 'prop_location_score2','srch_destination_id','dest_norm_score2')
        #save_data(data, filename)
    elif i == 5:
        data = normalize_by(data, 'prop_review_score','srch_id','srch_norm_review')
        save_data(data, filename)
        data = normalize_by(data, 'prop_review_score','srch_destination_id','dest_norm_review')
        #save_data(data, filename)
    elif i == 6:
        #data = load_data(filename)
        data = normalize_by(data, 'prop_starrating','srch_id','srch_norm_star')
        save_data(data, filename)
        data = normalize_by(data, 'prop_starrating','srch_destination_id','dest_norm_star')
        #save_data(data, filename)
    return data

def join_datasets(*datasets):
    def diff(a,b):
        b = set(b)
        return [x for x in a if x not in b]
    

    base = datasets[0]
    base_keys = list(base.keys())
    for i,d in enumerate(datasets[1:]):
        s = time.time()
        print("Joining set {}/6".format(i+1),flush=True)
        dif = diff(list(d.keys()),base_keys)
        for k in dif:
            print(k)
            base[k] = d.loc[:][k]
        print('Joined set {} in {} minutes'.format(i+1, (time.time()-s)/60), flush=True)
        d = None
    return base

    
def preprocess(data,filename):
    s = time.time()
    print('Adding time.',flush=True)
    data = pp_time(data)
    print('Adding time.',flush=True)
    data = join_comps(data)
    save_data(data, filename)
    print('Filling nans time.',flush=True)
    data = fill_nan_with(data,'srch_query_affinity_score',0)
    data = fill_nan_with(data,'prop_review_score',-1)
    data = fill_nan_with(data,'visitor_hist_adr_usd',-1)
    data = fill_nan_with(data,'visitor_hist_starrating',-1)
    save_data(data,filename)
    print('Filling orig_destination_distance',flush=True)
    data = fill_orig_dest(data)
    save_data(data,filename)
    print('Predict location_score2',flush=True)
    data = fill_location_score2(data)
    save_data(data,filename)
    print('Adding price_difference',flush=True)
    data = add_price_difference(data)
    save_data(data,filename)
    print('Adding price order.',flush=True)
    data = add_price_order(data)
    # Normalize by srch_id, prop_id, srch_destination_id
    t = time.time()-s
    save_data(data,filename)
    print("PREPROCESSING TOOK {} SECONDS/ {} MINUTES".format(t, t/60),flush=True)
    return data



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i',type=int)
    parser.add_argument('-type')
    parser.add_argument('-p',type=int)
    args = parser.parse_args()
    
    file_id = str(args.type)
    log_file = open("log_preprocess_{}.txt".format(file_id),"a")
    saveout = sys.stdout
    sys.stdout = log_file
    error_file = open("error_preprocess_{}.txt".format(file_id),"a")
    sys.stderr = error_file
    s = time.time() 
    if args.i == 0:
        print('Starting preprocessing',flush=True)
        if args.type == 'train':
            #with open('TEMP/train_0.pickle','rb') as f:
                #data = pickle.load(f)
            data = pd.read_csv('train.csv')
        else:
            data = pd.read_csv('test.csv')

        # TEMP splitting data in two
        #srch_ids = data.srch_id.unique()
        #n = len(srch_ids)
        #if args.p == 1:
            #data = data.loc[data['srch_id'].isin(srch_ids[:int(n/2)])]
        #elif args.p == 2:
            #data = data.loc[~data['srch_id'].isin(srch_ids[:int(n/2)])]

        #data = sample_by_query(data,1000)
        filename = 'TEMP/{}_{}.pickle'.format(args.type,args.i)
        #data = load_data(filename)
        data = preprocess(data,filename)
        with open('TEMP/{}_{}F.pickle'.format(args.type,args.i),'wb') as f:
            pickle.dump(data,f)
    elif args.i == 10:
        all_done = False
        while(not all_done):
            dir_list = os.listdir('.')
            all_done = True
            for i in range(7):
                name = 'TEMP/{}_{}F.pickle'.format(args.type,i)
                if name not in dir_list:
                    all_done = False
                    print('{} not found. Waiting untill finish.'.format(name),flush=True)
                    #print(dir_list)
            time.sleep(5)
        print('All files found. Joining datasets..',flush=True)
        data_list = []
        for i in range(7):
            with open('TEMP/{}_{}F.pickle'.format(args.type,i),'rb') as f:
                d = pickle.load(f)
                data_list.append(d)
        dataset = join_datasets(*data_list)
        with open('sets/{}_dataset.pickle'.format(args.type),'wb') as f:
            pickle.dump(dataset,f)

    else:

        try:
            with open('TEMP/{}_0F.pickle'.format(args.type),'rb') as f:
                data = pickle.load(f)
        except FileNotFoundError as e:
            print(e,flush=True)
            quit()

        filename = 'TEMP/{}_{}.pickle'.format(args.type,args.i)
        time_str = strftime("%H:%M:%S", gmtime())
        print('Started generating {} at {}'.format(filename, time_str),flush=True)
        data = normalize(data,args.i,filename)
        with open('TEMP/{}_{}F.pickle'.format(args.type,args.i),'wb') as f:
            pickle.dump(data,f)

    t = time.time()-s
    time_str = strftime("%H:%M:%S", gmtime())
    print('TASK {} TOOK {} MINUTES. Endtime: {}'.format(args.i,t/60,time_str),flush=True)
    sys.stdout.close()
    sys.stderr.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
