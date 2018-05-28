import pickle
import numpy as np
from subprocess import Popen, PIPE
import copy
import time
import sys
import os


feature_list = ["srch_id","site_id","visitor_location_country_id","visitor_hist_starrating","visitor_hist_adr_usd","prop_country_id","prop_id","prop_starrating","prop_review_score","prop_brand_bool","prop_location_score1","prop_location_score2","prop_log_historical_price","price_usd","promotion_flag","srch_destination_id","srch_length_of_stay","srch_booking_window","srch_adults_count","srch_children_count","srch_room_count","srch_saturday_night_bool","srch_query_affinity_score","orig_destination_distance","year","month","day","hour","comp_rate","comp_inv","comp_diff","price_diff","price_rank","srch_norm_distance","dest_norm_distance","window_norm_price","prop_norm_price","srch_norm_price","dest_norm_price","srch_norm_score1","dest_norm_score1","srch_norm_score2","dest_norm_score2","srch_norm_review","dest_norm_review","srch_norm_star","dest_norm_star"]
top_15 =[ 'price_rank','srch_norm_score2','srch_norm_star','dest_norm_star','prop_location_score2','prop_location_score1','srch_norm_review','prop_starrating','promotion_flag','srch_norm_score1','dest_norm_score1','dest_norm_score2','prop_review_score','prop_brand_bool','comp_rate']

name2id = {feat:(i+1) for i,feat in enumerate(feature_list)}

score_dict = {'prop_id': -0.0018026107075866367, 'dest_norm_price': 0.008695112228315652, 'visitor_hist_adr_usd': -0.009588108993786466, 'srch_norm_distance': -0.04030937193643347, 'srch_booking_window': 0.0023977255443631534, 'srch_destination_id': -0.0044032428393624605, 'srch_norm_star': 0.20481601465639238, 'promotion_flag': 0.09406147439647229, 'srch_norm_price': -0.03968932116406042, 'srch_norm_review': 0.10239182435769117, 'dest_norm_score1': -0.07846017685487483, 'prop_country_id': 0.01089039661056516, 'comp_inv': 0.0042432231981853735, 'srch_saturday_night_bool': 0.014296162605161769, 'srch_room_count': 0.013845178124964785, 'prop_brand_bool': 0.02882603313925817, 'dest_norm_review': -0.010446557750596793, 'srch_children_count': 0.03381204941324428, 'dest_norm_star': 0.12274018868354006, 'srch_norm_score1': 0.11243363228029878, 'srch_query_affinity_score': 0.022509501284238845, 'srch_length_of_stay': 0.016278429466504752, 'site_id': 0.014223308951751095, 'dest_norm_distance': -0.00888075085244777, 'hour': 0.002047172772040398, 'visitor_location_country_id': -0.013492960433510753, 'day': 0.004163318041758526, 'srch_id': -0.003028318629142734, 'visitor_hist_starrating': -0.006475499349402724, 'price_usd': 0.029711207849304627, 'prop_norm_price': -0.03757436428894241, 'srch_adults_count': 0.00540369339711885, 'orig_destination_distance': 0.010420374711979164, 'srch_norm_score2': 0.1325548069268707, 'prop_starrating': 0.10933741838547423, 'window_norm_price': 0.027449795373970114, 'year': -0.0008699286653392185, 'comp_diff': -0.01851676242700972, 'prop_review_score': 0.06914460985958744, 'prop_location_score1': -0.12850824405627626, 'month': 0.0035128115137472793, 'price_diff': -0.0048211416801935796, 'prop_log_historical_price': 0.013440055655254794, 'prop_location_score2': 0.1373352667030146, 'dest_norm_score2': 0.20211859292615297, 'price_rank': -0.6116609798462854, 'comp_rate': 0.0638190974490789}
               


def str2cmd(cmd_string):
    return [x for x in cmd_string.split(' ')]

def parseline(output,val):
    start_line = '#iter   | NDCG@38-T | NDCG@38-V |'
    #print(output)
    if '|' in output:
        split = output.split('|')
        if split[0].strip().isdigit():
            #print(split)
            train = float(split[1].strip())
            if val:
                val = float(split[2].strip())
            else:
                val = None
            iteration = int(split[0].strip())
            return iteration,train, val
        else:
            return 0,None, None
    else:
        return 0,None, None
        
def cmd2str(command):
    s = ''
    for x in command:
        s+= x
        s+= ' '
    return s
    

def run_command(command,i,val,save):
    print('COMMAND:')
    print(cmd2str(command))
    s = time.time()
    pipe = Popen(command, stdout=PIPE, stderr=PIPE,shell=False)
    #text = pipe.communicate()[0]
    train_scores=[]
    val_scores = []
    while True:
        iteration = 1
        output = pipe.stdout.readline()
        if not output:
            output = pipe.stderr.readline()
            if output:
                print('ERROR')
                print(output)
        else:
            if save:
                iteration,train_score, val_score = parseline(output.decode('utf-8'),val)
                #print(iteration)
                if train_score:
                    train_scores.append(train_score)
                    if val:
                        val_scores.append(val_score)
        if output == '' and pipe.poll() is not None:
            break
        if output:
            if (iteration+1)%verbness == 0:
                thing = 0
                print(output.strip())
        if b'Finished sucessfully.' in output:
            print("MODEL TRAINED")
            break
        rc = pipe.poll()
    
    pipe.stdout.close()
    pipe.stderr.close()
    if save:
        with open('train_progress/{}_trainscores.pickle'.format(i),'wb') as f:
            pickle.dump(train_scores,f)
        if val:
            with open('train_progress/{}_valscores.pickle'.format(i),'wb') as f:
                pickle.dump(val_scores,f)
        else:
            val_scores = [None]
        return max(train_scores), max(val_scores)
    else:
        return None, None

def get_best(feat_list):
    best = 0
    best_k = None
    for key in feat_list:
        score = np.abs(score_dict[key])
        if score > best:
            best = score
            best_k = key
            #print(best_k, score)
    return best_k
            
    
def remove(list_, target):
    return [x for x in list_ if x != target]

def generate_feat_file(feat_list,i,try_idx):
    # Given a list of features, create a feature file that can be used by ranklib
    feat_ids = [name2id[x] for x in feat_list]
    print('n features',len(feat_ids))
    filename = 'feature_files/try{}/featfile_it{}.txt'.format(try_idx,i)
    f = open(filename,'w')
    for idx in feat_ids:
        #print(idx)
        f.write('{}\n'.format(idx))
    f.close()
    return filename

def create_command(command,feat_list, idx,try_idx):
    # Given a feature list, create a feature file and add it to the command
    filename = generate_feat_file(feat_list, idx,try_idx)
    feature_cmd = ['-feature',filename]
    print(feature_cmd)
    try_command = copy.deepcopy(command)
    try_command.extend(feature_cmd)
    return try_command
    
def try_models1(cmd_str,name):
    # Start with top-15, then in order of logistic regression coefficient add new feature
    s = time.time()
    not_included = [x for x in feature_list if x not in top_15]
    command = cmd2str(cmd_str)
    print('Train/validate on baseline (top 15) model')
    best_features = top_15
    base_command = create_command(command,top_15,'base',1)
    train, base_line_val = run_command(base_command,'feat_select/try1/base',True, save=True)
    t = time.time()-s
    print('Validating baseline took {} minutes'.format(t/60))
    print("Baseline: t:{} v{}".format(train, base_line_val))
    print('Trying out models')
    file = open('training_progress/feat_select/{}_try_results.txt'.format(name),'w')
    result_line = '{0}|t{1:.3f}|v{2:.3f}|{3:.2f}min{4}'.format('base',train,base_line_val,t/60,str(top_15))
    file.write(result_line+"\n")
    print(len(not_included))
    for i in range(len(not_included)):
        print('________________')
        features = copy.deepcopy(best_features)
        key = get_best(not_included)
        features.append(key)
        try_command = create_command(command,features,i+1,1)
        train, val = run_command(try_command,'feat_select/try1/{}'.format(i+1),True, save=True)
        print('Best: {}, This: {}'.format(base_line_val, val))
        if val > base_line_val:
            best_features = features
            base_line_val = val
            print("Better model found: {}:{}".format(val,best_features))
        else:
            print('Feat {} performs worse. Removing from future it'.format(key))
            not_included = remove(not_included, key)

        t = time.time()-s
        result_line = '{0}|t{1:.3f}|v{2:.3f}|{3:.2f}min{4}'.format(i,train,val,t/60,str(features))
        file.write(result_line+"\n")
        print(result_line[:25])
    t = time.time() -s
    file.close()
    print('--> Done trying models in {} minutes'.format(t/60))
    print('Best val: {}'.format(base_line_val))
    print('Best features: {}'.format(str(best_features)))

def try_models2(cmd_str,name):
    # Start with top-15, then in order of logistic regression coefficient add new feature
    s = time.time()
    command = cmd2str(cmd_str)
    final_features = []
    print('Train/validate on baseline (top 15) model')
    best_features = features
    base_command = create_command(command,features,'base',2)
    train, base_line_val = run_command(base_command,'feat_select/try2/base',True, save=True)
    t = time.time()-s
    print('Validating baseline took {} minutes'.format(t/60))
    print("Baseline: t:{} v{}".format(train, base_line_val))
    print('Trying out models')
    file = open('training_progress/feat_select/{}_try_results.txt'.format(name),'w')
    result_line = '{0}|t{1:.3f}|v{2:.3f}|{3:.2f}min{4}'.format('base',train,base_line_val,t/60,str(top_15))
    file.write(result_line+"\n")
    for i in range(len(features)):
        print('________________')
        test_feats = copy.deepcopy(features)
        del test_feats[i]

        try_command = create_command(command,test_feats,i+1,2)
        train, val = run_command(try_command,'feat_select/try2/{}'.format(i+1),True, save=True)
        print('Base val: {}, Current val: {}'.format(base_line_val, val))
        if val < base_line_val:
            print('Model performce worse when dropping {}'.format(features[i]))
            final_features.append(features[i])
        else:
            print('Model performance stayed the same or increased when dropping {}')
        t = time.time()-s
        result_line = '{0}|t{1:.3f}|v{2:.3f}|{3:.2f}min{4}'.format(i,train,val,t/60,str(features[i]))
        file.write(result_line+"\n")
        print(result_line[:25])
    file.write('final features:'+str(final_features)+'\n')
    t = time.time() -s
    file.close()
    print('--> Done trying models in {} minutes'.format(t/60))
    #print('Best val: {}'.format(base_line_val))
    print('Best features: {}'.format(str(final_features)))
    print('Saving feature file')
    generate_feat_file(final_features,'BEST',2):


if __name__ == '__main__':
    import argparse
    s = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-verbness',type=int) #(1 = show all lines, 25 = show every 25th line)
    parser.add_argument('-train') # Train file
    parser.add_argument('-feat_select') # whether feature selection should be done
    parser.add_argument('-test') # test file
    parser.add_argument('-tvs') # Train validation split
    parser.add_argument('-out_model') # output model
    parser.add_argument('-load') # load model
    parser.add_argument('-out_score') # output of scores
    parser.add_argument('-feature') # Feature file
    parser.add_argument('-name') # run identifier
    args = parser.parse_args()
    verbness = args.verbness

    filename = './shuffled/SUBpart_sampled_train.txt.shuffled'
    testname = './test/test_fileL.txt'
    if args.train:
        # Training a model with specified features
        cmd_str = "java -jar RankLib.jar -train {} -feature {} -ranker 6 -estop 200 -metric2t NDCG@38".format(filename)
        val = False
        if args.tvs:
            cmd_str += ' -tvs {}'.format(args.tvs)
            val = True
        if args.out_model:
            cmd_str += ' -save {}'.format(args.out_model)
        run_command(str2cmd(cmd_str),args.name,val,save=True)
    elif args.test:
        # Rank a test file using a trained model
        cmd_str = 'java -jar RankLib.jar -feature {} -load {} -rank {} -indri {}'.format(args.feature,args.load,args.test,args.out_score)
        run_command(str2cmd(cmd_str),args.name,val, save=False)
    elif args.feat_select == 1:
        cmd_str = "java -jar RankLib.jar -train {} -ranker 6 -estop 200 -metric2t NDCG@38 -tvs 0.8".format(filename,modelname)
        try_models1(cmd_str, args.name)
    elif args.feat_select == 2:
        cmd_str = "java -jar RankLib.jar -train {} -ranker 6 -estop 200 -metric2t NDCG@38 -tvs 0.8".format(filename,modelname)
        try_models2(cmd_str, args.name)
    t = (time.time()-s)/60
    print('Running {} took {} minutes'.format(args.name, t))
