# After preprocessing: evaluation, feature selection and model training

# Building model files from datasets
python3 build_model_file.py -name 'negsamp6_sub' -type 'train' -negsamp 6 -sample 1&
python3 build_model_file.py -name 'negsamp1_sub' -type 'train' -negsamp 1 -sample 1&
python3 build_model_file.py -name 'negsamp0_sub' -type 'train' -negsamp 0 -sample 1&
python3 build_model_file.py -name 'full' -type 'test' -negsamp 0 -sample 0


# First do feature selection
python3 run_lambaMART.py -feat_select 1 -verbness 25 -train 'model_input/negsamp6_sub_train.txt' -name 'try1_feature'
python3 run_lambaMART.py -feat_select 2 -verbness 25 -train 'model_input/negsamp6_sub_train.txt' -name 'try2_feature'

# Try different Negative Sampling methods
python3 run_lambdaMART.py -tvs 0.8 -verbness 1 -name 'train_negsamp6_sub' -train 'model_input/negsamp6_sub_train.txt' -feature '/feature_files/try2/featfile_itBEST.txt' 
python3 run_lambdaMART.py -tvs 0.8 -verbness 1 -name 'train_negsamp1_sub' -train 'model_input/negsamp1_sub_train.txt' -feature '/feature_files/try2/featfile_itBEST.txt' 
python3 run_lambdaMART.py -tvs 0.8 -verbness 1 -name 'train_negsamp0_sub' -train 'model_input/negsamp0_sub_train.txt' -feature '/feature_files/try2/featfile_itBEST.txt' 


# When we know which one is better build the full train file
#python3 build_model_file.py -name 'full' -type 'train' -negsamp X -sample 0


# Train FULL model
python3 run_lambdaMART.py -tvs 0.8 -verbness 1 -out_model 'full_model.txt' -name 'train_full' -train 'model_input/negsampX_full_train.txt' -feature '/feature_files/try2/featfile_itBEST.txt' 

# Rank test file
python3 run_lambdaMART.py -verbness 1 -out_score 'model_scores.txt' -load 'full_model.txt' -name 'train_full' -test 'test/full_test.txt' -feature '/feature_files/try2/featfile_itBEST.txt' 


# Create listing
python3 create_listing.py -score 'model_scores.txt'
