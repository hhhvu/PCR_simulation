import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

#scp -i ~/.ssh/my_private_key.pem Users/alexa/Downloads/new_groundtruth_df_1.csv alexberkeley@13.57.176.147:/home/alexberkeley/PCR_simulation/data


if __name__ == "__main__":
    groundtruth_df = pd.read_csv('data/new_groundtruth_df_no_invalid.csv')
    # groundtruth_df = pd.read_csv('data/new_groundtruth_df_1.csv')
    print(f"label data shape {groundtruth_df.shape}")

    curve_data = pd.read_hdf('data/new_data_1.h5',key = 'curve_data')
    print(f"curve data shape {curve_data.shape}")
    print(len(curve_data['curve_idx'].unique()))
    print(len(groundtruth_df['curve_idx'].unique()))

    is_subset = set(groundtruth_df['curve_idx'].unique()).issubset(set(curve_data['curve_idx'].unique()))
    print(is_subset)

    print(curve_data['curve_idx'].max())
    print(groundtruth_df['curve_idx'].max())


    #print(curve_data.head(50))
    curve_data = curve_data.sort_values(by=['curve_idx', 'cycle_no'])
    print(curve_data.head(81))

    ###########################################
    ## Create dataframe with PCR curve labels
    ###########################################

    ### We should use the groundtruth column as a training label

    #Get target dataframe
    target_data = groundtruth_df.groupby('curve_idx').tail(1)
    print(target_data.columns)
    target_data['Igi_call_quant'] = (target_data['igi_call']=="Positive").astype(int)
    target_data['groundtruth_target'] = (target_data['groundtruth']==1).astype(int)
    print(target_data.sort_values(by="curve_idx").head())
    print(target_data['split'].unique())
    print(target_data['sample_type'].unique())
    print(target_data.shape)

    target_data.to_csv('data/groundtruth_df_target_data_split_v3.csv')
    target_data.to_csv('data/new_groundtruth_df_target_data_no_invalid.csv')

    # target_df2 = pd.read_csv('data/groundtruth_df_target_data.csv') 
    # print(target_df2.sort_values(by="curve_idx").head())

    
    ####################################
    ## Create dictionary of PCR curves
    ####################################

    print(groundtruth_df)
    print(groundtruth_df.columns)
    groundtruth_ids = groundtruth_df['curve_idx'].unique()

    # Initialize an empty dictionary
    curve_dict_fn = {}
    curve_dict_drn = {}

    # Initialize variables to track the current cycle and array
    current_cycle = None
    current_array = None

    # Iterate over each row in the DataFrame
    for index, row in tqdm(curve_data.iterrows()):
        curve_idx = row['curve_idx']
        cycle_no = row['cycle_no']
        Fn = row['Fn']
        drn = row['drn']

        # Create a new array if cycle_no resets to 1
        if cycle_no == 1:
            current_array_fn = []
            current_array_drn = []
            
        # Append Fn to the current array
        current_array_fn.append(Fn)
        current_array_drn.append(drn)

        # Save the array in the dictionary with key as curve_idx
        curve_dict_fn[curve_idx] = current_array_fn.copy()
        curve_dict_drn[curve_idx] = current_array_drn.copy()
    
    # Open the file in read-binary mode and load the dictionary
    with open('data/groundtruth_df_curve_dict_split_v3.pkl', 'wb') as file:
        pickle.dump(curve_dict, file)
    groundtruth_curve_dict_fn = {key: curve_dict_fn[key] for key in groundtruth_ids}
    groundtruth_curve_dict_drn = {key: curve_dict_drn[key] for key in groundtruth_ids}

    # # Open the file in read-binary mode and load the dictionary
    with open('data/new_groundtruth_df_curve_dict_fn_no_invalid.pkl', 'wb') as file:
        pickle.dump(groundtruth_curve_dict_fn, file)

    with open('data/new_groundtruth_df_curve_dict_drn_no_invalid.pkl', 'wb') as file:
        pickle.dump(groundtruth_curve_dict_drn, file)
    
    with open('data/new_full_curve_dict_fn_no_invalid.pkl', 'wb') as file:
        pickle.dump(curve_dict_fn, file)
    
    with open('data/new_full_curve_dict_drn_no_invalid.pkl', 'wb') as file:
        pickle.dump(curve_dict_drn, file)
