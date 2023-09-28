import numpy as np
import pandas as pd
import pickle

if __name__ == "__main__":
    groundtruth_df = pd.read_csv('data/groundtruth_df.csv')

    ####################################
    ## Create dictionary of PCR curves
    ####################################

    # Initialize an empty dictionary
    curve_dict = {}

    # Initialize variables to track the current cycle and array
    current_cycle = None
    current_array = None

    # Iterate over each row in the DataFrame
    for index, row in groundtruth_df.iterrows():
        curve_idx = row['curve_idx']
        cycle_no = row['cycle_no']
        Fn = row['Fn']

        # Create a new array if cycle_no resets to 1
        if cycle_no == 1:
            current_array = []
            
        # Append Fn to the current array
        current_array.append(Fn)

        # Save the array in the dictionary with key as curve_idx
        curve_dict[curve_idx] = current_array.copy()
    
    # Open the file in read-binary mode and load the dictionary
    with open('data/groundtruth_df_curve_dict.pkl', 'wb') as file:
        pickle.dump(curve_dict, file)
    
    ###########################################
    ## Create dataframe with PCR curve labels
    ###########################################

    ### We should use the groundtruth column as a training label

    #Get target dataframe
    target_data = groundtruth_df.groupby('curve_idx').tail(1)
    target_data['Igi_call_quant'] = (target_data['igi_call']=="Positive").astype(int)
    target_data['groundtruth_target'] = (target_data['groundtruth']==1).astype(int)
    print(target_data.head())

    target_data.to_csv('data/groundtruth_df_target_data.csv')