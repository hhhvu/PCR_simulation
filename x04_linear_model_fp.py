from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import io, os
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import KBinsDiscretizer
import statsmodels.api as sm

# Define a function to conduct logistic regression and report results
def run_logistic_regression(df, features, target, eval_df=None, eval_feats=None):
    print(df.shape)
    relevant = features + [target]
    reldf = df[relevant]
    df2 = reldf.dropna()
    print(df2.shape)


    # Calculate the sum per column
    column_sums = df2.sum()
    print(column_sums)
    print()

    X = np.array(df2[features])
    y = np.array(df2[target])

    # Step 2: Run logistic regression
    model = sm.Logit(y, sm.add_constant(X)).fit()

    # Step 3: Report coefficients and significance
    print(model.summary())

    # For sklearn LogisticRegression model
    # model = LogisticRegression(max_iter=10000).fit(X, y)
    # coef_dict = {}
    # for coef, feat in zip(model.coef_[0], features):
    #     coef_dict[feat] = coef
    # print(coef_dict)

    # Step 4: Report AUC
    if (type(eval_df) != pd.DataFrame) or (eval_feats == None):
        y_prob = model.predict(sm.add_constant(X))
        auc = roc_auc_score(y, y_prob)
        print(f"AUC: {auc:.4f}\n")
    else:
        relevant = eval_feats + [target]
        reldf = eval_df[relevant]
        df2 = reldf.dropna()

        X = np.array(df2[eval_feats])
        y = np.array(df2[target])

        y_prob = model.predict(sm.add_constant(X))
        auc = roc_auc_score(y, y_prob)
        print(f"AUC on test set: {auc:.4f}\n")

if __name__ == "__main__":

    ####################################
    #### Load the data
    ####################################

    with open('data/groundtruth_df_curve_dict_split_v2.pkl', 'rb') as file:
        curve_dict = pkl.load(file)
    
    target_df = pd.read_csv('data/groundtruth_df_target_data_split_v2.csv')  # Load your DataFrame here
    target_df.loc[:,['groundtruth_target']] = 1*(target_df.groundtruth == 1)

    #Define two error targets
    target_df.loc[:,'Igi_call_quant'] = 1*(target_df.igi_call == 'Positive')
    target_df['igi_fp'] = (target_df['Igi_call_quant'] > target_df['groundtruth_target']).astype(int)
    target_df['igi_fn'] = (target_df['Igi_call_quant'] < target_df['groundtruth_target']).astype(int)
    

    # Load the predictions
    pred_df = pd.read_csv('data/model_outputs/10_27_fusion_model_vit_delta64_test_pred_df.csv')
    pred_df_val = pd.read_csv('data/model_outputs/10_27_fusion_model_vit_delta64_val_pred_df.csv') 

    target_df_test = target_df[target_df.split == 'test']
    target_df_test = target_df_test.merge(pred_df, on = 'curve_idx', how='left')

    target_df_val = target_df[target_df.split == 'val']
    target_df_val = target_df_val.merge(pred_df_val, on = 'curve_idx', how='left')

    ####################################
    #### Encode prediction into 10 bins
    ####################################

    # Step 1: One-hot encode the column "outputs" into 10 bins
    encoder = KBinsDiscretizer(n_bins=10, encode='onehot-dense', strategy='uniform')
    outputs_binned = encoder.fit_transform(target_df_test[['outputs']])
    bin_edges = encoder.bin_edges_[0]
    bin_names_test = ['outputs_bin_' + str(round(bin_edges[i],2)) + '_' + str(round(bin_edges[i+1],2)) for i in range(len(bin_edges)-1)]
    target_df_test[bin_names_test] = pd.DataFrame(outputs_binned, columns=bin_names_test)

    outputs_binned = encoder.fit_transform(target_df_val[['outputs']])
    bin_edges = encoder.bin_edges_[0]
    bin_names_val = ['outputs_bin_' + str(round(bin_edges[i],2)) + '_' + str(round(bin_edges[i+1],2)) for i in range(len(bin_edges)-1)]
    target_df_val[bin_names_val] = pd.DataFrame(outputs_binned, columns=bin_names_val)

    ####################################
    #### Run regression analysis
    ####################################

    # Features for regression
    features_test = bin_names_test[1:] + ['Igi_call_quant']
    features_val = bin_names_val[1:] + ['Igi_call_quant']

    # Conduct logistic regression for different target variables
    targets = ['igi_fn_x', 'groundtruth_target'] #'igi_fp_x',
    for target in targets:
        print(f"Analysis for predicting {target}")
        run_logistic_regression(target_df_val,features_val, target, target_df_test, features_test)
        
        print()
        print('#############################')
        print('#############################')
        print()






