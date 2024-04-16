import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay, multilabel_confusion_matrix


if __name__ == "__main__":
    
    HIGHRISK_THRES = 0.566

    #########################################################################################
    ########## Load data
    #########################################################################################

    # Load data files
    curve_df = pd.read_hdf('data/data.h5', key='curve_data')
    sample_info = pd.read_hdf('data/data.h5', key='sample_info')
    igi_gene_call = pd.read_hdf('data/data.h5', key='igi_gene_call')

    join_df = (curve_df
            .merge(sample_info, how='inner', on=['well_position','pcr_plate'])
            .merge(igi_gene_call, how ='inner', on=['pcr_plate','sample_id','target']))

    groundtruth_df = pd.read_csv('data/groundtruth_df.csv')
    groundtruth_df['encoded_igi_call'] = 1*(groundtruth_df.igi_call == 'Positive')
    groundtruth_df['groundtruth_label'] = 1*(groundtruth_df.groundtruth == 1)

#     fusion_val_pred = pd.read_csv('data/model_outputs/fusion_vit_delta64_val_pred_df.csv')
#     fusion_train_pred = pd.read_csv('data/model_outputs/fusion_vit_delta64_train_pred_df.csv')
#     fusion_test_pred = pd.read_csv('data/model_outputs/fusion_vit_delta64_test_pred_df.csv')

#     fusion_all_pred = pd.concat([fusion_train_pred, fusion_val_pred, fusion_test_pred])
    
    fusion_all_pred = pd.read_csv('data/model_outputs/fusion_vit_delta64_clinical_pred_df.csv')
    print(fusion_all_pred.shape)

    fusion_all_pred = (fusion_all_pred
            .merge(groundtruth_df[['curve_idx','groundtruth_label','encoded_igi_call']].drop_duplicates(), 
                    how='left', on = 'curve_idx')
            .merge(join_df[['curve_idx','target','sample_id','final_patient_result','current_sample_result','sample_type','record_type','retest_sample_id_1']].drop_duplicates(), 
                    how = 'left', on = 'curve_idx'))
    fusion_all_pred = fusion_all_pred.rename(columns={'prob':'outputs'})

    # Subset to clinical samples
    print(fusion_all_pred.shape)
    fusion_all_pred = fusion_all_pred[fusion_all_pred.sample_type == 'Clinical Sample'] 
    print(fusion_all_pred.shape)

    # Get curve-level predictions
    fusion_all_pred['pred'] = (fusion_all_pred['outputs']>HIGHRISK_THRES).astype(int)

    # Aggregate to sample-level predictions
    fusion_all_pred['encoded_igi_call'] = (fusion_all_pred['current_sample_result']=='Positive').astype(int)
    fusion_all_pred['invalid_sample'] = (fusion_all_pred['current_sample_result']=='Invalid').astype(int)
    fusion_all_pred['inconclusive'] = (fusion_all_pred['current_sample_result']=='Inconclusive').astype(int)
    fusion_all_pred['retest'] = (~fusion_all_pred['current_sample_result'].isin(['Positive', 'Negative'])).astype(int)
    fusion_all_pred['conclusive'] = (fusion_all_pred['current_sample_result'].isin(['Positive', 'Negative'])).astype(int)
    fusion_all_pred = fusion_all_pred[~fusion_all_pred.target.isin(['MS2','RnaseP'])]
    print(fusion_all_pred.shape)
    fusion_all_pred = fusion_all_pred[fusion_all_pred['current_sample_result'].isin(['Positive', 'Negative', 'Inconclusive'])]
    fusion_all_pred = fusion_all_pred[~fusion_all_pred.current_sample_result.isna()]
    print(fusion_all_pred.shape)
    print(fusion_all_pred.current_sample_result.unique())

    fusion_all_pred_sample = (fusion_all_pred
                      .groupby(['sample_id'])
                      .agg(sample_pred = ('pred', 'mean'),
                           sample_igi_call = ('encoded_igi_call', 'mean')).reset_index())
    

#     known_sampled_pred = (fusion_all_pred
#                       .groupby(['dilution_level', 'well_position'])
#                       .agg(sample_pred = ('pred', 'mean'),
#                            sample_igi_call = ('new_igi_call', 'mean')
#                       .reset_index()))
    
    fusion_all_pred_sample['sample_pred'] = 1*(fusion_all_pred_sample.sample_pred >= 0.5).astype(int)
    fusion_all_pred_sample['sample_igi_call'] = 1*(fusion_all_pred_sample.sample_igi_call >= 0.5).astype(int)

    #########################################################################################
    ########## Obtain Confusion Matrix
    #########################################################################################

    cm = confusion_matrix(fusion_all_pred.encoded_igi_call, 
                      fusion_all_pred.pred)
    
    ConfusionMatrixDisplay(cm,).plot(values_format='d') #display_labels=['Negative','Positive']
    plt.ylabel('IGI Label')
    plt.xlabel('Model Predictions')
    plt.title('Agreement Matrix')
    plt.savefig('20240115_clinical_sample_confusion_curve.png')
    plt.show()

    cm = confusion_matrix(fusion_all_pred_sample.sample_igi_call, 
                      fusion_all_pred_sample.sample_pred)
    
    ConfusionMatrixDisplay(cm,).plot(values_format='d') #display_labels=['Negative','Positive']
    plt.ylabel('IGI Label')
    plt.xlabel('Model Predictions')
    plt.title('Agreement Matrix')
    plt.savefig('20240115_clinical_sample_confusion_sample.png')
    plt.show()

    cm = confusion_matrix(fusion_all_pred.encoded_igi_call, fusion_all_pred.pred)
    cm = np.flip(cm)  # Flip the matrix to rearrange the blocks

    # Plotting with custom settings
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Positive', 'Negative'])
    disp.plot(cmap='GnBu', values_format='d', colorbar=False)  # Using a subtle color map
    plt.ylabel('IGI Label')
    plt.xlabel('Model Predictions')
    #plt.title('Agreement Matrix')
    #plt.gca().invert_yaxis()  # Invert y-axis to place Positive on top
    plt.gca().invert_xaxis()  # Invert x-axis to place Positive on left
    plt.savefig('20240115_clinical_sample_confusion_curve.png')
    plt.show()

    # Assuming fusion_all_pred_sample.sample_igi_call and fusion_all_pred_sample.sample_pred are defined
    # Customizing the second confusion matrix plot
    cm_sample = confusion_matrix(fusion_all_pred_sample.sample_igi_call, fusion_all_pred_sample.sample_pred)
    cm_sample = np.flip(cm_sample)  # Flip the matrix to rearrange the blocks

    # Plotting with custom settings
    disp_sample = ConfusionMatrixDisplay(confusion_matrix=cm_sample, display_labels=['Positive', 'Negative'])
    disp_sample.plot(cmap='GnBu', values_format='d', colorbar=False)  # Using a subtle color map
    plt.ylabel('IGI Label')
    plt.xlabel('Model Predictions')
    #plt.title('Agreement Matrix')
    #plt.gca().invert_yaxis()  # Invert y-axis to place Positive on top
    plt.gca().invert_xaxis()  # Invert x-axis to place Positive on left
    plt.savefig('20240115_clinical_sample_confusion_sample.png')
    plt.show()

    #########################################################################################
    ########## Estimate positive rates for diagreement groups with known label
    #########################################################################################

    # Load data
    fusion_val_pred = pd.read_csv('data/model_outputs/fusion_vit_delta64_val_pred_df.csv')
    fusion_test_pred = pd.read_csv('data/model_outputs/fusion_vit_delta64_test_pred_df.csv')

    # Merge in additional information
    fusion_all_pred_control = pd.concat([fusion_val_pred, fusion_test_pred])

    fusion_all_pred_control = (fusion_all_pred_control
            .merge(groundtruth_df[['curve_idx','groundtruth_label','encoded_igi_call']].drop_duplicates(), 
                    how='left', on = 'curve_idx')
            .merge(join_df[['curve_idx','target','sample_id','final_patient_result','current_sample_result','sample_type','record_type','retest_sample_id_1']].drop_duplicates(), 
                    how = 'left', on = 'curve_idx'))
    
    fusion_all_pred_control = fusion_all_pred_control.rename(columns={'prob':'outputs'})
    fusion_all_pred_control['pred'] = (fusion_all_pred_control['outputs']>HIGHRISK_THRES).astype(int)

    print(fusion_all_pred_control.shape)
    print(fusion_all_pred_control.current_sample_result.unique())
    print(fusion_all_pred_control.shape)
    print(fusion_all_pred_control.head())

   # Define the subpopulations
   
    subpopulations = {
    'Both Positive': ((fusion_all_pred_control['pred'] == 1) & (fusion_all_pred_control['encoded_igi_call'] == 1)),
    'Both Negative': ((fusion_all_pred_control['pred'] == 0) & (fusion_all_pred_control['encoded_igi_call'] == 0)),
    'Pred Positive IGI Neg': ((fusion_all_pred_control['pred'] == 1) & (fusion_all_pred_control['encoded_igi_call'] == 0)),
    'Pred Negative IGI Pos': ((fusion_all_pred_control['pred'] == 0) & (fusion_all_pred_control['encoded_igi_call'] == 1))
    }

    # Calculate the share of 'groundtruth_label' == 1 in each subpopulation
    results = {}
    for key, mask in subpopulations.items():
        subpopulation = fusion_all_pred_control[mask]
        share = (subpopulation['groundtruth_label'] == 1).mean()
        results[key] = share

    # Print the results
    print("Share of 'groundtruth_label' == 1 in each subpopulation:")
    for key, value in results.items():
        print(f"{key}: {value:.2%}")
    
    # Plotting the results as a bar chart
    labels = list(results.keys())
    shares = [value * 100 for value in results.values()]  # Convert to percentages

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, shares, width=0.4)
    #plt.xlabel('Subpopulations')
    plt.ylabel('Share of positive control samples, %')
    #plt.title('Share of Positive Control')
    plt.ylim(0, 100)  # Ensure the y-axis starts at 0 and ends at 100 for percentage clarity
    plt.xticks(rotation=45)  # Rotate labels to improve readability

    # Adding labels to each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.2f}%', ha='center', va='bottom')  # Adjust text alignment and position

    plt.tight_layout()
    plt.savefig('20240115_subpopulation_analysis.png')
    plt.show()

    #########################################################################################
    ########## Print key claims
    #########################################################################################
    