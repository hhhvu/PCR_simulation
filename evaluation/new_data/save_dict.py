import pickle
import pandas as pd

groundtruth_df = pd.read_csv('/media/ssd1/huong/PCR-huong/data/new_groundtruth_df.csv')  # Load your DataFrame here
groundtruth_df = groundtruth_df
groundtruth_df.loc[:,['groundtruth_target']] = 1*(groundtruth_df.groundtruth == 1)
#Define two error targets
groundtruth_df.loc[:,'Igi_call_quant'] = 1*(groundtruth_df.igi_call == 'Positive')
groundtruth_df.loc[:,'igi_fp'] = (groundtruth_df['Igi_call_quant'] > groundtruth_df['groundtruth_target']).astype(int)
groundtruth_df.loc[:,'igi_fn'] = (groundtruth_df['Igi_call_quant'] < groundtruth_df['groundtruth_target']).astype(int)


groundtruth_sample_curve_dict = {}
for i in groundtruth_df.curve_idx.unique():
    df = groundtruth_df[groundtruth_df.curve_idx == i]
    groundtruth_sample_curve_dict[i] = df['Fn'].to_list()

groundtruth_df = groundtruth_df[groundtruth_df.cycle_no == 40]
groundtruth_df.to_csv('/media/ssd1/huong/PCR-huong/data/groundtruth_target_data.csv', index=False)

with open('/media/ssd1/huong/PCR-huong/data/new_groundtruth_curve_dict.pkl', 'wb') as handle:
    pickle.dump(groundtruth_sample_curve_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)