import pandas as pd
import os


if __name__ == "__main__":

    # Create directories
    os.makedirs('baseline/qpcrdeepnet', exist_ok=True)
    os.makedirs('baseline/qpcrdeepnet/input', exist_ok=True)
    os.makedirs('baseline/qpcrdeepnet/images', exist_ok=True)
    os.makedirs('baseline/qpcrdeepnet/output', exist_ok=True)

    # Process data
    curve_df = pd.read_hdf('data/data.h5', key='curve_data')
    sample_info = pd.read_hdf('data/data.h5', key='sample_info')
    igi_gene_call = pd.read_hdf('data/data.h5', key='igi_gene_call')

    join_df = (curve_df
            .merge(sample_info, how='inner', on=['well_position','pcr_plate'])
            .merge(igi_gene_call, how ='inner', on=['pcr_plate','sample_id','target']))
    
    thermo_sample_ids = join_df[join_df.target == 'MS2'].sample_id.unique()
    join_df['test_kit'] = 'LuNER'
    join_df.loc[join_df.sample_id.isin(thermo_sample_ids),'test_kit'] = 'Thermo'

    # Create test Rn dataframe
    df = (join_df.loc[(join_df['pcr_plate'] == 'AC00GY9K') & (join_df['cycle_no'] <= 40),['curve_idx','well_position','cycle_no','rn']]
    .pivot(index=['curve_idx','well_position'], columns = 'cycle_no', values='rn')
    .reset_index()
    )

    df.columns = ['curve_idx','well_position', 'Rn'] + ['']*39
    df.to_csv('baseline/qpcrdeepnet/input/rn_test.tsv', sep='\t')

