import h5py
import json  # Import json for proper JSON handling

# Path to your model file
file_path = '/home/alexberkeley/PCR_simulation/qPCRdeepNet/qpcrdeepnet/model_trn_covidcdc-ctf_40_ims_299_net_1_rgb_orig.h5'

with h5py.File(file_path, 'r+') as f:
    print(f.attrs)  # This will print all attributes, you can inspect them

    if 'training_config' in f.attrs:
        # Directly load the string into a Python dictionary using json.loads
        config = json.loads(f.attrs['training_config'])

        # Now config is a Python dictionary, and you can check and modify it
        print(config)  # See the current configuration

        # Assuming 'loss' is within config and has a 'reduction' property, modify it
        # First, ensure you're addressing the right path within the nested config
        if 'loss' in config and 'config' in config['loss'] and 'reduction' in config['loss']['config']:
            config['loss']['config']['reduction'] = 'sum_over_batch_size'  # Modify to a valid reduction setting
        elif 'loss' in config:
            # Set a default loss config if not properly set
            config['loss'] = {'class_name': 'BinaryCrossentropy', 'config': {'reduction': 'sum_over_batch_size'}}

        # Convert the modified dictionary back to a JSON string to save back to the HDF5 attribute
        f.attrs['training_config'] = json.dumps(config)

print('done')
