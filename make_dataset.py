import json
import glob
import random
import matplotlib
matplotlib.use('Agg')

base_path = '/home/tonyxu74/scratch/data/mri/'
base_path = '.'

data_paths = glob.glob(base_path + '*/*.nii.gz')

grouped_data_dict = {}

for dp in data_paths:
    field_strength = None
    image_type = None
    data_id = None

    if 'IXI' in dp:
        _, hospital, _, image_type = dp.split('/')[-1].replace('.nii.gz', '').split('-')
        if hospital == 'Guys':
            field_strength = '1.5T'
        elif hospital == 'HH':
            field_strength = '3T'
        elif hospital == 'IOP':
            field_strength = '1.5T'
        data_id = f'IXI-{hospital}-{field_strength}-{image_type}'

    elif 'CC' in dp:
        image_type = 'T1'
        _, scanner, field_strength, _, _ = dp.split('/')[-1].replace('.nii.gz', '').split('_')
        if field_strength == '3':
            field_strength = '3T'
        elif field_strength == '15':
            field_strength = '1.5T'
        data_id = f'CC-{scanner}-{field_strength}-{image_type}'

    else:
        raise ValueError('Unknown dataset: {}'.format(dp))

    print(data_id)
    if data_id not in grouped_data_dict:
        grouped_data_dict[data_id] = []
    grouped_data_dict[data_id].append({
        'image': dp,
        'hospital/scanner': hospital if 'IXI' in dp else scanner,
        'field_strength': field_strength,
        'image_type': image_type,
    })

# split into train/val/test, stratified by scanner/hospital, field_strength and image_type
train_data = []
val_data = []
test_data = []
train_val_ratio = 0.9
for k, v in grouped_data_dict.items():
    if 'IOP' in k:
        # hold out IOP data, unique hospital, 148 volumes
        test_data.extend(v)
    else:
        # stratify by scanner/hospital, field_strength and image_type
        random.shuffle(v)
        train_val_split = int(len(v) * train_val_ratio)
        train_data.extend(v[:train_val_split])
        val_data.extend(v[train_val_split:])

# save to json
with open('train.json', 'w') as f:
    json.dump(train_data, f)
with open('val.json', 'w') as f:
    json.dump(val_data, f)
with open('test.json', 'w') as f:
    json.dump(test_data, f)


