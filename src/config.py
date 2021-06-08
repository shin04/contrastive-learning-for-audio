device = 'cuda'

audio_path = '/ml/dataset/audio/balanced_train_segments'
# metadata_path = '/ml/dataset/meta_data/balanced_train_segments.csv'
metadata_path = '/ml/meta.csv'

n_epoch = 400000
batch_size = 16
lr = 0.001  # to 0.00001
audio_crop_size = 3  # sec
temperature = 0.1
