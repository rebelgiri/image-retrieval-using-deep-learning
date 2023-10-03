import collections
import pathlib

import tensorflow as tf

from keras import layers
from keras import losses
from keras import utils
from keras.layers import TextVectorization

import tensorflow_datasets as tfds



def pre_process(data, label):
    print(data.shape)
    print(label.shape)
    return data, label


data_url = 'https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz'

dataset_dir = utils.get_file(
    origin=data_url,
    untar=True,
    cache_dir='stack_overflow',
    cache_subdir='')

dataset_dir = pathlib.Path(dataset_dir).parent


train_dir = dataset_dir/'train'
list(train_dir.iterdir())


batch_size = 32
seed = 42

raw_train_ds = utils.text_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed)




raw_train_ds = raw_train_ds.map(pre_process)








