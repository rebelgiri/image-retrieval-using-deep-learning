from tensorflow import keras
import argparse
import tensorflow as tf
import tensorboard
import random
import tensorflow_datasets as tfds
import torch
from torch.utils.tensorboard import SummaryWriter



input_shape = (28, 28, 1)
input_reshape = [1, 28, 28, 1]

from datetime import datetime


writer = SummaryWriter()




def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label

    
def get_dataset(dataset_name):

    (ds_train, ds_val, ds_test), ds_info = tfds.load(dataset_name, data_dir='.',
                                             split=['train', 'train[0:10000]', 'test'], shuffle_files=True,
                                             as_supervised=True,
                                             with_info=True)

    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(2)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.cache()
    ds_test = ds_test.shuffle(ds_info.splits['test'].num_examples)
    #ds_test = ds_test.batch(2)
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)


    ds_val = ds_val.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_val = ds_val.cache()
    ds_val = ds_val.shuffle(ds_info.splits['test'].num_examples)
    ds_val = ds_val.batch(2)
    ds_val = ds_val.prefetch(tf.data.AUTOTUNE)

    return (ds_train, ds_val, ds_test), ds_info

def main():

    model = tf.keras.models.load_model('model_20230308-205254.h5')
    model.summary()

    (ds_train, ds_val, ds_test), ds_info = get_dataset('fashion_mnist')
    ds_test = ds_test.batch(100)

    for i in range(1):
        # testing
        # accumulate tensors for embeddings visualization
        test_imgs = []
        test_targets = []
        hash_embeddings = []
        embeddings = []
        for images, labels in ds_test:
            logits = model(images)
            # show all images that consist the pairs
            test_imgs.extend([logits[0:5]])
            test_targets.extend([labels[0:5]])

            # embedding1: hamming space embedding
            hash = tf.round(tf.clip_by_value(logits, clip_value_min=-1, clip_value_max=1) * 0.5 + 0.5)
            hash_embeddings.extend([hash[0:5]])

            # emgedding2: raw embedding
            embeddings.extend([logits[0:5]])

        writer.add_histogram(
                'embedding_distribution',
                tf.concat(embeddings, 0).numpy(),
                global_step=i + 1)

        # draw embeddings for a single batch - very nice for visualizing clusters
        writer.add_embedding(
                tf.concat(hash_embeddings, 0),
                metadata=tf.concat(test_targets, 0),
                label_img=tf.concat(test_imgs, 0),
                global_step=i + 1)


        writer.close()


            
if __name__ == "__main__":
    main()
