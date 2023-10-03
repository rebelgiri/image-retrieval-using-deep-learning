import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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

    model = tf.keras.models.load_model('model_20230312-105140.h5')
    model.summary()

    (ds_train, ds_val, ds_test), ds_info = get_dataset('fashion_mnist')
    ds_test = ds_test.batch(1)

    hash_array = []
    labels = []
    for image, label in ds_test:
        logits = model(image)
        # Hamming space embedding
        embedding = tf.round(tf.clip_by_value(logits, clip_value_min=-1, clip_value_max=1) * 0.5 + 0.5)
        hash_array = embedding.numpy() if not len(hash_array) else np.vstack((hash_array, embedding.numpy()))
        labels = label.numpy() if not len(labels) else np.vstack((labels, label.numpy()))

    print(hash_array.shape)
        

    # We want to get TSNE embedding with 2 dimensions
    n_components = 2
    tsne = TSNE(n_components)
    tsne_result = tsne.fit_transform(hash_array)
    tsne_result.shape

    # Plot the result of our TSNE with the label color coded
    # A lot of the stuff here is about making the plot look pretty and not TSNE
    tsne_result_df = pd.DataFrame({'x': tsne_result[:,0].flatten(), 'y': tsne_result[:,1].flatten(), 
                                   'label': labels.flatten()})
    fig, ax = plt.subplots(1)
    sns.scatterplot(x='x', y='y', hue='label', data=tsne_result_df, ax=ax, s=120)
    lim = (tsne_result.min()-5, tsne_result.max()+5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    plt.savefig('result.png')
        

    plt.close(fig)




            
if __name__ == "__main__":
    main()
