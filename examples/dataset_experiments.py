import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# ds = tfds.load('fashion_mnist', data_dir='.', split=['train', 'test'], shuffle_files=True)
# assert isinstance(ds, tf.data.Dataset)
# print(ds)


# builder = tfds.builder('fashion_mnist')
# builder.download_and_prepare()
# ds = builder.as_dataset(split='train', shuffle_files=True)
# print(ds)


import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.set_printoptions(precision=4)


dataset = tf.data.Dataset.from_tensor_slices([8, 3, 0, 8, 2, 1])
print(dataset)


print(dataset.reduce(5, lambda state, value: state + value).numpy())

dataset1 = tf.data.Dataset.from_tensor_slices(tf.random.uniform([4, 10]))
print(dataset1.element_spec)


dataset2 = tf.data.Dataset.from_tensor_slices((tf.random.uniform([4, 100]),
    tf.random.uniform([4, 100], maxval=100, dtype=tf.int32)))


for i in dataset2:
    print(i)



dataset3 = tf.data.Dataset.from_tensor_slices((tf.random.uniform([4, 100])))
                                              

                        


