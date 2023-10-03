from tensorflow import keras
import argparse
import tensorflow as tf
import tensorboard
import random
import tensorflow_datasets as tfds

batch_size = 2
momentum = 0.9
weight_decay = 0.004
code_length = 8
MARGIN = 2 * code_length
total_epochs =  1000
input_shape = (28, 28, 1)
input_reshape = [1, 28, 28, 1]
lr_init = 3e-4
optimizer = keras.optimizers.Adam(learning_rate=lr_init)
ALPHA = 0.01
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = "logs/scalars/" + timestamp

writer = tf.summary.create_file_writer(logdir)

mse = tf.keras.losses.MeanSquaredError()
mae = tf.keras.losses.MeanAbsoluteError()

def parse_args():
    parser = argparse.ArgumentParser(description='List of arguments to run this script:')
    parser.add_argument('--model_name', type=str,
                        default='IRDL', help='The name of the model.')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='The path of the training and testing dataset.')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='The path of the output directory.')
    parser.add_argument('--batch_size', type=int, default=2, help='The batch size.')
    
    parser.add_argument('--margin', type=int, help='Margin')
    parser.add_argument('--code_length', type=int, help='Code length')
    parser.add_argument('--momentum', type=float, help='')
    parser.add_argument('--weight_decay', type=float, help='')

    parser.add_argument('--total_epochs', type=float, help='The number of training Epochs.')
    args = parser.parse_args()
    return args

def get_model(input_shape, code_length):
    '''
    Please refer paper Deep Supervised Hashing 
    for Fast Image Retrieval [here](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Liu_Deep_Supervised_Hashing_CVPR_2016_paper.pdf).
    The default kernel initializer is Xavier uniform initializer.
    '''
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv2D(filters=32, kernel_size=5, activation='relu',
                                data_format="channels_last")(input_layer)
    max_pooling = keras.layers.MaxPooling2D(pool_size=3, strides=2)(conv1)

    conv2 = keras.layers.Conv2D(filters=32, kernel_size=5, activation='relu',
                                data_format="channels_last",)(max_pooling)
    average_pooling1 = keras.layers.AveragePooling2D(
        pool_size=3, strides=2)(conv2)

    # conv3 = keras.layers.Conv2D(filters=32, kernel_size=5, activation='relu',
    #                           data_format="channels_last")(average_pooling1)
    # average_pooling2 = keras.layers.AveragePooling2D(pool_size=3, strides=2)(conv3)

    flatten = keras.layers.Flatten()(average_pooling1)

    dense1 = keras.layers.Dense(500, activation='relu')(flatten)
    dense2 = keras.layers.Dense(code_length)(dense1)

    return keras.models.Model(inputs=input_layer, outputs=dense2)

def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label

@tf.function
def run_step(model, images, similarity, ):
    with tf.GradientTape() as tape:
        b1 = model(tf.reshape(images[0], input_reshape))
        b2 = model(tf.reshape(images[1], input_reshape))

        squared_loss = mse(b1, b2)

        # T1: 0.5 * (1 - y) * dist(x1, x2)
        positive_pair_loss = (0.5 * (1 - similarity) * squared_loss)
        # mean_positive_pair_loss = tf.reduce_mean(positive_pair_loss)

        # T2: 0.5 * y * max(margin - dist(x1, x2), 0)
        zeros = tf.zeros_like(squared_loss)
        margin = MARGIN * tf.ones_like(squared_loss)
        negative_pair_loss = 0.5 * similarity * tf.math.maximum(zeros, margin - squared_loss)
        # mean_negative_pair_loss = tf.reduce_mean(negative_pair_loss)

        # T3: alpha(dst_l1(abs(x1), 1)) + dist_l1(abs(x2), 1)))
        value_regularization = ALPHA * (
                mae(tf.abs(b1), tf.ones_like(b1)) +
                mae(tf.abs(b2), tf.ones_like(b2)))

        loss = positive_pair_loss + negative_pair_loss + value_regularization

    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    
    
    return loss, positive_pair_loss, negative_pair_loss, value_regularization

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
    ds_test = ds_test.batch(2)
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)


    ds_val = ds_val.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_val = ds_val.cache()
    ds_val = ds_val.shuffle(ds_info.splits['test'].num_examples)
    ds_val = ds_val.batch(2)
    ds_val = ds_val.prefetch(tf.data.AUTOTUNE)

    return (ds_train, ds_val, ds_test), ds_info

def main():
    model = get_model(input_shape, code_length)
    model.summary()

    (ds_train, ds_val, ds_test), ds_info = get_dataset('fashion_mnist')
    
    with writer.as_default():
        for i in range(total_epochs):
            # training
            train_loss = 0
            train_positive_pair_loss = 0
            train_negative_pair_loss = 0
            train_value_regularization = 0

            for images, labels in ds_train:
                similarity = 0 if labels[0] == labels[1] else 1
                loss, positive_pair_loss, negative_pair_loss, \
                    value_regularization = run_step(model, images, similarity)

                train_loss = train_loss + loss
                train_positive_pair_loss = train_positive_pair_loss + positive_pair_loss
                train_negative_pair_loss = train_negative_pair_loss + negative_pair_loss
                train_value_regularization = train_value_regularization + value_regularization


            train_loss = train_loss / (len(ds_train) / 2)
            train_positive_pair_loss = train_positive_pair_loss / (len(ds_train) / 2)
            train_negative_pair_loss = train_negative_pair_loss / (len(ds_train) / 2)
            train_value_regularization = train_value_regularization / (len(ds_train) / 2)

            print('Epoch {} -> Training Stage -> Loss : {}, Positive Pair Loss: {} \
                  Negative Pair Loss: {}, Regularization Value: {}'.format(i + 1,
                train_loss, train_positive_pair_loss, train_negative_pair_loss, train_value_regularization))
            
            # log them to tensorboard
            tf.summary.scalar('loss', train_loss, i + 1)
            tf.summary.scalar('positive_pair_loss', train_positive_pair_loss, i + 1)
            tf.summary.scalar('negative_pair_loss', train_negative_pair_loss, i + 1)
            tf.summary.scalar('regularizer_loss', train_value_regularization, i + 1)
            
            # validation
            val_loss = 0
            val_positive_pair_loss = 0
            val_negative_pair_loss = 0
            val_value_regularization = 0

            for images, labels in ds_val:
                similarity = 0 if labels[0] == labels[1] else 1
                loss, positive_pair_loss, negative_pair_loss, \
                    value_regularization = run_step(model, images, similarity)

                val_loss = val_loss + loss
                val_positive_pair_loss = val_positive_pair_loss + positive_pair_loss
                val_negative_pair_loss = val_negative_pair_loss + negative_pair_loss
                val_value_regularization = val_value_regularization + value_regularization


            val_loss = val_loss / (len(ds_val) / 2)
            val_positive_pair_loss = val_positive_pair_loss / (len(ds_val) / 2)
            val_negative_pair_loss = val_negative_pair_loss / (len(ds_val) / 2)
            val_value_regularization = val_value_regularization / (len(ds_val) / 2)

            # log them to tensorboard
            tf.summary.scalar('val_loss', val_loss, i + 1)
            tf.summary.scalar('val_positive_pair_loss', val_positive_pair_loss, i + 1)
            tf.summary.scalar('val_negative_pair_loss', val_negative_pair_loss, i + 1)
            tf.summary.scalar('val_regularizer_loss', val_value_regularization, i + 1)

            print('Epoch {} -> Validation Stage -> Loss : {}, Positive Pair Loss: {} \
                  Negative Pair Loss: {}, Regularization Value: {}'.format(i + 1,
                val_loss, val_positive_pair_loss, val_negative_pair_loss, val_value_regularization))

            writer.flush()


            model.save('model_' + timestamp + '.h5')


            
if __name__ == "__main__":
    main()
