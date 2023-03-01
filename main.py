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
total_epochs = 100
input_shape = (28, 28, 1)
input_reshape = [1, 28, 28, 1]
lr_init = 3e-4
optimizer = keras.optimizers.Adam(learning_rate=lr_init)
ALPHA = 0.01
# writer = tf.summary.SummaryWriter

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

    # conv3 = keras.layers.Conv2D(filters=64, kernel_size=5, activation='relu',
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
def run_step(model, images, labels, similarity, epoch):

    with tf.GradientTape() as tape:
        b1 = model(tf.reshape(images[0], input_reshape))
        b2 = model(tf.reshape(images[1], input_reshape))

        squared_loss = mse(b1, b2)

        # T1: 0.5 * (1 - y) * dist(x1, x2)
        positive_pair_loss = (0.5 * (1 - similarity) * squared_loss)
        mean_positive_pair_loss = tf.reduce_mean(positive_pair_loss)

        # T2: 0.5 * y * max(margin - dist(x1, x2), 0)
        zeros = tf.zeros_like(squared_loss)
        margin = MARGIN * tf.ones_like(squared_loss)
        negative_pair_loss = 0.5 * similarity * tf.math.maximum(zeros, margin - squared_loss)
        mean_negative_pair_loss = tf.reduce_mean(negative_pair_loss)

        # T3: alpha(dst_l1(abs(x1), 1)) + dist_l1(abs(x2), 1)))
        mean_value_regularization = ALPHA * (
                mae(tf.abs(b1), tf.ones_like(b1)) +
                mae(tf.abs(b2), tf.ones_like(b2)))

        loss = mean_positive_pair_loss + mean_negative_pair_loss + mean_value_regularization


        grads = tape.gradient(loss, model.trainable_weights)

        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        
        
        return loss, mean_positive_pair_loss, mean_negative_pair_loss, mean_value_regularization

        # writer.add_scalar('loss', loss.item(), epoch)
        # writer.add_scalar('positive_pair_loss', mean_positive_pair_loss.item(), epoch)
        # writer.add_scalar('negative_pair_loss', mean_negative_pair_loss.item(), epoch)
        # writer.add_scalar('regularizer_loss', mean_value_regularization.item(), epoch)


def main():
 
    (ds_train, ds_val, ds_test), ds_info = tfds.load('fashion_mnist', data_dir='.',
                                             split=['train', 'train[0:10000]', 'test'], shuffle_files=True,
                                             as_supervised=True,
                                             with_info=True)
    model = get_model(input_shape, code_length)
    model.summary()

    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(2)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)


    ds_val = ds_val.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_val = ds_val.cache()
    ds_val = ds_val.shuffle(ds_info.splits['train'].num_examples)
    ds_val = ds_val.batch(2)
    ds_val = ds_val.prefetch(tf.data.AUTOTUNE)

    for epoch in range(total_epochs):
        
        print("Start of epoch {}".format(epoch))
        total_loss = 0
        total_positive_pair_loss = 0  
        total_negative_pair_loss = 0
        total_mean_value_regularization = 0
    
        for images, labels in ds_train:
            similarity = 0 if labels[0] == labels[1] else 1
            loss, positive_pair_loss, negative_pair_loss, mean_value_regularization = run_step(
                model, images, labels, similarity, epoch)

            total_loss = loss + total_loss
            total_positive_pair_loss = positive_pair_loss + total_positive_pair_loss  
            total_negative_pair_loss = negative_pair_loss + total_negative_pair_loss
            total_mean_value_regularization = mean_value_regularization + total_mean_value_regularization

        print('epoch: {}-> loss: {}, positive_loss: {}, negative_loss: {}, regularize_loss: {}'.format(
        epoch,
        total_loss / (len(ds_train) / 2),
        total_positive_pair_loss / (len(ds_train) / 2),
        total_negative_pair_loss / (len(ds_train) / 2),
        total_mean_value_regularization / (len(ds_train) / 2)
        ))

    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)


    # writer.close()


if __name__ == "__main__":
    # args = parse_args()
    main()
