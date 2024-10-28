import gin
import logging
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
import numpy as np
from input_pipeline.preprocessing import preprocess, augment
from pathlib import Path
from tensorflow.data.experimental import TFRecordWriter
from tensorflow.train import BytesList, FloatList, Int64List
from tensorflow.train import Example, Features, Feature
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from scipy import stats


@gin.configurable
def load(name, data_dir):
    if name == "hapt":

        logging.info(f"Preparing dataset {name}...")
        # ...
        # directories
        directory_data = data_dir + 'HAPT_dataset/RawData/'
        tfrecord_train = 'ds_train.tfrecord'
        tfrecord_val = 'ds_val.tfrecord'
        tfrecord_test = 'ds_test.tfrecord'

        # Check if TFRecord Files need to be created
        if Path(tfrecord_test).exists() and Path(tfrecord_train).exists() and Path(tfrecord_val).exists():
        # if Path(tfrecord_val).exists():
            print('TFRecord Files already exist.')
        else:
            print('TFRecord Files will be created...')

            # load all files and convert to float/int
            data = []
            files = [f for f in listdir(directory_data) if isfile(join(directory_data, f))]
            for idx, filename in enumerate(files):
                file = open(directory_data + filename)
                if filename == "labels.txt":
                    labels = file.readlines()
                    for index in range(len(labels)):
                        labels[index] = labels[index].rstrip("\n")
                        labels[index] = list(map(int, labels[index].split()))
                else:
                    data.append(file.readlines())
                    for index in range(len(data[idx])):
                        data[idx][index] = data[idx][index].rstrip("\n")
                        data[idx][index] = list(map(float, data[idx][index].split()))
                file.close()

            # concatenate acc and gyro vectors
            concatenation = []
            for i in range(int(len(data) / 2)):
                concatenation.append([])
                for j in range(len(data[i])):
                    concatenation[i].append([*data[i][j], *data[i + int(len(data) / 2)][j]])

            # create useful labels
            label = []
            for exp in range(len(concatenation)):
                label.append(np.zeros(len(concatenation[exp]), dtype=int))
                for idx in range(len(labels)):
                    if labels[idx][0] - 1 == exp:
                        for i in range(labels[idx][3] - 1, labels[idx][4] - 1):
                            label[exp][i] = labels[idx][2]

            # create training lists
            label_train = label[0]
            for exp in range(1, 43):
                label_train = np.concatenate((label_train, label[exp]), axis=None, dtype=int)
            data_train = concatenation[0]
            for exp in range(1, 43):
                data_train = np.concatenate((data_train, concatenation[exp]))

            # create test lists
            label_test = label[43]
            for exp in range(44, 55):
                label_test = np.concatenate((label_test, label[exp]), axis=None, dtype=int)
            data_test = concatenation[43]
            for exp in range(44, 55):
                data_test = np.concatenate((data_test, concatenation[exp]))

            # create validation lists
            label_val = label[55]
            for exp in range(56, 61):
                label_val = np.concatenate((label_val, label[exp]), axis=None, dtype=int)
            data_val = concatenation[55]
            for exp in range(56, 61):
                data_val = np.concatenate((data_val, concatenation[exp]))

            # datasets pre sort out
            ds_train = tf.data.Dataset.from_tensor_slices((data_train, label_train))
            ds_val = tf.data.Dataset.from_tensor_slices((data_val, label_val))
            ds_test = tf.data.Dataset.from_tensor_slices((data_test, label_test))

            # windows pre sort out
            ds_train_window = ds_train.window(size=250, shift=125, drop_remainder=True)  # 6165 windows
            ds_val_window = ds_val.window(size=250, shift=125, drop_remainder=True)  # 947 windows
            ds_test_window = ds_test.window(size=250, shift=125, drop_remainder=True)  # 1866 windows

            print("Checkpoint")
            # 0 -> lose 3282 all -> lose 3295
            # sort out unusable windows
            datasets = [ds_train_window, ds_val_window, ds_test_window]
            # datasets = [ds_val_window]
            for dataset in datasets:
                # skip = []
                data_sorted = None
                label_sorted = None
                for idx, window in enumerate(dataset):
                    label_list = (list(window[1].as_numpy_iterator()))
                    if 0 in label_list or 7 in label_list or 8 in label_list or 9 in label_list or 10 in label_list \
                            or 11 in label_list or 12 in label_list:
                        pass
                    elif len(set(label_list)) > 1:  # sort out when multiple classes in one window
                        pass
                    else:
                        dataset_list = list(window[0].as_numpy_iterator())

                        if label_sorted is None:
                            label_sorted = label_list
                        else:
                            label_sorted = np.concatenate((label_sorted, label_list), axis=None, dtype=int)
                        if data_sorted is None:
                            data_sorted = dataset_list
                        else:
                            data_sorted = np.concatenate((data_sorted, dataset_list))

                # z-standardization for each column
                standardized = []
                for column in range(len(data_sorted[0])):
                    col_array = []
                    for element in data_sorted:
                        col_array.append(element[column])
                    standardized_col = stats.zscore(col_array)
                    standardized.append(standardized_col)

                standardized_zip = [list(a) for a in zip(standardized[0], standardized[1], standardized[2],
                                                         standardized[3], standardized[4], standardized[5])]

                sequence_to_label_data = []
                sequence_to_label_label = []
                for i in range(int(len(standardized_zip) / 250)):
                    sequence_to_label_data.append(np.row_stack(standardized_zip[i*250:250*(i+1)]))
                    sequence_to_label_label.append(label_sorted[i*250])

                if dataset == ds_train_window:
                    ds_train = tf.data.Dataset.from_tensor_slices((sequence_to_label_data, sequence_to_label_label))
                elif dataset == ds_val_window:
                    ds_val = tf.data.Dataset.from_tensor_slices((sequence_to_label_data, sequence_to_label_label))
                elif dataset == ds_test_window:
                    ds_test = tf.data.Dataset.from_tensor_slices((sequence_to_label_data, sequence_to_label_label))

            def _bytes_feature(value):
                """Returns a bytes_list from a string / byte."""
                if isinstance(value, type(tf.constant(0))):
                    value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
                return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

            def _float_feature(value):
                return tf.train.Feature(float_list=tf.train.FloatList(value=value))

            def _int64_feature(value):
                """Returns an int64_list from a bool / enum / int / uint."""
                return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

            def serialize_example(data, label):
                """
              Creates a tf.train.Example message ready to be written to a file.
              """
                # Create a dictionary mapping the feature name to the tf.train.Example-compatible
                # data type.
                feature = {
                    'data': _float_feature(np.reshape(data.numpy(), (-1))),
                    'label': _int64_feature(int(label.numpy())),
                }

                # Create a Features message using tf.train.Example.
                example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
                return example_proto.SerializeToString()

            def tf_serialize_example(data, label):
                tf_string = tf.py_function(
                    serialize_example,
                    (data, label),
                    tf.string)
                return tf.reshape(tf_string, ())

            serialized_ds_train = ds_train.map(tf_serialize_example)
            serialized_ds_val = ds_val.map(tf_serialize_example)
            serialized_ds_test = ds_test.map(tf_serialize_example)

            def generator_train():
                for features in ds_train:
                    yield serialize_example(*features)

            serialized_ds_train = tf.data.Dataset.from_generator(
                generator_train, output_types=tf.string, output_shapes=())

            def generator_val():
                for features in ds_val:
                    yield serialize_example(*features)

            serialized_ds_val = tf.data.Dataset.from_generator(
                generator_val, output_types=tf.string, output_shapes=())

            def generator_test():
                for features in ds_test:
                    yield serialize_example(*features)

            serialized_ds_test = tf.data.Dataset.from_generator(
                generator_test, output_types=tf.string, output_shapes=())

            writer = tf.data.experimental.TFRecordWriter(tfrecord_train)
            writer.write(serialized_ds_train)

            writer = tf.data.experimental.TFRecordWriter(tfrecord_val)
            writer.write(serialized_ds_val)

            writer = tf.data.experimental.TFRecordWriter(tfrecord_test)
            writer.write(serialized_ds_test)

        # parse func for TFRecord decoding
        # read TFRecords files
        raw_ds_train = tf.data.TFRecordDataset(tfrecord_train)
        raw_ds_val = tf.data.TFRecordDataset(tfrecord_val)
        raw_ds_test = tf.data.TFRecordDataset(tfrecord_test)

        # Create a dictionary describing the features.
        data_feature_description = {
            'data': tf.io.VarLenFeature(dtype=tf.float32),
            'label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        }

        def _parse_image_function(example_proto):
            # Parse the input tf.train.Example proto using the dictionary above.
            return tf.io.parse_single_example(example_proto, data_feature_description)

        parsed_ds_train = raw_ds_train.map(_parse_image_function)
        parsed_ds_val = raw_ds_val.map(_parse_image_function)
        parsed_ds_test = raw_ds_test.map(_parse_image_function)

        @tf.function
        def decode_images(feature):
            data = tf.sparse.to_dense(feature["data"], default_value=0)
            data = tf.reshape(data, [250, 6])
            labels = feature['label']
            return data, labels

        ds_train = parsed_ds_train.map(decode_images)
        ds_val = parsed_ds_val.map(decode_images)
        ds_test = parsed_ds_test.map(decode_images)

        # # display data !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
        # labels = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]
        # for idx, (data, label) in enumerate(ds_train.take(200)):
        #     fig, (ax1, ax2) = plt.subplots(1, 2)
        #     ax1.set_title('Acc')
        #     ax2.set_title('Gyro')
        #     fig.suptitle('Label = ' + str(labels[label.numpy()-1]))
        #     ax1.set_ylim([-3, 3])
        #     ax2.set_ylim([-3, 3])
        #     acc_x = data[:, 0]
        #     acc_y = data[:, 1]
        #     acc_z = data[:, 2]
        #     gyro_x = data[:, 3]
        #     gyro_y = data[:, 4]
        #     gyro_z = data[:, 5]
        #     time = np.linspace(0, 4.16666, 250)
        #
        #     ax1.plot(time, acc_x, label="acc_x")
        #     ax1.plot(time, acc_y, label="acc_y")
        #     ax1.plot(time, acc_z, label="acc_z")
        #     ax2.plot(time, gyro_x, label="gyro_x")
        #     ax2.plot(time, gyro_y, label="gyro_y")
        #     ax2.plot(time, gyro_z, label="gyro_z")
        #     ax1.legend()
        #     ax2.legend()
        #     # print(label)
        #     fig.savefig('data_visualization/image_' + str(idx) + '.jpg', dpi=150)
        # exit()


        ds_train = ds_train.batch(16)
        ds_val = ds_val.batch(16)
        ds_test = ds_test.batch(16)

        ds_train = ds_train.repeat(-1)

        ds_info = {
            "num_examples": 372,
            "image_shape": (256, 256, 3),
            "n_classes": 2,
        }

        return ds_train, ds_val, ds_test, ds_info

#     elif name == "eyepacs":
#         logging.info(f"Preparing dataset {name}...")
#         (ds_train, ds_val, ds_test), ds_info = tfds.load(
#             'diabetic_retinopathy_detection/btgraham-300',
#             split=['train', 'validation', 'test'],
#             shuffle_files=True,
#             with_info=True,
#             data_dir=data_dir
#         )
#
#         # next(iter(ds_train)), list(ds_train)
#         def _preprocess(img_label_dict):
#             return img_label_dict['image'], img_label_dict['label']
#
#         ds_train = ds_train.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#         ds_val = ds_val.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#         ds_test = ds_test.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#
#         ds_info = {
#             "num_examples": 35126,
#             "image_shape": (None, None, 3),
#             "n_classes": 5,
#         }
#         return prepare(ds_train, ds_val, ds_test, ds_info)
#
#     elif name == "mnist":
#         logging.info(f"Preparing dataset {name}...")
#         (ds_train, ds_val, ds_test), ds_info = tfds.load(
#             'mnist',
#             split=['train[:90%]', 'train[90%:]', 'test'],
#             shuffle_files=True,
#             as_supervised=True,
#             with_info=True,
#             data_dir=data_dir
#         )
#         ds_info = {
#             "num_examples": 60000,
#             "image_shape": (28, 28, 1),
#             "n_classes": 10,
#         }
#         return prepare(ds_train, ds_val, ds_test, ds_info)
#
#     else:
#         raise ValueError
#
#
# @gin.configurable
# def prepare(ds_train, ds_val, ds_test, ds_info, batch_size, caching):
#     # Prepare training dataset
#     ds_train = ds_train.map(
#         preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#     if caching:
#         ds_train = ds_train.cache()
#     ds_train = ds_train.map(
#         augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#     # ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples // 10)
#     ds_train = ds_train.shuffle(ds_info["num_examples"] // 10)
#     ds_train = ds_train.batch(batch_size)
#     ds_train = ds_train.repeat(-1)
#     ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
#
#     # Prepare validation dataset
#     ds_val = ds_val.map(
#         preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#     ds_val = ds_val.batch(batch_size)
#     if caching:
#         ds_val = ds_val.cache()
#     ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)
#
#     # Prepare test dataset
#     ds_test = ds_test.map(
#         preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#     ds_test = ds_test.batch(batch_size)
#     if caching:
#         ds_test = ds_test.cache()
#     ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
#
#     return ds_train, ds_val, ds_test, ds_info
