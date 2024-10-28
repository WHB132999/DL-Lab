import gin
import logging
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
import numpy as np
from input_pipeline.preprocessing import preprocess, augment
from pathlib import Path
import cv2
import glob
import os

# THIS PART WAS CREATED BY SAMUEL BRUCKER

@gin.configurable
def load(name, data_dir_prefix, img_height, img_width, sobel_preprocessing):
    if name == "idrid":
        logging.info(f"Preparing dataset {name}...")

        # TFRecord directories
        tfrecord_train = 'ds_train.tfrecord'
        tfrecord_val = 'ds_val.tfrecord'
        tfrecord_test = 'ds_test.tfrecord'
        tfrecord_sobel_train = 'ds_train_sobel.tfrecord'
        tfrecord_sobel_val = 'ds_val_sobel.tfrecord'
        tfrecord_sobel_test = 'ds_test_sobel.tfrecord'

        # Check if TFRecord Files need to be created
        if sobel_preprocessing is True:
            if Path(tfrecord_sobel_train).exists() and Path(tfrecord_sobel_val).exists() and Path(tfrecord_sobel_test).exists():
                print('TFRecord Files for Sobel Preprocessing already exist.')
                tfrecord_exists = True
            else:
                print('TFRecord Files for Sobel Preprocessing will be created...')
                tfrecord_exists = False
        else:
            directory_preprocessed = 'IDRID_preprocessed/'
            if Path(tfrecord_test).exists() and Path(tfrecord_train).exists() and Path(tfrecord_val).exists():
                print('TFRecord Files for Graham Preprocessing already exist.')
                tfrecord_exists = True
            else:
                print('TFRecord Files for Graham Preprocessing will be created...')
                tfrecord_exists = False
        if tfrecord_exists is False:
            # other Directories
            directory_labels = data_dir_prefix + 'IDRID_dataset/labels/'
            directory_train = data_dir_prefix + 'IDRID_dataset/images/train/'
            directory_test = data_dir_prefix + 'IDRID_dataset/images/test/'
            df_train = pd.read_csv(directory_labels + 'train.csv')
            df_test = pd.read_csv(directory_labels + 'test.csv')
            files_train = df_train['Image name'].values + '.jpg'
            labels_train = df_train['Retinopathy grade'].values
            files_test = df_test['Image name'].values + '.jpg'
            labels_test = df_test['Retinopathy grade'].values

            # manipulate grade values -> 2 classes
            labels_train[labels_train < 2] = 0
            labels_train[labels_train > 1] = 1
            labels_test[labels_test < 2] = 0
            labels_test[labels_test > 1] = 1

            # Graham Preprocessing - Produce preprocessed images & safe in /IDRID_preprocessed/...
            # -> will be used to create tf_record files when sobel preprocessing is disabled
            if sobel_preprocessing is False:
                def scaleRadius(img, scale):
                    x = img[int(img.shape[0] / 2), :, :].sum(1)
                    r = (x < x.mean() / 10).sum() / 2
                    s = scale * 1.0 / r
                    return cv2.resize(img, (0, 0), fx=s, fy=s)

                preprocessed_exist = False
                for _ in Path(directory_preprocessed + 'train/').glob('*'):
                    preprocessed_exist = True
                    break
                if preprocessed_exist is False:
                    print("Graham Preprocessing...")
                    os.mkdir(directory_preprocessed)
                    os.mkdir(directory_preprocessed + 'train/')
                    os.mkdir(directory_preprocessed + 'test/')
                    for f in glob.glob(directory_train + '*.jpg'):
                        image = cv2.imread(f)
                        scale = 300
                        # scale image to given radius
                        image = scaleRadius(image, scale)
                        # subtract local mean color
                        image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), scale / 30), -4, 128)
                        cv2.imwrite(directory_preprocessed + 'train/' + f[-13:], image)
                    for f in glob.glob(directory_test + '*.jpg'):
                        image = cv2.imread(f)
                        scale = 300
                        # scale image to given radius
                        image = scaleRadius(image, scale)
                        # subtract local mean color
                        image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), scale / 30), -4, 128)
                        cv2.imwrite(directory_preprocessed + 'test/' + f[-13:], image)

                path_train = directory_preprocessed + 'train/' + files_train
                path_test = directory_preprocessed + 'test/' + files_test
            else:
                path_train = directory_train + files_train
                path_test = directory_train + files_test

            # shuffle and validation set
            n_training_examples = round(0.9 * len(files_train))
            shuffle_idx = np.arange(len(files_train))
            np.random.shuffle(shuffle_idx)
            path_train = [path_train[i] for i in shuffle_idx]
            labels = [labels_train[i] for i in shuffle_idx]
            path_val = path_train[(n_training_examples + 1):]
            labels_val = labels[(n_training_examples + 1):]
            path_train = path_train[:n_training_examples]
            labels_train = labels[:n_training_examples]

            # datasets
            ds_train = tf.data.Dataset.from_tensor_slices((path_train, labels_train))
            ds_val = tf.data.Dataset.from_tensor_slices((path_val, labels_val))
            ds_test = tf.data.Dataset.from_tensor_slices((path_test, labels_test))

            def parse_func(filename, label):
                image_string = tf.io.read_file(filename)
                image_decoded = tf.io.decode_jpeg(image_string, channels=3)
                image = preprocess(image_decoded, img_height=img_height, img_width=img_width,
                                          sobel_preprocessing=sobel_preprocessing)
                return image, label

            # maps real image data to the previously defined dataset
            ds_train = ds_train.map(parse_func)
            ds_val = ds_val.map(parse_func)
            ds_test = ds_test.map(parse_func)

            # Resample Dataset (https://www.tensorflow.org/guide/data#resampling)
            low_ds_train = ds_train.filter(lambda image, label: tf.reshape(label, []) == 0).repeat(2)
            low_len = len(list(low_ds_train.take(1000)))
            high_ds_train = ds_train.filter(lambda image, label: tf.reshape(label, []) == 1)
            high_len = len(list(high_ds_train.take(500)))
            low_ds_train = low_ds_train.shuffle(low_len)
            low_ds_train = low_ds_train.take(high_len)

            ds_train = tf.data.experimental.sample_from_datasets(
                [low_ds_train, high_ds_train], [0.5, 0.5])

            def _bytes_feature(value):
                """Returns a bytes_list from a string / byte."""
                if isinstance(value, type(tf.constant(0))):
                    value = value.numpy()
                return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

            def _float_feature(value):
                """Returns a float_list from a float / double."""
                return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

            def _int64_feature(value):
                """Returns an int64_list from a bool / enum / int / uint."""
                return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

            def serialize_example(image, label):
                """
              Creates a tf.train.Example message ready to be written to a file.
              """
                # Create a dictionary mapping the feature name to the tf.train.Example-compatible
                # data type.
                feature = {
                    'image': _bytes_feature(bytes(image)),
                    'label': _int64_feature(int(label)),
                }

                # Create a Features message using tf.train.Example.
                example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
                return example_proto.SerializeToString()

            def tf_serialize_example(image, label):
                tf_string = tf.py_function(
                    serialize_example,
                    (image, label),  # Pass these args to the above function.
                    tf.string)  # The return type is `tf.string`.
                return tf.reshape(tf_string, ())  # The result is a scalar.

            serialized_features_ds_train = ds_train.map(tf_serialize_example)
            serialized_features_ds_val = ds_val.map(tf_serialize_example)  # Do I realy need that ????!?!??!?!!?!? try without!!!!!!!!!!!!!!!!!!!
            serialized_features_ds_test = ds_test.map(tf_serialize_example)

            def generator():
                for features in ds_train:
                    yield serialize_example(*features)

            serialized_features_ds_train = tf.data.Dataset.from_generator(
                generator, output_types=tf.string, output_shapes=())

            def generator():
                for features in ds_val:
                    yield serialize_example(*features)

            serialized_features_ds_val = tf.data.Dataset.from_generator(
                generator, output_types=tf.string, output_shapes=())

            def generator():
                for features in ds_test:
                    yield serialize_example(*features)

            serialized_features_ds_test = tf.data.Dataset.from_generator(
                generator, output_types=tf.string, output_shapes=())

            if sobel_preprocessing is False:
                # TFRecord Files writer for Graham Preprocessing
                writer = tf.data.experimental.TFRecordWriter(tfrecord_train)
                writer.write(serialized_features_ds_train)

                writer = tf.data.experimental.TFRecordWriter(tfrecord_val)
                writer.write(serialized_features_ds_val)

                writer = tf.data.experimental.TFRecordWriter(tfrecord_test)
                writer.write(serialized_features_ds_test)
            else:
                # TFRecord Files writer for Sobel Preprocessing
                writer = tf.data.experimental.TFRecordWriter(tfrecord_sobel_train)
                writer.write(serialized_features_ds_train)

                writer = tf.data.experimental.TFRecordWriter(tfrecord_sobel_val)
                writer.write(serialized_features_ds_val)

                writer = tf.data.experimental.TFRecordWriter(tfrecord_sobel_test)
                writer.write(serialized_features_ds_test)

        # parse func for TFRecord decoding
        # read TFRecords files
        if sobel_preprocessing is False:
            raw_ds_train = tf.data.TFRecordDataset(tfrecord_train)
            raw_ds_val = tf.data.TFRecordDataset(tfrecord_val)
            raw_ds_test = tf.data.TFRecordDataset(tfrecord_test)
        else:
            raw_ds_train = tf.data.TFRecordDataset(tfrecord_sobel_train)
            raw_ds_val = tf.data.TFRecordDataset(tfrecord_sobel_val)
            raw_ds_test = tf.data.TFRecordDataset(tfrecord_sobel_test)

        # Create a dictionary describing the features.
        image_feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        }

        def _parse_image_function(example_proto):
            # Parse the input tf.train.Example proto using the dictionary above.
            return tf.io.parse_single_example(example_proto, image_feature_description)

        parsed_ds_train = raw_ds_train.map(_parse_image_function)
        parsed_ds_val = raw_ds_val.map(_parse_image_function)
        parsed_ds_test = raw_ds_test.map(_parse_image_function)

        def decode_images(feature):
            image_ = feature['image']
            image_ = tf.io.decode_raw(image_, tf.float32)
            images_ = tf.reshape(image_,
                                [img_height, img_width, 3])
            labels_ = feature['label']
            labels_ = tf.expand_dims(tf.cast(labels_, tf.float32), axis=-1)
            return images_, labels_

        ds_train = parsed_ds_train.map(decode_images)
        ds_val = parsed_ds_val.map(decode_images)
        ds_test = parsed_ds_test.map(decode_images)

        ds_info = {
            "num_examples": 372,
            "image_shape": (img_height, img_width, 3),
            "n_classes": 2,
            "data_directory_prefix": data_dir_prefix,
            "sobel_preprocessing": sobel_preprocessing
        }
        ds_train, ds_val, ds_test, ds_info = prepare(ds_train, ds_val, ds_test, ds_info)

        return ds_train, ds_val, ds_test, ds_info

    elif name == "eyepacs":
        logging.info(f"Preparing dataset {name}...")
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            'diabetic_retinopathy_detection/btgraham-300',
            split=['train', 'validation', 'test'],
            shuffle_files=True,
            with_info=True,
            data_dir=data_dir_prefix
        )

        def _preprocess(img_label_dict):
            return img_label_dict['image'], img_label_dict['label']

        ds_train = ds_train.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_val = ds_val.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_test = ds_test.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        ds_info = {
            "num_examples": 35126,
            "image_shape": (None, None, 3),
            "n_classes": 5,
        }
        return prepare(ds_train, ds_val, ds_test, ds_info)

    elif name == "mnist":
        logging.info(f"Preparing dataset {name}...")
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            'mnist',
            split=['train[:90%]', 'train[90%:]', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
            data_dir=data_dir_prefix
        )
        ds_info = {
            "num_examples": 60000,
            "image_shape": (28, 28, 1),
            "n_classes": 10,
        }
        return prepare(ds_train, ds_val, ds_test, ds_info)

    else:
        raise ValueError


@gin.configurable
def prepare(ds_train, ds_val, ds_test, ds_info, batch_size, caching):
    # Prepare training dataset
    if caching:
        ds_train = ds_train.cache()
    ds_train = ds_train.map(
        augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.shuffle(ds_info["num_examples"] // 10)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.repeat(-1)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare validation dataset
    ds_val = ds_val.batch(1)
    if caching:
        ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare test dataset
    ds_test = ds_test.batch(1)
    if caching:
        ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_val, ds_test, ds_info