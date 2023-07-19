import tensorflow as tf


def get_train_dataset(data_directory):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_directory,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(200, 200),
        batch_size=4)
    return train_ds


def get_val_dataset(data_directory):
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_directory,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(200, 200),
        batch_size=4)
    return val_ds
