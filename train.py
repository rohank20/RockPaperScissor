import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

from dataset import get_train_dataset, get_val_dataset
from model import get_model
from utils import f1_score

exp_name = 'rps_exp_1'
data_directory = 'dataset'


def train(train_dataset, val_dataset, epochs=15):
    model = get_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy', f1_score])

    result = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs
    )

    return model, result


def evaluate_training(result):
    history_frame = pd.DataFrame(result.history)
    history_frame.loc[:, ['loss', 'val_loss']].plot()
    history_frame.loc[:, ['accuracy', 'val_accuracy']].plot();
    plt.subplot()
    plt.show()
    val_accuracy = history_frame.val_accuracy.iloc[[-1]]
    print(f'Last validation accuracy: {val_accuracy}')


def save_model(model, exp_name):
    path = f'saved_models/{exp_name}'
    model.save(path)
    print(f'Model saved at path: {path}')


def main():
    train_dataset = get_train_dataset(data_directory)
    val_dataset = get_val_dataset(data_directory)
    model, result = train(train_dataset, val_dataset)
    evaluate_training(result)
    save_model(model, exp_name)


if __name__ == '__main__':
    main()
