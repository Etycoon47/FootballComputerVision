import tensorflow as tf
from tensorflow import keras
from testtensorflow import train_dataset, x_val, y_val
import numpy as np
import matplotlib.pyplot as plt
import time

model = keras.models.load_model('models/testmodel')

val_data = np.array([x for x, y in train_dataset])
val_labels = np.array([y for x, y in train_dataset])

predictions = np.argmax(model.predict(val_data[0]), axis=1)
print(predictions)

for i in range(len(val_labels)):
    image = val_data[i]
    label = np.argmax(val_labels[i])
    print(predictions[i], label)
