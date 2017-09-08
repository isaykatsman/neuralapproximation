#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics, preprocessing, linear_model
from sklearn.metrics import log_loss
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import CSVLogger
import os, sys
import tensorflow as tf
import random

exp_name = 'tournament_'+sys.argv[1]
results_folder = 'results/'+exp_name+'/'
step = 50
bottom_range = list(range(0, step))
mid_range = list(range(step, 2*step))
top_range = list(range(2*step, 3*step))

def gen_func(x):
    return 50*math.sin(x/10)

# setup folders
# setup experiment directory
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# setup other folders
if not os.path.exists(results_folder+'plots'):
    os.makedirs(results_folder+'plots')

# Set seed for reproducibility
# np.random.seed(0)

print("Loading data...")

# setup x and y training data
#x_train_old = [float(x) for x in range(step)]
#x_train = random.sample(set(x_train_old), int(step*3/5))
x_train = [float(x) for x in bottom_range]
x_train.extend([float(x) for x in top_range])
# x_comp = list(set(x_train_old) - set(x_train))
# # write out to file
# pred_file_vals = open('intermediatevals.txt','w')
# pred_file_vals.write("actual training: \n")
# for i in x_train:
#     pred_file_vals.write("%f " % i)
# pred_file_vals.write("\n Testing intermed (effectively): \n")
# for i in x_comp:
#     pred_file_vals.write("%f " % i)
# pred_file_vals.write("\n")

# pred_file_vals.close()

y_train = [x**2 for x in x_train]

# setup x and y validation data
x_val = [float(x) for x in mid_range]
y_val = [gen_func(x) for x in x_val]

# custom loss
def custom_loss(y_true, y_pred):
    return tf.norm(y_true-y_pred)

# binary classification deep NN
model = Sequential()
model.add(Dense(100, input_dim=1, activation='relu'))
#model.add(Dense(200, activation='relu'))
#model.add(Dense(200, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))
#model.add(Dense(100, input_dim=1, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))
#model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss=custom_loss,#'binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# use a CSV logger to continuously write out a log
csv_logger = CSVLogger(results_folder+'exp_log.csv', append=True, separator=';')

# save model json
model_json = model.to_json()
with open("main_model.json", "w") as json_file:
    json_file.write(model_json)

# train
history = model.fit(x_train,
                    y_train,
                    validation_data=(x_val, y_val),
                    epochs=25000,#5000,
                    batch_size=10000,
                    callbacks=[csv_logger],
                    shuffle=True)

# save plots
#plot history for just training loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training Loss','Validation Loss'], loc='upper right')
plt.savefig(results_folder+'plots/reconstructed_loss.png')
plt.clf()

#plot history for classification 1 - fooling ratio
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training Classification Accuracy','Validation Classification Accuracy'], loc='upper right')
plt.savefig(results_folder+'plots/reconstructed_acc.png')
plt.clf()
print("Saved plots.")

# predict on test data, will be submitted to numerai
# test data (only have x, numerai will evaluate on y)
x_pred = [float(x) for x in bottom_range + mid_range + top_range]
y_pred = model.predict(x_pred)

# write out to file
pred_file = open('preds.txt','w')
for i,j in zip(x_pred, y_pred):
    pred_file.write("%f %f\n" % (i,j))
pred_file.close()

# graph matplot lib
x_axis = [x for x in bottom_range + mid_range + top_range]
perf = [gen_func(x) for x in bottom_range + mid_range + top_range]
plt.plot(y_pred)
plt.plot(perf)
plt.title('Function Approximation')
plt.legend(['Y predicted', 'Y perfect (original)'], loc='upper right')
plt.savefig('main_pred.png')
plt.clf()
