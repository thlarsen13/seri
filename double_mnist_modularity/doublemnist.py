#!/usr/bin/env python
# coding: utf-8

# In[65]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.losses import Loss
import time 
# from tensorflow.keras.applications import *
# from matplotlib import plotly as plt
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE
import pickle
import os


# In[2]:


batch_size = 32
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

N = x_train.shape[0] // 2
x_train = np.concatenate((x_train[:N], x_train[N:]), axis=1)
y_train = np.stack((y_train[:N], y_train[N:]), axis=1)
# y_train = tf.ones(y_train.shape)
N = x_test.shape[0] // 2

x_test = np.concatenate((x_test[:N], x_test[N:]), axis=1)
y_test = np.stack((y_test[:N], y_test[N:]), axis=1)
# y_test = tf.ones(y_test.shape)
print(x_train.shape)
print(y_train.shape)


input_shape = x_train[0].shape

x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

x_test = tf.expand_dims(x_test, -1)
x_train = tf.expand_dims(x_train, -1)
print(x_test.shape)
print(y_test.shape)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# Prepare the validation dataset.
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(batch_size)

input_shape = x_test[0].shape
num_classes = 10
print(input_shape)



def run_experiment(model, model_name, n = 10):
    newpath = f'plots/n={n}/{model_name}/'
    if not os.path.exists(newpath):
        os.makedirs(newpath)    

    optimizer = tf.keras.optimizers.Adam(0.001)

    cce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # loss_fn = cce_loss
    def loss_fn(y, y_hat): 
    #     print(y[:, 0], y_hat[:, :10])
        return cce_loss(y[:, 0], y_hat[:, :10]) + cce_loss(y[:, 1], y_hat[:, 10:])

    #         cce_loss(y_batch_train[:, 0], logits[:, :10]) + cce_loss(y_batch_train[:, 0], logits[:, 10:])



    # In[72]:


    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=loss_fn
    )

    model.fit(
        train_dataset,
        epochs=6,
        validation_data=test_dataset,
    )

    jac_dict = {}

    """
    set up as:
    {
      "layer_0": [[derivative vector for point 0 wrt L_d1], [derivative vector for point 0 wrt L_d2], 
      ..., [derivative vector for point n wrt L_d1], [derivative vector for point n wrt L_d2]],
    .
    .
    .
    {
      "layer_k": [[derivative vector for point 0 wrt L_d1], [derivative vector for point 0 wrt L_d2], 
      ..., [derivative vector for point n wrt L_d1], [derivative vector for point n wrt L_d2]]
    } 
    """


    def jacobian_f(weights, layer_name): 
        print(f'layer_name: {layer_name}')
    #     print(f'weighst: {weights}')
        for i in tqdm(range(n)): 
            x = x_test[i:i+1]
            y = y_test[i:i+1]
            with tf.GradientTape() as t:
                logits = model(x) 
                cce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                cce_loss2 = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

                loss1 = cce_loss(y[:, 0], logits[:, :10]) 
                loss2 = cce_loss2(y[:, 1], logits[:, 10:])
                loss = tf.convert_to_tensor([loss1, loss2])
            jac = t.jacobian(loss, weights)

            
    #         print(jac[0].shape)
    #         break
         
            jac_dict[name].append(jac[0])

        with open(f'{newpath}/jac_dict.pickle', 'wb') as handle:
            pickle.dump(jac_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


    #         if jac1 != None: 
    #             print(jac1[0].shape)
    #             print(tf.reshape(jac1[0], (-1, jac1[0].shape[-1])).shape)

    for layer in model.layers: 
        name = layer.name 
        weights = layer.trainable_weights
        if weights != []:
            jac_dict[name] = []
            jacobian_f(weights, name)


    with open('jac_dict.pickle', 'wb') as handle:
        pickle.dump(jac_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    plt.rcParams['figure.figsize'] = [12, 8]

    for layer in model.layers: 
        name = layer.name 
        weights = layer.trainable_weights
        if weights != []:
            grads = jac_dict[name]
        #     print(len(grads))
        #     break
            for observable in ['saliency', 'jacobian']:
                X = np.array([g for g in grads])[:n] 
                assert(X.shape[1] == 2)
                if observable == 'saliency':
                    X = X * weights[0].numpy()

                X = X.reshape(n, 2, -1)
                X = X.reshape(n*2, -1)

                Y = y_test[:n].numpy().reshape(-1).astype(np.int32)

                X_embedded = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(X)

                labels = [i % 2 for i in range(2*n)]
                plt.figure()
                plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c = labels)
                for i in range(0, len(X_embedded), 2):
                    plt.plot(X_embedded[i:i+2, 0], X_embedded[i:i+2, 1], '-', lw=.4)
                for i in range(X_embedded.shape[0]):
                    plt.text(X_embedded[i,0], X_embedded[i,1], str(Y[i]))

             

                plt.savefig(f'{newpath}/{str(name)}_{observable}')
                # plt.show()


dilation_rate = 1
model1 = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(64, kernel_size=(3, 3), dilation_rate=dilation_rate, padding="same", activation="relu"),
        layers.Conv2D(64, kernel_size=(3, 3), dilation_rate=dilation_rate, padding="same", activation="relu"),
        layers.Conv2D(32, kernel_size=(3, 3), dilation_rate=dilation_rate, padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, kernel_size=(3, 3), dilation_rate=dilation_rate, padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(50, activation="relu"),
        layers.Dense(2*num_classes, activation=None),
    ]
)

model1_name = '3_3_filters_same_padding_6_layers'

dilation_rate = 2
model2 = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(64, kernel_size=(3, 3), dilation_rate=dilation_rate, padding="same", activation="relu"),
        layers.Conv2D(64, kernel_size=(3, 3), dilation_rate=dilation_rate, padding="same", activation="relu"),
        layers.Conv2D(32, kernel_size=(3, 3), dilation_rate=dilation_rate, padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, kernel_size=(3, 3), dilation_rate=dilation_rate, padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(50, activation="relu"),
        layers.Dense(2*num_classes, activation=None),
    ]
)

model2_name = '3_3_filters_same_padding_6_layers_2_dilation'

strides= 2

model3 = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(64, kernel_size=(3, 3), strides=strides, padding="same", activation="relu"),
        # layers.Conv2D(64, kernel_size=(3, 3), dilation_rate=dilation_rate, padding="same", activation="relu"),
        # layers.Conv2D(32, kernel_size=(3, 3), dilation_rate=dilation_rate, padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, kernel_size=(3, 3), strides=strides, padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(50, activation="relu"),
        layers.Dense(2*num_classes, activation=None),
    ]
)

model3_name = '3_3_filters_same_padding_4_layers_2_stride'


strides=1
model4 = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(64, kernel_size=(3, 3), strides=strides, padding="same", activation="relu"),
        # layers.Conv2D(64, kernel_size=(3, 3), dilation_rate=dilation_rate, padding="same", activation="relu"),
        # layers.Conv2D(32, kernel_size=(3, 3), dilation_rate=dilation_rate, padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, kernel_size=(3, 3), strides=strides, padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(50, activation="relu"),
        layers.Dense(2*num_classes, activation=None),
    ]
)

model4_name = '3_3_filters_same_padding_4_layers'

model5 = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(64, kernel_size=(5, 5), strides=strides, padding="same", activation="relu"),
        layers.Conv2D(64, kernel_size=(5, 5), dilation_rate=dilation_rate, padding="same", activation="relu"),
        layers.Conv2D(32, kernel_size=(5, 5), dilation_rate=dilation_rate, padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, kernel_size=(5, 5), strides=strides, padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(50, activation="relu"),
        layers.Dense(2*num_classes, activation=None),
    ]
)

model5_name = '5_5_filters_same_padding_6_layers'


models = [
            [model1, model1_name], 
            [model2, model2_name], 
            [model3, model3_name], 
            [model4, model4_name], 
            [model5, model5_name], 

            ]
for model, name in tqdm(models): 
    run_experiment(model, name, n=500)



