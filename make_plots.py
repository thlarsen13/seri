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
from sklearn.decomposition import PCA
import traceback
import os 

from multiprocessing import Pool
from keras.utils import plot_model 


plt.rcParams['figure.figsize'] = [12, 8]


# from varname import nameof

# from model import models

#activation_space experiment

def activation_space(model, model_name):
    n = 100

    print('starting activation_space')
    k=10
    x = x_test[k:k+n]
    y = y_test[k:k+n]

    print(y_test[k])
    layer_idx = 1
    for i, layer in enumerate(model.layers): 
        print(layer.name)
        if i == 0:
            aux_model1 = lambda x: x
        else: 
            aux_model1 = tf.keras.Model(inputs=model.inputs, outputs=model.layers[i-1].output)
       
        aux_model2 = tf.keras.Model(inputs=layer.input, outputs=model.output)
        jac = [] 
        for idx in range(n):

            with tf.GradientTape() as t:
                hidden = aux_model1(x[idx:idx+1])

                hidden = tf.Variable(hidden)
                logits = aux_model2(hidden) 
                probs = tf.concat((tf.nn.softmax(logits[:, :10]), tf.nn.softmax(logits[:, 10:])), axis=1)
    #             print(f'loss: {cce_loss(y[i, 0], logits[:, :10])},  {cce_loss(y[i, 1], logits[:, 10:])}')
    #             print(f'probs: {probs.shape}')
            j = t.jacobian(logits, hidden) 

            jac.append(j)

        
        jac = np.array(jac)

        jac = jac.reshape(n, 20, jac.size // (20*n))
    #     jac = jac.reshape(20, jac.size // 20)

    #     print(np.linalg.norm(jac, axis=1))
        M = np.einsum('mij,mkj->mik', jac / np.expand_dims(np.linalg.norm(jac, axis=2), axis=2), jac)
        M = M.mean(axis=0) 
        if 'conv' in layer.name: 
            layer_type = "Convolutional"
        else: 
            layer_type = "Fully Connected"
        if layer.weights != []:
            plt.matshow(M)
            plt.title(f'Layer {layer_idx} ({layer_type})')    
            plt.savefig(f'plots/{model_name}/activation_space/{layer_type}_{layer_idx}.png')
            layer_idx += 1
    #     print([M[i, i] for i in range(20)])

def pca_experiment(model, model_name):
    print('starting pca_experiment')

    #PCA experiment
    raise NotImplementedError
    for j in range(5):
        my_dict = {}

        for layer in model.layers:
            if 'conv' in layer.name: 

                aux_model = tf.keras.Model(inputs=model.inputs,
                                           outputs=layer.output)

                # Access both the final and intermediate output of the original model
                # by calling `aux_model.predict()`.

                # we have 6 layers, some principle components, and n data points and n filters. 

                intermediate_layer_output = tf.reduce_mean(aux_model(x_test[:n]), axis=0)
                my_dict[layer.name] = []

        #         print(intermediate_layer_output.shape)

                for i in range(intermediate_layer_output.shape[-1]): 
                    s, u, v = tf.linalg.svd(intermediate_layer_output[:, :, i])            
                    my_dict[layer.name].append(s[j])



        fig, ax = plt.subplots()
        ax.boxplot(my_dict.values())
        ax.set_xticklabels(my_dict.keys())

# jacobian_experiment
def jac_f_i(args): 
    x, y, model = args[0], args[1], args[2]
    # x = x_test[i:i+1]
    # y = y_test[i:i+1]
    with tf.GradientTape() as t:
        logits = model(x) 
        cce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        cce_loss2 = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        loss1 = cce_loss(y[:, 0], logits[:, :10]) 
        loss2 = cce_loss2(y[:, 1], logits[:, 10:])
        loss = tf.convert_to_tensor([loss1, loss2])
    jac = t.jacobian(loss, weights)
    return jac[0]

def jacobian_experiment(model, model_name):
    n=500
    print('starting jacobian_experiment')

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
        # with Pool(40) as p:
        #     jac_dict[layer_name] = p.map(jac_f_i, [ [x_test[i:i+1], y_test[i:i+1], model] for i in range(n)])
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
         
            jac_dict[layer_name].append(jac[0])

    #         if jac1 != None: 
    #             print(jac1[0].shape)
    #             print(tf.reshape(jac1[0], (-1, jac1[0].shape[-1])).shape)

    for layer in model.layers: 
        weights = layer.trainable_weights
        if weights != []:
            jac_dict[layer.name] = []
            jacobian_f(weights, layer.name)

    m = n
    plt.rcParams['figure.figsize'] = [12, 8]
    j = 1

    for layer in model.layers: 
        name = layer.name 
        weights = layer.trainable_weights
        if weights != []:
            grads = jac_dict[name]
        #     print(len(grads))
        #     break
            X = np.array([g for g in grads])[:m] 
            dist = 0
            for i in range(X.shape[0]): 
                dist += np.sqrt(np.sum((X[i, 0] - X[i, 1]) ** 2))
    #         assert(X.shape[1] == 2)
            print(X.shape)
            X = X * weights[0].numpy()

            X = X.reshape(m, 2, -1)
            X = X.reshape(m*2, -1)

            Y = y_test[:m].numpy().reshape(-1).astype(np.int32)

            X_embedded = TSNE(n_components=2, perplexity=10.0, n_iter = 2000, learning_rate='auto', init='pca').fit_transform(X)
            """
            n_components=2, *, perplexity=30.0, early_exaggeration=12.0, learning_rate='warn', n_iter=1000, 
            n_iter_without_progress=300, min_grad_norm=1e-07, metric='euclidean', metric_params=None, init='warn', 
            verbose=0, random_state=None, method='barnes_hut', angle=0.5, n_jobs=None, square_distances='deprecated') 
            """
           
            labels = [i % 2 for i in range(2*m)]
            plt.figure()
            plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c = labels)
    #         for i in range(0, len(X_embedded), 2):
    #             plt.plot(X_embedded[i:i+2, 0], X_embedded[i:i+2, 1], '-', lw=.4)
            for i in range(X_embedded.shape[0]):
                plt.text(X_embedded[i,0], X_embedded[i,1], str(Y[i]))
            plt.xticks([])
            plt.yticks([])
            if 'conv' in layer.name: 
                layer_type = "Convolutional"
            else: 
                layer_type = "Fully Connected"

            plt.title(f'Layer {j} ({layer_type})')
            print(f'Layer {j} ({layer_type}), dist = {dist}')
            plt.savefig(f'plots/{model_name}/jacobian_tsne/Layer_{j}_{layer_type}')
    #         plt.legend()
            # plt.show()
            j += 1  
    connection_by_layer(model, model_name, jac_dict)
    avg_gradient_box_plot(model, model_name, jac_dict)

#Average gradient for the output 

def avg_gradient_box_plot(model, model_name, jac_dict):
    print('starting avg_gradient_box_plot')

    my_dict = {}
    x_vals = []
    y_vals = []
    layer_idx = 1


    for i, layer in enumerate(model.layers): 
        name = layer.name 
        weights = layer.trainable_weights
        if weights != []:
            grads = jac_dict[name]
        #     print(len(grads))
        #     break
            X = np.array([g for g in grads])
    #         assert(X.shape[1] == 2)
            print(X.shape)
       
            if 'conv' in name: 
                X = X.mean(axis=(0,2,3,4))
              
            if 'dense' in name: 
                X = X.mean(axis=(0,3))
            
            print(f'{name} std: {np.std(X[0])},  {np.std(X[1])}')
    #         my_dict[name] = (X[0] / np.std(X[0])) - (X[1] / np.std(X[1]))
            
            if 'conv' in layer.name: 
                layer_type = "Conv"
            else: 
                layer_type = "Dense"

            layer_name = f'{layer_idx}_{layer_type}'
            my_dict[layer_name] = []

            for x in (X[0] / np.std(X[0])) - (X[1] / np.std(X[1])): 
                if x > .00001: 
                    my_dict[layer_name].append(x)
            layer_idx += 1
                
    # ax.scatter(x_vals, y_vals)
    plt.figure() 

    plt.boxplot(my_dict.values())
    # plt.xticks(my_dict.keys())
    plt.xticks(range(1, len(my_dict.keys()) + 1), my_dict.keys())


    plt.savefig(f'plots/{model_name}/avg_gradient_box_plot')
# fig2, ax2 = plt.subplots()
# ax2.boxplot(my_dict2.values())
# ax2.set_xticklabels(my_dict2.keys())



# Percent connected by layer 
def connection_by_layer(model, model_name, jac_dict):
    print('starting connection_by_layer')

    eps = .000000001
    # eps = 0
    plt.figure() 
    layer_names = []
    percent_connected_layer = []
    for i, layer in enumerate(model.layers): 
        name = layer.name 
        print(f'\n---layer: {name}---\n')
        weights = layer.trainable_weights
        if weights != []:
            grads = jac_dict[name]
        #     print(len(grads))
        #     break
            X = np.array([g for g in grads])[:1] 
            
    #         assert(X.shape[1] == 2)
            
            connected = ((-eps < X) & (X < eps)) == 0 
            #things that are not close to 0 are connected. negation of things close to 0
            print(connected.shape)
            connected_both = connected[:, 0] & connected[:, 1] #conected to both losses
            print(connected_both.shape)


            if 'conv' in name: 
                print(f'num_connected for each loss: {connected.sum(axis=(2, 3, 4, 5))}')
                print(f'num_disconnected for each loss: {(connected == 0).sum(axis=(2, 3, 4, 5))}')
                percent_connected = (connected).mean(axis=(2, 3, 4, 5))
                percent_connected_both = (connected_both).mean(axis=(1, 2, 3, 4))

            if 'dense' in name: 
                print(f'num_connected for each loss: {connected.sum(axis=(2, 3,))}')
                print(f'num_disconnected for each loss: {(connected == 0).sum(axis=(2, 3))}')
                percent_connected = (connected).mean(axis=(2, 3))
                percent_connected_both = (connected_both).mean(axis=(1, 2))

            
            print(f'percent_connected for each loss: {percent_connected}')
            print(f'percent_connected for both loss: {percent_connected_both}')

            print(percent_connected.shape)
            print(percent_connected_both.shape)
            temp = np.concatenate((percent_connected, percent_connected_both.reshape((1, 1))), axis=1)
            print(temp.shape)
            percent_connected_layer.append(temp)
            layer_names.append(name)

    percent_connected_layer = np.array(percent_connected_layer).reshape(len(layer_names), 3)
    print(percent_connected_layer.shape)
    X_axis = np.arange(len(layer_names))
    print(X_axis.shape)
    plt.figure() 

    plt.bar(X_axis - 0.2, percent_connected_layer[:,0], 0.2, label = 'L1')
    plt.bar(X_axis + 0, percent_connected_layer[:,1], 0.2, label = 'L2')
    plt.bar(X_axis + 0.2, percent_connected_layer[:,2], 0.2, label = 'Both')
     
    plt.xticks(X_axis, layer_names)
    plt.xlabel("Layer")
    plt.ylabel("Percent of neurons connected")
    # plt.title("Number of Students in each group")
    plt.legend()
    plt.savefig(f'plots/{model_name}/specialization_by_layer.png')
    plt.show()


#first layer filters 
def show_first_layer_weights(model, model_name):
    print('starting show_first_layer_weights')

    layer = model.layers[0]
    if 'conv' in layer.name: 
        filters, biases = layer.get_weights()
        # normalize filter values to 0-1 so we can visualize them

        print(filters.shape)
        # plot first few filters
        n_filters, ix = filters.shape[-1], filters.shape[-2]
        
        R = 8 
        C = n_filters // R                               
        fig,axes = plt.subplots(R, C)

        for i in range(n_filters):
            # get the filter
            f = filters[:, :, 0, i]
            # plot each channel separately
                # specify subplot and turn of axis
            axes[i % 8, i // 8].set_xticks([])
            axes[i % 8, i // 8].set_yticks([])
            # plot filter channel in grayscale
            axes[i % 8, i // 8].matshow(f[:, :])
            axes[i % 8, i // 8].axis('off')
            ix += 1
        # show the figure
        fig.savefig(f'plots/{model_name}/first_layer_weights.png')

def train_model(model):


    optimizer = tf.keras.optimizers.Adam(0.001)

    cce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # loss_fn = cce_loss
    def loss_fn(y, y_hat): 
    #     print(y[:, 0], y_hat[:, :10])
        return cce_loss(y[:, 0], y_hat[:, :10]) + cce_loss(y[:, 1], y_hat[:, 10:])

    #         cce_loss(y_batch_train[:, 0], logits[:, :10]) + cce_loss(y_batch_train[:, 0], logits[:, 10:])


    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=loss_fn
    )

    model.fit(
        train_dataset,
        epochs=6,
        validation_data=test_dataset,
    )
# !mkdir -p saved_model

# model.save('saved_model/my_model')
#set up input 
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



kernel_size = (3, 3 )

dilation_rate = 1

small_cnn = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        # layers.Conv2D(32, kernel_size=kernel_size, dilation_rate=dilation_rate, padding="same", activation="relu"),
        layers.Conv2D(16, kernel_size=kernel_size, dilation_rate=dilation_rate, padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(8, kernel_size=kernel_size, dilation_rate=dilation_rate, padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(4, kernel_size=kernel_size, dilation_rate=dilation_rate, padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(2, kernel_size=kernel_size, dilation_rate=dilation_rate, padding="same", activation="relu"),
        # layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(20, activation="relu"),
        layers.Dense(2*num_classes, activation=None),
    ]
)

short_wide = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        # layers.Conv2D(32, kernel_size=kernel_size, dilation_rate=dilation_rate, padding="same", activation="relu"),
        layers.Conv2D(64, kernel_size=kernel_size, dilation_rate=dilation_rate, padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=kernel_size, dilation_rate=dilation_rate, padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(100, activation="relu"),
        layers.Dense(2*num_classes, activation=None),
    ]
)

tall_narrow = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        # layers.Conv2D(32, kernel_size=kernel_size, dilation_rate=dilation_rate, padding="same", activation="relu"),
        layers.Conv2D(16, kernel_size=kernel_size, dilation_rate=dilation_rate, padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(16, kernel_size=kernel_size, dilation_rate=dilation_rate, padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(16, kernel_size=kernel_size, dilation_rate=dilation_rate, padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(16, kernel_size=kernel_size, dilation_rate=dilation_rate, padding="same", activation="relu"),
        # layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(48, activation="relu"),
        layers.Dense(24, activation="relu"),
        layers.Dense(2*num_classes, activation=None),
    ]
)


big_cnn = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(64, kernel_size=kernel_size, dilation_rate=dilation_rate, padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(64, kernel_size=kernel_size, dilation_rate=dilation_rate, padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(32, kernel_size=kernel_size, dilation_rate=dilation_rate, padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, kernel_size=kernel_size, dilation_rate=dilation_rate, padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(50, activation="relu"),
        layers.Dense(2*num_classes, activation=None),
    ]
)


models = [
[small_cnn, 'small_cnn'], 
            [short_wide, 'short_wide'], [tall_narrow, 'tall_narrow'], [big_cnn, 'big_cnn']]
experiments_to_run = [jacobian_experiment, activation_space, show_first_layer_weights]
#To add: activation PCA 

def main(): 

    success = {}
    for model, model_name in models: 
        success[model_name] = []

        for dir_to_make in [f'plots/{model_name}', f'plots/{model_name}/jacobian_tsne', f'plots/{model_name}/activation_space/']: 

            if os.path.isdir(dir_to_make):
                print(f'{dir_to_make} exists')
            else: 
                os.mkdir(dir_to_make)
                print(f'made dir: {dir_to_make}')
        
        train_model(model)
        print(f'{model_name} trained')
        plot_model(model, to_file=f'plots/{model_name}_architecture', show_shapes=True, show_layer_names=True)
        
        for ex in experiments_to_run: 
            try: 
                ex(model, model_name)
                success[model_name].append(True)
                fig.close('all')
            except: 
                print('failed')
                traceback.print_exc()

                success[model_name].append(False)

    print(success)

if __name__ == '__main__':
    main() 


