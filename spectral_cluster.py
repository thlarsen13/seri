import numpy as np
import tensorflow as tf 
from tensorflow import keras

from tensorflow.keras import layers
from tensorflow.keras.losses import Loss
import scipy.sparse as sparse
from sklearn.cluster import SpectralClustering
from tqdm import tqdm 
import matplotlib.pyplot as plt

from joblib import Parallel, delayed

import pickle
import os 

def picklify(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def unpicklify(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def model_to_adj_matrix(model): 

    weights = []

    for layer in model.layers: 

        if layer.weights != []:
            if 'conv' in layer.name: 
                path = f'pickles/{layer.name}_matrix'
                w = conv_to_matrix(layer.input_shape[1:], layer.weights[0])
                picklify(path, w)
            elif 'dense' in layer.name: 
                w = layer.weights[0]

            print(f'appending shape: {w.shape}')
            weights.append(w)


    return weights_to_graph(weights) 


def conv_to_matrix(in_layer_shape, conv_tensor,
                              max_weight_convention='all_one'):
    '''
    TODO do max pool connection 

    take in the shape of the incoming layer, a dict representing info about the
    conv operation, and the weight tensor of the convolution, and return an 
    array of sparse weight matrices representing the operation. the array should
    have a single element if layer_dict['max_pool_after']==False, but should have 
    two (one representing the action of the max pooling) otherwise.
    for max pooling, we linearise by connecting the maxed neurons to everything
    in their receptive field. if max_weight_convention=='all_one', all the 
    weights are one, otherwise if max_weight_convention=='one_on_n', the weights
    are all one divided by the receptive field size
    '''
    # TODO: see if vectorisation will work
    kernel_height, kernel_width, n_chan_in, n_chan_out = conv_tensor.shape
    in_height = in_layer_shape[0]
    in_width = in_layer_shape[1]
    # assert (kernel_height, kernel_width) == tuple(conv_dict['kernel_size']), f"weight tensor info doesn't match conv layer dict info - kernel size from conv_tensor.shape is {(kernel_height, kernel_width)}, but conv_dict says it's {conv_dict['kernel_size']}"
    # assert n_chan_out == conv_dict['filters'], f"weight tensor info doesn't match conv layer dict info: weight tensor says num channels out is {n_chan_out}, conv dict says it's {conv_dict['filters']}"
    assert in_layer_shape[2] == n_chan_in, f"weight tensor info doesn't match previous layer shape: weight tensor says it's {n_chan_in}, prev layer says it's {in_layer_shape[2]}"

    kernel_height_centre = int((kernel_height - 1) / 2)
    kernel_width_centre = int((kernel_width - 1) / 2)

    in_layer_size = np.product(in_layer_shape)
    out_layer_shape = (in_height, in_width, n_chan_out)
    out_layer_size = np.product(out_layer_shape)

    conv_weight_matrix = np.zeros((in_layer_size, out_layer_size))

    # THIS WORKS ONLY FOR SAME and not for VALID!!!
    for i in tqdm(range(in_height)):
        for j in range(in_width):
            for c_out in range(n_chan_out):
                out_int = cnn_layer_tup_to_int((i,j,c_out), out_layer_shape)
                for n in range(kernel_height):
                    for m in range(kernel_width):
                        for c_in in range(n_chan_in):
                            weight = conv_tensor[n][m][c_in][c_out]
                            h_in = i + n - kernel_height_centre
                            w_in = j + m - kernel_width_centre
                            in_bounds_check = (h_in in range(in_height)
                                               and w_in in range(in_width))
                            if in_bounds_check:
                                in_int = cnn_layer_tup_to_int((h_in, w_in,
                                                               c_in),
                                                              in_layer_shape)
                                conv_weight_matrix[in_int][out_int] = weight


    return conv_weight_matrix

def cnn_layer_tup_to_int(tup, layer_shape):
    '''
    take a (num_down, num_across, channel) tuple and a layer_shape of the form
    (height, width, num_channels).
    return an int that is unique within the layer representing that particular 
    'neuron'.
    '''
    down, across, channel = tup
    _, width, n_c = layer_shape
    return down * width * n_c + across * n_c + channel

def weights_to_graph(weights_array):
    # take an array of weight matrices, and return the adjacency matrix of the
    # neural network it defines.
    # if the weight matrices are A, B, C, and D, the adjacency matrix should be
    # [[0   A   0   0   0  ]
    #  [A^T 0   B   0   0  ]
    #  [0   B^T 0   C   0  ]
    #  [0   0   C^T 0   D  ]
    #  [0   0   0   D^T 0  ]]

    block_mat = []

    layer_size = [weights_array[0].shape[0]]
    # for everything in the weights array, add a row to block_mat of the form
    # [None, None, ..., sparsify(np.abs(mat)), None, ..., None]

    for (i, mat) in enumerate(weights_array):
        print(mat.shape)
        layer_size.append(mat.shape[1])
        print(f'layer_size = {layer_size}')
        sp_mat = sparse.coo_matrix(np.abs(mat))
        if i == 0:
            # add a zero matrix of the right size to the start of the first row
            # so that our final matrix is of the right size
            n = mat.shape[0]
            first_zeroes = sparse.coo_matrix((n, n))
            block_row = [first_zeroes] + [None]*len(weights_array)
        else:
            block_row = [None]*(len(weights_array) + 1)
        block_row[i+1] = sp_mat
        block_mat.append(block_row)

    # add a final row to block_mat that's just a bunch of [None]s followed by a
    # zero matrix of the right size6^^^^^^^^^^
    m = weights_array[-1].shape[1]
    final_zeroes = sparse.coo_matrix((m, m))
    nones_row = [None]*len(weights_array)
    nones_row.append(final_zeroes)
    block_mat.append(nones_row)

    # turn block_mat into a sparse matrix
    up_tri = sparse.bmat(block_mat, 'csr')

    # we now have a matrix that looks like
    # [[0   A   0   0   0  ]
    #  [0   0   B   0   0  ]
    #  [0   0   0   C   0  ]
    #  [0   0   0   0   D  ]]
    # add this to its transpose to get what we want
    adj_mat = up_tri + up_tri.transpose()
    return adj_mat, layer_size
    
def cluster_net(n_clusters, adj_mat, eigen_solver='arpack', assign_labels='kmeans'):
    if adj_mat.shape[0] > 2000:
        n_init = 100
    else:
        n_init = 25
    cluster_alg = SpectralClustering(n_clusters=n_clusters,
                                     eigen_solver=eigen_solver,
                                     affinity='precomputed',
                                     assign_labels=assign_labels,
                                     n_init=n_init)
    clustering = cluster_alg.fit(adj_mat)
    return clustering.labels_



if __name__ == '__main__': 


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

    plt.figure()
    plt.imshow(x_test[0, :, :, 0]) 
    plt.show()
    print(y_test)


    # In[12]:


    kernel_size = (3, 3 )

    dilation_rate = 1

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(64, kernel_size=kernel_size, dilation_rate=dilation_rate, padding="same", activation="relu"),
            layers.Conv2D(64, kernel_size=kernel_size, dilation_rate=dilation_rate, padding="same", activation="relu"),
            layers.Conv2D(32, kernel_size=kernel_size, dilation_rate=dilation_rate, padding="same", activation="relu"),
            # layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(32, kernel_size=kernel_size, dilation_rate=dilation_rate, padding="same", activation="relu"),
            # layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(50, activation="relu"),
            layers.Dense(2*num_classes, activation=None),
        ]
    )

    model.summary()


    optimizer = tf.keras.optimizers.Adam(0.001)

    cce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # loss_fn = cce_loss
    def loss_fn(y, y_hat): 
    #     print(y[:, 0], y_hat[:, :10])
        return cce_loss(y[:, 0], y_hat[:, :10]) + cce_loss(y[:, 1], y_hat[:, 10:])

    #         cce_loss(y_batch_train[:, 0], logits[:, :10]) + cce_loss(y_batch_train[:, 0], logits[:, 10:])



    # In[ ]:


    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=loss_fn
    )

    model.fit(
        train_dataset,
        epochs=6,
        validation_data=test_dataset,
    )
    del train_dataset
    del test_dataset



    adj, layer_size = model_to_adj_matrix(model)

    print(f'DONE!!!: adj shape: {adj.shape}')

    clustering = cluster_net(3, adj)
    print(f'clustering finished!')

    cluster_count = { l : [] for l in range(len(layer_size))}

    j = 0
    for i, sz in enumerate(layer_size): 
        print(f'layer {i}, sz={sz}')
        for l in range(len(layer_size)): 
            cluster_count[l].append(np.sum(clustering[j:j+sz] == l))
            
    # for l in range(len(layer_size)): 

    #         print(count)
    #     j += sz

  
# plot bars in stack manner
# plt.bar(x, y1, color='r')
# plt.bar(x, y2, bottom=y1, color='b')
# plt.show()
