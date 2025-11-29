import numpy as np
import matplotlib.pyplot as plt
import main_functions as main

import random # get_mini_batch
from tqdm.auto import tqdm # aesthetics 

def get_mini_batch(im_train, label_train, batch_size):
    mini_batch_x, mini_batch_y = None, None
    
    n = im_train.shape[1]

    idxs = random.sample(range(n), n) # reorder indices randomly

    mini_batch_x = []
    mini_batch_y = []

    for i in range(0, n, batch_size):
        batch_idxs = idxs[i:i+batch_size]
        mini_batch_x.append(im_train[:, batch_idxs]) 

        labels = label_train[:, batch_idxs]
        one_hot = np.zeros((10, labels.shape[1]))
        for j in range(labels.shape[1]):
            one_hot[labels[0, j], j] = 1
        mini_batch_y.append(one_hot)
                    
    return mini_batch_x, mini_batch_y

def fc(x, w, b):
    y = None

    y = w @ x + b
    
    return y

def fc_backward(dl_dy, x, w, b, y):
    dl_dx, dl_dw, dl_db = None, None, None

    dl_dx = w.T @ dl_dy
    dl_dw = dl_dy @ x.T
    dl_db = np.sum(dl_dy, axis=1, keepdims=True)

    return dl_dx, dl_dw, dl_db

def loss_euclidean(y_tilde, y):
    l, dl_dy = None, None

    l = np.linalg.norm(y - y_tilde) ** 2 
    dl_dy = (y_tilde - y) / y.shape[1]

    return l, dl_dy

def loss_cross_entropy_softmax(a, y):
    l, dl_da = None, None

    # getting y_tilde 
    exp_a = np.exp(a)
    denom = np.sum(exp_a, axis=0, keepdims=True)
    y_tilde = exp_a / denom
    dl_da = (y_tilde - y) / y.shape[1] # y_tilde is softmax output

    # getting loss
    l = - np.sum(y * np.log(y_tilde + 1e-15)) / y.shape[1] # adding small value to avoid log(0)

    return l, dl_da

def relu(x):
    y = None

    y = np.maximum(0, x)

    return y

def relu_backward(dl_dy, x, y):
    dl_dx = None

    # smart way of doing things
    # x > 0 gives boolean (0 or 1) matrix
    dl_dx = dl_dy * (x > 0)

    return dl_dx

def conv(x, w_conv, b_conv):
    y = None
    
    x = np.pad(x, ((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0) # zero padding
    h = w_conv.shape[0]
    w = w_conv.shape[1]
    stride = 1 
    
    cols = im2col(x, h, w, stride) # (new_h*new_w,c*hh*ww)
    cols = cols.T # (c*hh*ww,new_h*new_w)

    # flattening filters
    w_conv = w_conv.reshape(-1, w_conv.shape[3]) # (hh*ww*c,num_filters)

    y = w_conv.T @ cols + b_conv # (num_filters,new_h*new_w)
    y = y.T # (new_h*new_w,num_filters)
    y = y.reshape((x.shape[0]-h)//stride + 1, (x.shape[1]-w)//stride + 1, -1) # (new_h,new_w,num_filters)

    return y

# copied from https://leonardoaraujosantos.gitbook.io/artificial-inteligence/machine_learning/deep_learning/convolution_layer/making_faster
# adjusted for use in this project
def im2col(x,hh,ww,stride):

    """
    Args:
      x: image matrix to be translated into columns, (C, H, W) [OLD] -> (H, W, C) [NEW]
      hh: filter height
      ww: filter width
      stride: stride
    Returns:
      col: (new_h*new_w,hh*ww*C) matrix, each column is a cube that will convolve with a filter
            new_h = (H-hh) // stride + 1, new_w = (W-ww) // stride + 1
    """

    # c,h,w = x.shape [OLD]
    h, w, c = x.shape # [NEW]
    new_h = (h-hh) // stride + 1
    new_w = (w-ww) // stride + 1
    col = np.zeros([new_h*new_w,c*hh*ww])

    for i in range(new_h):
       for j in range(new_w):
           # patch = x[...,i*stride:i*stride+hh,j*stride:j*stride+ww] [OLD]
           patch = x[i*stride:i*stride+hh,j*stride:j*stride+ww,:] # [NEW]
           col[i*new_w+j,:] = np.reshape(patch,-1)
    return col

def conv_backward(dl_dy, x, w_conv, b_conv, y):
    dl_dw, dl_db = None, None

    dl_dy = dl_dy.reshape(-1, dl_dy.shape[2]) # (new_h*new_w,num_filters)
    dl_dy = dl_dy.T # (num_filters,new_h*new_w)

    x = np.pad(x, ((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0) # zero padding
    h = w_conv.shape[0]
    w = w_conv.shape[1]
    stride = 1 

    cols = im2col(x, h, w, stride) # (new_h*new_w,c*hh*ww)
    cols = cols.T # (c*hh*ww,new_h*new_w)

    dl_db = np.sum(dl_dy, axis=1, keepdims=True) # (num_filters, 1)
    dl_dw = dl_dy @ cols.T # (num_filters,c*hh*ww)
    dl_dw = dl_dw.T # (c*hh*ww,num_filters)
    dl_dw = dl_dw.reshape(w_conv.shape) # (hh,ww,c,num_filters)

    return dl_dw, dl_db

def pool2x2(x):
    y = None

    H, W, C = x.shape
    H_out, W_out = H // 2, W // 2

    x = x.reshape(H_out, 2, W_out, 2, C)
    y = x.max(axis=1).max(axis=2)

    return y

def pool2x2_backward(dl_dy, x, y):
    dl_dx = None
    
    # need to get singe max value positions
    H, W, C = x.shape
    H_out, W_out = H // 2, W // 2
    
    x = x.reshape(H_out, 2, W_out, 2, C)
    x = x.transpose(0, 2, 4, 1, 3) # (H_out, W_out, C, 2, 2)
    x = x.reshape(H_out, W_out, C, 4) # (H_out, W_out, C, 4)
    
    max_idx = np.argmax(x, axis=3) # (H_out, W_out, C)
    
    mask = np.zeros_like(x) # (H_out, W_out, C, 4)
    
    # getting max specific indices
    H_idx = np.arange(H_out)[:, None, None]
    W_idx = np.arange(W_out)[None, :, None]
    C_idx = np.arange(C)[None, None, :]
    
    mask[H_idx, W_idx, C_idx, max_idx] = 1 # one per block
    
    # reshaping mask to original x shape
    mask = mask.reshape(H_out, W_out, C, 2, 2)
    mask = mask.transpose(0, 3, 1, 4, 2)
    
    dl_dy = dl_dy.reshape(H_out, 1, W_out, 1, C)
    
    dl_dx = mask * dl_dy
    dl_dx = dl_dx.reshape(H, W, C)

    return dl_dx

def flattening(x):
    y = None

    y = x.reshape(-1, 1)

    return y

def flattening_backward(dl_dy, x, y):
    dl_dx = None

    dl_dx = dl_dy.reshape(x.shape)

    return dl_dx

def train_slp_linear(mini_batch_x, mini_batch_y):
    w, b = None, None

    # hyperparameters
    gamma_ = 0.01 # learning rate
    lambda_ = 1 # decay rate 
    nIters = 5000 # hyperparameter: number of iterations

    k = 1
    w = np.random.randn(10, mini_batch_x[0].shape[0])
    b = np.random.randn(10, 1) # bias intialization (not given in algorithm, but necessary)

    for iIter in tqdm(range(1, nIters), desc="Training SLP Linear"):
        # at every 1000th iteration, gamma <- lambda * gamma
        if iIter % 1000 == 0:
            gamma_ = lambda_ * gamma_

        dL_dw = np.zeros(w.shape)
        dL_db = np.zeros(b.shape)

        # for each image xi in kth mini-batch 
        for j in range(mini_batch_x[k-1].shape[1]):
            xi = mini_batch_x[k-1][:, [j]] # get jth image in kth mini-batch
            y = mini_batch_y[k-1][:, [j]] # get corresponding one-hot label

            # forward pass
            y_tilde = fc(xi, w, b)
            _, dl_dy = loss_euclidean(y_tilde, y)

            # backward pass
            _, dl_dw, dl_db = fc_backward(dl_dy, xi, w, b, y_tilde)

            dL_dw += dl_dw
            dL_db += dl_db
        
        # set k = 1 if k is greater than the number of mini-batches
        if k >= len(mini_batch_x):
            k = 1
        else:
            k += 1

        # update weights and bias
        R = mini_batch_x[k-2].shape[1]
        w = w - gamma_ / R * dL_dw 
        b = b - gamma_ / R * dL_db 

    return w, b

def train_slp(mini_batch_x, mini_batch_y):
    w, b = None, None

    # hyperparameters
    gamma_ = 0.05 # learning rate
    lambda_ = 1 # decay rate 
    nIters = 20000 # hyperparameter: number of iterations

    k = 1
    w = np.random.randn(10, mini_batch_x[0].shape[0])
    b = np.random.randn(10, 1) # bias intialization (not given in algorithm, but necessary)

    for iIter in tqdm(range(1, nIters), desc="Training SLP"):
        # at every 1000th iteration, gamma <- lambda * gamma
        if iIter % 1000 == 0:
            gamma_ = lambda_ * gamma_

        dL_dw = np.zeros(w.shape)
        dL_db = np.zeros(b.shape)

        # for each image xi in kth mini-batch 
        for j in range(mini_batch_x[k-1].shape[1]):
            xi = mini_batch_x[k-1][:, [j]] # get jth image in kth mini-batch
            y = mini_batch_y[k-1][:, [j]] # get corresponding one-hot label

            # forward pass
            y_tilde = fc(xi, w, b)
            _, dl_dy = loss_cross_entropy_softmax(y_tilde, y)

            # backward pass
            _, dl_dw, dl_db = fc_backward(dl_dy, xi, w, b, y_tilde)

            dL_dw += dl_dw
            dL_db += dl_db
        
        # set k = 1 if k is greater than the number of mini-batches
        if k >= len(mini_batch_x):
            k = 1
        else:
            k += 1

        # update weights and bias
        R = mini_batch_x[k-2].shape[1]
        w = w - gamma_ / R * dL_dw 
        b = b - gamma_ / R * dL_db 

    return w, b

def train_mlp(mini_batch_x, mini_batch_y):
    w1, b1, w2, b2 = None, None, None, None

    # hyperparameters
    gamma_ = 0.05 # learning rate
    lambda_ = 1 # decay rate
    nIters = 40000 # number of iterations

    k = 1
    w1 = np.random.randn(30, mini_batch_x[0].shape[0])
    b1 = np.random.randn(30, 1) 
    w2 = np.random.randn(10, 30)
    b2 = np.random.randn(10, 1)

    for iIter in tqdm(range(1, nIters), desc="Training MLP"):
        # at every 1000th iteration, gamma <- lambda * gamma
        if iIter % 1000 == 0:
            gamma_ = lambda_ * gamma_

        dL_dw1 = np.zeros(w1.shape)
        dL_db1 = np.zeros(b1.shape)
        dL_dw2 = np.zeros(w2.shape)
        dL_db2 = np.zeros(b2.shape)

        # for each image xi in kth mini-batch 
        for j in range(mini_batch_x[k-1].shape[1]):
            xi = mini_batch_x[k-1][:, [j]] # get jth image in kth mini-batch
            y = mini_batch_y[k-1][:, [j]] # get corresponding one-hot label

            # forward pass
            a1 = fc(xi, w1, b1)
            h = relu(a1)
            a2 = fc(h, w2, b2)
            _, dl_da2 = loss_cross_entropy_softmax(a2, y)

            # backward pass
            dl_dh, dl_dw2, dl_db2 = fc_backward(dl_da2, h, w2, b2, a2)
            dl_da1 = relu_backward(dl_dh, a1, h)
            _, dl_dw1, dl_db1 = fc_backward(dl_da1, xi, w1, b1, a1)

            dL_dw1 += dl_dw1
            dL_db1 += dl_db1
            dL_dw2 += dl_dw2
            dL_db2 += dl_db2

        # set k = 1 if k is greater than the number of mini-batches
        if k >= len(mini_batch_x):
            k = 1
        else:
            k += 1

        # update weights and bias
        R = mini_batch_x[k-2].shape[1]
        w1 = w1 - gamma_ / R * dL_dw1
        b1 = b1 - gamma_ / R * dL_db1
        w2 = w2 - gamma_ / R * dL_dw2
        b2 = b2 - gamma_ / R * dL_db2

    return w1, b1, w2, b2

def train_cnn(mini_batch_x, mini_batch_y):
    w_conv, b_conv, w_fc, b_fc = None, None, None, None

    # hyperparameters
    gamma_ = 0.04 # learning rate
    lambda_ = 0.99 # decay rate
    nIters = 100000 # number of iterations

    k = 1
    w_conv = np.random.randn(3, 3, 1, 3)
    b_conv = np.random.randn(3, 1)
    w_fc = np.random.randn(10, 147)
    b_fc = np.random.randn(10, 1)

    # loss plot CAN BE REMOVED
    losses = []

    for iIter in tqdm(range(1, nIters), desc="Training CNN"):
        # at every 1000th iteration, gamma <- lambda * gamma
        if iIter % 1000 == 0:
            gamma_ = lambda_ * gamma_

        dL_dw_conv = np.zeros(w_conv.shape)
        dL_db_conv = np.zeros(b_conv.shape)
        dL_dw_fc = np.zeros(w_fc.shape)
        dL_db_fc = np.zeros(b_fc.shape)

        # for each image xi in kth mini-batch 
        for j in range(mini_batch_x[k-1].shape[1]):
            xi = mini_batch_x[k-1][:, [j]].reshape((14, 14, 1), order='F') # get jth image in kth mini-batch
            y = mini_batch_y[k-1][:, [j]] # get corresponding one-hot label

            # forward pass
            a_conv = conv(xi, w_conv, b_conv)
            h_conv = relu(a_conv)
            h_pool = pool2x2(h_conv)
            h_flat = flattening(h_pool)
            a_fc = fc(h_flat, w_fc, b_fc)
            l, dl_da_fc = loss_cross_entropy_softmax(a_fc, y)

            # store loss CAN BE REMOVED
            losses.append(l)

            # backward pass
            dl_dh_flat, dl_dw_fc, dl_db_fc = fc_backward(dl_da_fc, h_flat, w_fc, b_fc, a_fc)
            dl_dh_pool = flattening_backward(dl_dh_flat, h_pool, h_flat)
            dl_dh_conv = pool2x2_backward(dl_dh_pool, h_conv, h_pool)
            dl_da_conv = relu_backward(dl_dh_conv, a_conv, h_conv)
            dl_dw_conv, dl_db_conv = conv_backward(dl_da_conv, xi, w_conv, b_conv, a_conv)
            
            dL_dw_conv += dl_dw_conv
            dL_db_conv += dl_db_conv
            dL_dw_fc += dl_dw_fc
            dL_db_fc += dl_db_fc

        # set k = 1 if k is greater than the number of mini-batches
        if k >= len(mini_batch_x):
            k = 1
        else:
            k += 1

        # update weights and bias
        R = mini_batch_x[k-2].shape[1]
        w_conv = w_conv - gamma_ / R * dL_dw_conv
        b_conv = b_conv - gamma_ / R * dL_db_conv
        w_fc = w_fc - gamma_ / R * dL_dw_fc
        b_fc = b_fc - gamma_ / R * dL_db_fc

    # plot losses CAN BE REMOVED
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss over Iterations')
    plt.show()

    return w_conv, b_conv, w_fc, b_fc

if __name__ == '__main__':
    main.main_slp_linear(load_weights=True) # 0.545
    main.main_slp(load_weights=True) # 0.887
    main.main_mlp(load_weights=True) # 0.906
    main.main_cnn(load_weights=True) # 0.938