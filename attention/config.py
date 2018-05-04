'''
Here are some variables(parameters) for attention network and train it.
'''

batch_size = 128
img_sz = 28
img_len = img_sz*img_sz
n_itr = 10000
n_class = 10
lstm_size = 256
hidden_size = 1024
T = 10
eps = 1e-7
lr = 1e-3
patch_size = 8
std = 0.03