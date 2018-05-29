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

n_glimpse_per_emlement = 5  # 한 성분마다 5번씩 흘려봄
n_element_per_charater = 3  # 초성, 중성, 종성
T = n_glimpse_per_emlement * n_element_per_charater  # 10
eps = 1e-7
lr = 1e-3
patch_size = 8
std = 0.03

channels = 1
g_depth = 3  # depth of glimpse sensors

sensor_bandwidth = 12
total_sensor_bandwidth = g_depth * channels * (sensor_bandwidth**2)

eye_centered = False
loc_sd = 0.22
