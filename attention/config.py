'''
Here are some variables(parameters) for attention network and train it.
'''

batch_size = 8
img_sz = 28
img_len = img_sz*img_sz
n_itr = 10000
n_class = 10
lstm_size = 256
hidden_size = 1024
hidden_size2= 128

n_glimpse_per_element = 4  # 한 성분마다 5번씩 흘려봄
n_element_per_character = 3  # 초성, 중성, 종성
T = n_glimpse_per_element * n_element_per_character  # 15

# Action network에서 초 / 중 / 종성마다의 classification을 위함
n_initial_character = 19
n_middle_character = 21
n_final_character = 27 + 1


eps = 1e-7
lr = 1e-3
std = 0.03

channels = 1
g_depth = 3  # depth of glimpse sensors : glimpse feature를 추출하기 위해 3단계의 영역 추출

patch_size = 3  # Glimpse 영역. loc 기준 주변을 보는 영역의 크기

sensor_bandwidth = 12   # 잘라온 패치를 sensor_bandwidth 크기로 resize
total_sensor_bandwidth = g_depth * channels * (sensor_bandwidth**2)

eye_centered = False
loc_sd = 0.5  # 0.22

