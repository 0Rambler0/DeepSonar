import multiprocessing
import pickle
import numpy as np
from data_utils import *

def check_activation_layer(layer_data, threshold, dim):
    new_layer_data = np.squeeze(layer_data)
    if dim == 3:
        for x in range(len(new_layer_data)):
            for y in range(len(new_layer_data[x])):
                for z in range(len(new_layer_data[x][y])):
                    new_layer_data[x][y][z] = 1 if new_layer_data[x][y][z] > threshold else 0
    if dim == 2:
        for x in range(len(new_layer_data)):
            for y in range(len(new_layer_data[x])):
                new_layer_data[x][y] = 1 if new_layer_data[x][y] > threshold else 0
    if dim == 1:
        for x in range(len(new_layer_data)):
            new_layer_data[x] = 1 if new_layer_data[x] > threshold else 0

    return new_layer_data

def check_activation_layer_new(layer_data, threshold, layer_num):
    start_depth = 0
    end_depth = 0
    if layer_num > 1:
        for i in range(layer_num-1):
            start_depth += int(512/2**i)
        if layer_num < 7:
            end_depth = start_depth + int(512/2**(layer_num-1))
        else:
            end_depth = start_depth + 1
    else:
        end_depth = 512
    #print('start depth:{} end depth:{} layer_num:{}'.format(start_depth, end_depth, layer_num))
    for i in range(len(layer_data)):
        for j in range(start_depth, end_depth):
            layer_data[i][j] = 1 if np.abs(layer_data[i][j]) > threshold else 0

    return layer_data

def check_activation_new(data, threshold_list):
    for i in range(7):
        activation_matrix = check_activation_layer_new(data, threshold_list[i], i+1)

    return activation_matrix


def check_activation(data, threshold_list):
    activation_matrix = []

    for i in range(5):
        layer_data = check_activation_layer(data[i], threshold_list[i], 3)
        activation_matrix.append(layer_data)
    layer_data = check_activation_layer(data[5], threshold_list[5], 2)
    activation_matrix.append(layer_data)
    layer_data = check_activation_layer(data[6], threshold_list[6], 1)
    activation_matrix.append(layer_data)
    activation_matrix = np.array(activation_matrix)

    return activation_matrix

def make_activation_matrix(feature_list, threshold_list):
    print('start make activation matrix')
    activation_matrix_list = []
    worker_pool = multiprocessing.Pool(processes = 19)
    process_res = []

    for feature in feature_list:
        res = worker_pool.apply_async(check_activation_new, [feature, threshold_list])
        # res = check_activation_new(feature, threshold_list)
        process_res.append(res)
    worker_pool.close()
    worker_pool.join
    count = 0
    for res in process_res:
        count += 1
        activation_matrix_list.append(res.get())
        if count%100 == 0:
            print('rate:{}/{}'.format(count, len(labels)))
    # activation_matrix_list = feature_list
    print('make activation matrix complete')

    return activation_matrix_list


if __name__ == '__main__':
    net_param = load_net_param('/home/dslab/hx/audio_detection/DeepSonar/param_test_abs.pkl')
    threshold_list = search_threshold(net_param['sum'], net_param['num_sample'], net_param['num_neuron'])
    feature_list, labels = load_feature('/home/dslab/hx/vfuzz/media/data/SR_feature/test_raw.pkl')
    for i in range(50):
        feature_list.pop(-1)
        labels.pop(-1)
    activation_matrix_list = make_activation_matrix(feature_list, threshold_list)
    write_data_pkl(activation_matrix_list, labels, '/home/dslab/hx/vfuzz/media/data/SR_feature/test.pkl')