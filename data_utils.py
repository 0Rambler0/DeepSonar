import numpy as np
import pickle
from cProfile import label
import sys
from tensorflow.python.framework.errors_impl import OutOfRangeError
sys.path.append('/home/dslab/hx/audio_detection/DeepSonar/SR_module')
import tensorflow as tf

def load_feature(path):
    feature_list = []
    labels = []

    with open(path ,'rb') as f:
        while True:
            try:
                data_dict = pickle.load(f)
                feature_list.extend(data_dict['data'])
                labels.extend(data_dict['labels'])
            except EOFError:
                break

    return feature_list, labels

def load_net_param(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def reshape(data):
    new_vector = []
    # session = tf.Session()
    for i in data:
        # i = session.run(i)
        i = np.squeeze(i)
    l1 = np.reshape(data[0], (-1, 64))
    l2 = np.reshape(data[1], (-1, 96))
    l3 = np.reshape(data[2], (-1, 128))
    l4 = np.reshape(data[3], (-1, 256))
    new_vector.append(l1)
    new_vector.append(l2)
    new_vector.append(l3)
    new_vector.append(l4)
    for i in range(0,len(new_vector)):
        if (int(new_vector[i].shape[0])*int(new_vector[i].shape[1]))%512:
            new_index = int(int(new_vector[i].shape[0])*int(new_vector[i].shape[1])/512)*512/int(new_vector[i].shape[1])
            new_index = int(new_index)
            new_vector[i] = new_vector[i][0:new_index, :]
        new_vector[i] = np.reshape(new_vector[i], (512, -1))

    l5 = np.reshape(data[4], (512, -1))
    l6 = np.reshape(data[5], (512, -1))
    l7 = np.reshape(data[6], (512, -1))
    new_vector.append(l5)
    new_vector.append(l6)
    new_vector.append(l7)

    return new_vector

def cut(vector_list):
    for i in range(len(vector_list)):
        vector_list[i] = vector_list[i][:, 0:int(512/2**i)]
    cut_matrix = np.concatenate(vector_list, 1)

    return cut_matrix

def write_data(writer, labels, activation_matrix_list):
    count = 0
    for activation_matrix, label in zip(activation_matrix_list, labels):
        activation_matrix = activation_matrix.tobytes()
        feature = {'data':tf.train.Feature(bytes_list=tf.train.BytesList(value=[activation_matrix])),
                   'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))}
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
        count += 1
        print('writing: {}/{}'.format(count, len(labels)))
    print('write {} data'.format(len(labels)))

def write_data_pkl(activation_matrix_list, labels, save_path):
    data_dict = {'data':activation_matrix_list, 'labels':labels}
    with open(save_path, 'ab') as f:
        pickle.dump(data_dict, f, pickle.HIGHEST_PROTOCOL)
    print('write {} data'.format(len(labels)))

def search_threshold(sum, sum_sample, sum_neuron):
    threshold_list = []
    layers_num = 7
    
    for i in range(layers_num):
        tmp_sum = np.sum(sum[i])
        threshold = tmp_sum/(sum_sample*sum_neuron[i])
        threshold_list.append(threshold)

    return threshold_list

def map_func(example):
    feature_map = {'data': tf.FixedLenFeature((), tf.string),
                   'label': tf.FixedLenFeature((), tf.int64)
                   }
    parsed_example = tf.parse_single_example(example, features=feature_map)
    data = tf.decode_raw(parsed_example["data"], tf.float32)
    label = parsed_example["label"]
    
    return data, label

def read_data(path):
    data = []
    labels = []

    dataset = tf.data.TFRecordDataset(path)
    dataset = dataset.map(map_func=map_func)
    iterator = dataset.make_one_shot_iterator()
    element = iterator.get_next()
    with tf.Session() as sess:
        while True:
            try:
                data_, label_ = sess.run(element)
                # data_ = sess.run(data_)
                data_ = np.reshape(data_, (512, 1009))
                data.append(data_)
                labels.append(label_)
            except OutOfRangeError:
                print('read completely')
                break
    data = np.array(data)
    # data = np.expand_dims(data, 1)
    data = np.reshape(data, newshape=(-1, 512, 1009))
    # data = np.expand_dims(data, 1)
    labels = np.array(labels)

    return data, labels

def plus_list(a, b):
    a = np.array(a)
    b = np.array(b)
    for i in range(len(a)):
        a[i] = np.maximum(a[i],-a[i])
        b[i] = np.maximum(b[i],-b[i])
    sum_abs = a + b

    return sum_abs