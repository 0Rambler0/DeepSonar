import os
import sys
from data_utils import *
sys.path.append('/home/dslab/hx/audio_detection/DeepSonar/SR_module')
from predict import convert_audio_processing, SR_Model
import pickle
from sr_model_param import *


def get_net_param(fake_path, real_path, sr_model, save_path):
    real_list = os.listdir(real_path)
    fake_list = os.listdir(fake_path)
    total = len(real_list) + len(fake_list)
    first_flag = 1
    num_sample = 0
    last_reload_code = 0
    sum = None
    num_neuron = []
    path_list = []

    for file in real_list:
        num_sample += 1 
        if num_sample%100 == 0:
            print('rate: {}% ({}/{})'.format(100*num_sample/total, num_sample, total))
        path_list.append(os.path.join(real_path, file))
        if len(path_list) == 100:
            specs_list = convert_audio_processing(sr_model, path_list)
            path_list = []
            feature_data = sr_model.get_feature(specs_list)
            feature_list = []
            for i in range(len(feature_data[0])):
                feature = []
                for j in range(7):
                    feature.append(feature_data[j][i])
                feature_list.append(feature)
            for feature in feature_list: 
                """for i in range(len(feature)):
                    feature[i] = np.squeeze(feature[i])"""
                if first_flag:
                    sum = feature
                    for layer in feature:
                        num_neuron.append(layer.size)
                    first_flag = 0
                else:
                    sum = plus_list(sum, feature)
        if num_sample - last_reload_code > 1000:
            sr_model.reload_model()
            last_reload_code = num_sample
    if len(path_list) >0:
        specs_list = convert_audio_processing(sr_model, path_list)
        path_list = []
        feature_data = sr_model.get_feature(specs_list)
        feature_list = []
        for i in range(len(feature_data[0])):
            feature = []
            for j in range(7):
                feature.append(feature_data[j][i])
            feature_list.append(feature)
        for feature in feature_list:
            # feature = sr_model.get_feature(specs)
            """for i in range(len(feature)):
                feature[i] = np.squeeze(feature[i])"""
            sum = plus_list(sum, feature)
    for file in fake_list:
        num_sample += 1 
        if num_sample%100 == 0:
            print('rate: {}% ({}/{})'.format(100*num_sample/total, num_sample, total))
        path_list.append(os.path.join(fake_path, file))
        if len(path_list) == 100:
            specs_list = convert_audio_processing(sr_model, path_list)
            path_list = []
            feature_data = sr_model.get_feature(specs_list)
            feature_list = []
            for i in range(len(feature_data[0])):
                feature = []
                for j in range(7):
                    feature.append(feature_data[j][i])
                feature_list.append(feature)
            for feature in feature_list:
                sum = plus_list(sum, feature)
        if num_sample - last_reload_code > 1000:
            sr_model.reload_model()
            last_reload_code = num_sample
    if len(path_list) > 0:
        specs_list = convert_audio_processing(sr_model, path_list)
        path_list = []
        feature_data = sr_model.get_feature(specs_list)
        feature_list = []
        for i in range(len(feature_data[0])):
            feature = []
            for j in range(7):
                feature.append(feature_data[j][i])
            feature_list.append(feature)
        for feature in feature_list:
            sum = plus_list(sum, feature)
    param_dict = {'sum':sum, 'num_neuron':num_neuron, 'num_sample':num_sample}
    with open(save_path, 'wb') as f:
        pickle.dump(param_dict, f, pickle.HIGHEST_PROTOCOL)

    

if __name__ == '__main__':
    dataset_path = '/home/dslab/hx/vfuzz/media/data/for-2seconds/testing'
    save_path = 'dataset_param/param_test_abs.pkl' #please change filename if you get a new dataset param
    real_path = os.path.join(dataset_path, 'real')
    fake_path = os.path.join(dataset_path, 'fake')
    sr_model = SR_Model(sr_model_path, sr_params,sr_args)
    sr_model.load_model()
    get_net_param(fake_path, real_path, sr_model, save_path)
    # param_file = load_net_param('param.pkl')
    print('get net param complete')
    