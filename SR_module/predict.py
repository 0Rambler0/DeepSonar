from __future__ import absolute_import
from __future__ import print_function
import multiprocessing
import os
import sys
import keras
import numpy as np
import model
import tensorflow as tf
import toolkits
import utils as ut
from keras import backend as K
import pdb
import gc
# ===========================================
#        Parse the argument
# ===========================================
import argparse
parser = argparse.ArgumentParser()
# set up training configuration.
parser.add_argument('--gpu', default='', type=str)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--data_path', default='/home/dslab/hx/vfuzz/media/data/for-fake-validation/file1.mp3.wav_16k.wav_norm.wav_mono.wav_silence.wav', type=str)
# set up network configuration.
parser.add_argument('--net', default='resnet34s', choices=['resnet34s', 'resnet34l'], type=str)
parser.add_argument('--ghost_cluster', default=2, type=int)
parser.add_argument('--vlad_cluster', default=8, type=int)
parser.add_argument('--bottleneck_dim', default=512, type=int)
parser.add_argument('--aggregation_mode', default='gvlad', choices=['avg', 'vlad', 'gvlad'], type=str)
# set up learning rate, training loss and optimizer.
parser.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax'], type=str)
parser.add_argument('--test_type', default='normal', choices=['normal', 'hard', 'extend'], type=str)

global args
args = parser.parse_args()

def main():

    # gpu configuration
    toolkits.initialize_GPU(args)

    
    # ==================================
    #       Get Train/Val.
    # ==================================
    print('==> calculating test({}) data lists...'.format(args.test_type))

    if args.test_type == 'normal':
        verify_list = np.loadtxt('../meta/voxceleb1_veri_test.txt', str)
    elif args.test_type == 'hard':
        verify_list = np.loadtxt('../meta/voxceleb1_veri_test_hard.txt', str)
    elif args.test_type == 'extend':
        verify_list = np.loadtxt('../meta/voxceleb1_veri_test_extended.txt', str)
    else:
        raise IOError('==> unknown test type.')

    verify_lb = np.array([int(i[0]) for i in verify_list])
    list1 = np.array([os.path.join(args.data_path, i[1]) for i in verify_list])
    list2 = np.array([os.path.join(args.data_path, i[2]) for i in verify_list])

    total_list = np.concatenate((list1, list2))
    unique_list = np.unique(total_list)

    # ==================================
    #       Get Model
    # ==================================
    # construct the data generator.
    params = {'dim': (257, None, 1),
              'nfft': 512,
              'spec_len': 250,
              'win_length': 400,
              'hop_length': 160,
              'n_classes': 5994,
              'sampling_rate': 16000,
              'normalize': True,
              }

    network_eval = model.vggvox_resnet2d_icassp(input_dim=params['dim'],
                                                num_class=params['n_classes'],
                                                mode='eval', args=args)

    # ==> load pre-trained model ???
    if args.resume:
        # ==> get real_model from arguments input,
        # load the model if the imag_model == real_model.
        if os.path.isfile(args.resume):
            network_eval.load_weights(os.path.join(args.resume), by_name=True)
            result_path = set_result_path(args)
            print('==> successfully loading model {}.'.format(args.resume))
        else:
            raise IOError("==> no checkpoint found at '{}'".format(args.resume))
    else:
        raise IOError('==> please type in the model to load')

    print('==> start testing.')

    # The feature extraction process has to be done sample-by-sample,
    # because each sample is of different lengths.
    total_length = len(unique_list)
    feats, scores, labels = [], [], []
    for c, ID in enumerate(unique_list):
        if c % 50 == 0: print('Finish extracting features for {}/{}th wav.'.format(c, total_length))
        specs = ut.load_data(ID, win_length=params['win_length'], sr=params['sampling_rate'],
                             hop_length=params['hop_length'], n_fft=params['nfft'],
                             spec_len=params['spec_len'], mode='eval')
        specs = np.expand_dims(np.expand_dims(specs, 0), -1)
    
        v = network_eval.predict(specs)
        feats += [v]
    
    feats = np.array(feats)

    # ==> compute the pair-wise similarity.
    for c, (p1, p2) in enumerate(zip(list1, list2)):
        ind1 = np.where(unique_list == p1)[0][0]
        ind2 = np.where(unique_list == p2)[0][0]

        v1 = feats[ind1, 0]
        v2 = feats[ind2, 0]

        scores += [np.sum(v1*v2)]
        labels += [verify_lb[c]]
        print('scores : {}, gt : {}'.format(scores[-1], verify_lb[c]))

    scores = np.array(scores)
    labels = np.array(labels)

    np.save(os.path.join(result_path, 'prediction_scores.npy'), scores)
    np.save(os.path.join(result_path, 'groundtruth_labels.npy'), labels)

    eer, thresh = toolkits.calculate_eer(labels, scores)
    print('==> model : {}, EER: {}'.format(args.resume, eer))


def set_result_path(args):
    model_path = args.resume
    exp_path = model_path.split(os.sep)
    result_path = os.path.join('../result', exp_path[2], exp_path[3])
    if not os.path.exists(result_path): os.makedirs(result_path)
    return result_path

class SR_Model():
    def __init__(self, model_path, params, args):
        self.model_path = model_path
        self.params = params
        self.args = args
        self.model = None
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    def get_feature(self, specs):
        # specs = tf.convert_to_tensor(specs, dtype=tf.float64)
        specs = np.array(specs)
        specs = np.squeeze(specs, 1)
        v = self.model.predict(specs, batch_size=32)
        # return specs
        return v
    
    def load_model(self):
        self.model = model.vggvox_resnet2d_icassp(input_dim=self.params['dim'],
                                                num_class=self.params['n_classes'],
                                                mode='eval', args=self.args)
        self.model.load_weights(os.path.join(self.model_path), by_name=True)
        print('load model successfully')

    def reload_model(self):
        K.clear_session()
        tf.reset_default_graph()
        self.model = model.vggvox_resnet2d_icassp(input_dim=self.params['dim'],
                                                num_class=self.params['n_classes'],
                                                mode='eval', args=self.args)
        self.model.load_weights(os.path.join(self.model_path), by_name=True)

def convert_audio(params, audio_path):
    specs = ut.load_data(audio_path, win_length=params['win_length'], sr=params['sampling_rate'],
                            hop_length=params['hop_length'], n_fft=params['nfft'],
                            spec_len=params['spec_len'], mode='eval')
    specs = np.expand_dims(np.expand_dims(specs, 0), -1)

    return specs

def convert_audio_processing(sr_model, file_path_list):
    worker_pool = multiprocessing.Pool(processes = 24)
    specs_list = []
    process_res = []
    for path in file_path_list:
        res = worker_pool.apply_async(convert_audio, [sr_model.params, path])
        process_res.append(res)
    worker_pool.close()
    worker_pool.join()
    for res in process_res:
        specs_list.append(res.get())

    return specs_list

if __name__ == "__main__":
    model_path = '/home/dslab/hx/audio_detection/DeepSonar/SR_module/resnet34_vlad8_ghost2_bdim512_deploy/weights.h5'

    sr_args = {'batch_size':16,
               'net':'resnet34s',
               'ghost_cluster':2,
               'vlad_cluster':8,
               'bottleneck_dim':512,
               'aggregation_mode':'gvlad',
               'loss':'softmax',
               'test_type':'normal'
                }
    params = {'dim': (257, None, 1),
              'nfft': 512,
              'spec_len': 250,
              'win_length': 400,
              'hop_length': 160,
              'n_classes': 5994,
              'sampling_rate': 16000,
              'normalize': True,
              }
    
    network_eval = model.vggvox_resnet2d_icassp(input_dim=params['dim'],
                                                num_class=params['n_classes'],
                                                mode='eval', args=sr_args)
    network_eval.load_weights(os.path.join(model_path), by_name=True)
    print('==> successfully loading model {}.'.format(model_path))

    audio_path = '/home/dslab/hx/vfuzz/media/data/for-2seconds/training/fake/file6.mp3.wav_16k.wav_norm.wav_mono.wav_silence.wav_2sec.wav'
    specs = ut.load_data(audio_path, win_length=params['win_length'], sr=params['sampling_rate'],
                             hop_length=params['hop_length'], n_fft=params['nfft'],
                             spec_len=params['spec_len'], mode='eval')
    specs = np.expand_dims(np.expand_dims(specs, 0), -1)
    
    v = network_eval.predict(specs)

    v = ut.reshape(v)
    feature_matrix = ut.cut(v)
    count = 1
    for i in v:
        print('layer {}:{}'.format(count, i.shape))
        count += 1
