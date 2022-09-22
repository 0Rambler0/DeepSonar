import sys
from matplotlib import test
sys.path.append('/home/dslab/hx/audio_detection/DeepSonar/SR_module')
import numpy as np
from classifer import *
from predict import SR_Model

def load_train_data(path):
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
    feature_list = np.array(feature_list)
    labels = np.array(labels)

    return feature_list, labels

if __name__ == '__main__':
    train_datapath = '/home/dslab/hx/vfuzz/media/data/SR_feature/train_new.pkl'
    test_datapath = '/home/dslab/hx/vfuzz/media/data/SR_feature/test.pkl'
    
    train_set, train_labels = load_train_data(train_datapath)
    print('load train data successfully')
    test_set, test_labels = load_train_data(test_datapath)
    print('load test data successfully')
    
    # train_set = train_set[:,:,896:1009]
    # test_set = test_set[:,:,896:1009]
    train_index = [i for i in range(len(train_set))]
    test_index = [i for i in range(len(test_set))]
    np.random.shuffle(train_index)
    np.random.shuffle(test_index)
    train_set = train_set[train_index]
    train_labels = train_labels[train_index]
    test_set = test_set[test_index]
    test_labels = test_labels[test_index]
    
    model = create_model()
    model_save_path = '/home/dslab/hx/audio_detection/DeepSonar/model/'
    # predict = model.predict(train_set[0])
    train(model, train_set, train_labels, test_set, test_labels, 128, 100, model_save_path)
    print('complete')