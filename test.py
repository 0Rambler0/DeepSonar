import pickle


def load_feature(path):
    feature_list = []
    labels = []

    with open(path ,'rb') as f:
        while True:
            try:
                data_dict = pickle.load(f, encoding='iso-8859-1')
                feature_list.extend(data_dict['data'])
                labels.extend(data_dict['labels'])
            except EOFError:
                break

    return feature_list, labels

def write_data_pkl(activation_matrix_list, labels, save_path):
    data_dict = {'data':activation_matrix_list[7000:], 'labels':labels[7000:]}
    with open(save_path, 'ab') as f:
        pickle.dump(data_dict, f, protocol = 2)
    print('write {} data'.format(len(labels)))

if __name__ == '__main__':
    path = '/home/dslab/hx/vfuzz/media/data/SR_feature/train.pkl'
    save_path = '/home/dslab/hx/vfuzz/media/data/SR_feature/train_new.pkl'
    feature_list, labels = load_feature(path)
    write_data_pkl(feature_list, labels, save_path)   
    # fake = min(labels)
    print('complete')