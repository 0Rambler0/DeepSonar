import sys
sys.path.append('/home/dslab/hx/audio_detection/DeepSonar/SR_module')
from predict import SR_Model, convert_audio_processing
from classifer import *
from data_utils import *
from sr_model_param import *

def create_raw_data(sr_model, dataset_path, save_path, type):
    dataset_path = os.path.join(dataset_path, type)
    # writer = tf.python_io.TFRecordWriter(save_path)
    feature_list = []
    labels = []

    real_path = os.path.join(dataset_path, 'real')
    fake_path = os.path.join(dataset_path, 'fake')
    real_list = os.listdir(real_path)
    fake_list = os.listdir(fake_path)
   
    count = 0
    last_reload_count = 0
    # print('start get net param')
    sr_model.load_model()
    # net_param = load_net_param('param.pkl')
    # threshold_list = search_threshold(net_param['sum'], net_param['num_sample'], net_param['num_neuron'])
    # print('get net param complete')
    feature_list = []
    labels = []
    path_list = []
    for file in real_list:
        # print('{} file {}:{}'.format(type, count, file))
        path_list.append(os.path.join(real_path, file))
        if len(path_list) == 100:
            start_time = time.time()
            specs_list = convert_audio_processing(sr_model, path_list)
            feature_data = sr_model.get_feature(specs_list)
            feature_list = []
            for i in range(len(feature_data[0])):
                feature = []
                for j in range(7):
                    feature.append(feature_data[j][i])
                feature = reshape(feature)
                feature = cut(feature)
                feature_list.append(feature)            
            for path in path_list:
                file = path.split('/')[-1]
                count += 1
                # print('{} file {}:{}'.format(type, count, file))
                labels.append(1)
            end_time = time.time()
            print('process_rate: {}%({}/{})  run time:{}s'.format(100*count/(len(real_list)+len(fake_list)),
                                                                  count,
                                                                  len(real_list)+len(fake_list),
                                                                  end_time-start_time))
            path_list = []
        if len(feature_list) > 0:
            # activation_matrix_list = make_activation_matrix(feature_list, threshold_list)
            write_data_pkl(feature_list, labels, save_path)
            # objgraph.show_growth()
            feature_list = []
            labels = []
            specs_list = []
            gc.collect()
            # objgraph.show_most_common_types()
            # objgraph.show_growth()
            # objgraph.get_leaking_objects()
            # objgraph.show_backrefs(specs_list, max_depth=5, filename='{}.dot'.format(count))
        if count - last_reload_count > 1000:
            sr_model.reload_model()
            last_reload_count = count
    specs_list = convert_audio_processing(sr_model, path_list)
    feature_data = sr_model.get_feature(specs_list)
    feature_list = []
    for i in range(len(feature_data[0])):
        feature = []
        for j in range(7):
            feature.append(feature_data[j][i])
        feature = reshape(feature)
        feature = cut(feature)
        feature_list.append(feature)            
    for path in path_list:
        file = path.split('/')[-1]
        count += 1
        # print('{} file {}:{}'.format(type, count, file))
        labels.append(1)
    path_list = []   
    for file in fake_list:
        path_list.append(os.path.join(fake_path, file))
        if len(path_list) == 100:
            start_time = time.time()
            specs_list = convert_audio_processing(sr_model, path_list)
            feature_data = sr_model.get_feature(specs_list)
            feature_list = []
            for i in range(len(feature_data[0])):
                feature = []
                for j in range(7):
                    feature.append(feature_data[j][i])
                feature = reshape(feature)
                feature = cut(feature)
                feature_list.append(feature)            
            for path in path_list:
                file = path.split('/')[-1]
                count += 1
                # print('{} file {}:{}'.format(type, count, file))
                labels.append(0)
            end_time = time.time()
            print('process_rate: {}%({}/{})  run time:{}s'.format(100*count/(len(real_list)+len(fake_list)),
                                                                  count,
                                                                  len(real_list)+len(fake_list),
                                                                  end_time-start_time))
            path_list = []
        if len(feature_list) > 0:
            # activation_matrix_list = make_activation_matrix(feature_list, threshold_list)
            # write_data(writer, labels, activation_matrix_list)
            write_data_pkl(feature_list, labels, save_path)
            feature_list = []
            labels = []
            specs_list = []
            gc.collect()
        if count - last_reload_count > 1000:
            sr_model.reload_model()
            last_reload_count = count
    feature_data = sr_model.get_feature(specs_list)
    feature_list = []
    for i in range(len(feature_data[0])):
        feature = []
        for j in range(7):
            feature.append(feature_data[j][i])
        feature = reshape(feature)
        feature = cut(feature)
        feature_list.append(feature)            
    for path in path_list:
        file = path.split('/')[-1]
        count += 1
        # print('{} file {}:{}'.format(type, count, file))
        labels.append(0)
    # activation_matrix_list = make_activation_matrix(feature_list, threshold_list)
    # write_data(writer, labels, activation_matrix_list)
    write_data_pkl(feature_list, labels, save_path)
    # writer.close()
    print('complete')

    return

if __name__ == '__main__':
    dataset_path = '/home/dslab/hx/vfuzz/media/data/for-2seconds'
    train_save_path = '/home/dslab/hx/vfuzz/media/data/SR_feature/train_raw.pkl'
    test_save_path = '/home/dslab/hx/vfuzz/media/data/SR_feature/test_raw.pkl'
    valid_save_path = '/home/dslab/hx/vfuzz/media/data/SR_feature/valid.tfrecords'

    sr_model = SR_Model(sr_model_path, sr_params,sr_args)
    # create_raw_data(sr_model, dataset_path, train_save_path, 'training')
    # create_raw_data(sr_model, dataset_path, test_save_path, 'testing')
    create_raw_data(sr_model, dataset_path, valid_save_path, 'validation')
    print('complete')
