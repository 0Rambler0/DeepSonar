import sys
sys.path.append('/home/dslab/hx/audio_detection/DeepSonar/SR_module')
import numpy as np
from classifer import *
from predict import SR_Model

def evaluate(model, eval_set, eval_labels):
    FR_count = 0
    FA_count = 0
    error_count = 0
    count = 0
    total = len(eval_labels)

    for data, label in zip(eval_set, eval_labels):
        count += 1
        data = np.expand_dims(data, 0)
        predict = model.predict(data)
        predict = predict[0]
        print('sample {}  predict label:{} real label:{}'.format(count, predict, label))
        if predict != label:
            error_count += 1 
            if label:
                FA_count += 1
            else:
                FR_count += 1
    FR_rate = FR_count/total
    FA_rate = FA_count/total
    acc = error_count/total

    return FA_rate, FR_rate, acc

if __name__ == '__main__':
    eval_path = '/home/dslab/hx/vfuzz/media/data/SR_feature/valid.tfrecords'
    eval_set, eval_labels = read_data(eval_path)
    print('load eval data successfully')
    model = create_model()
    model.load_weights('/home/dslab/hx/audio_detection/DeepSonar/model/weight.h5')
    FA_rate, FR_rate, acc = evaluate(model, eval_set, eval_labels)
    print('FA_rate:{}  FR_rate:{}  ACC:{}'.format(FA_rate, FR_rate, acc))