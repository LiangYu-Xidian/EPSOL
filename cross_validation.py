import tensorflow as tf
import keras
import numpy as np
import random
import pickle
import os
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, auc,matthews_corrcoef

import models.EPSOL_graph as Models
import models.utils as utils

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def set_seed(seed):
    tf.set_random_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)

def load_data():
    dataset = pickle.load(open("./data/data_all.pkl", 'rb'))
    return dataset


def get_callbacks():
    stopping = utils.get_early_stopping_cbk(monitor='val_loss', patience=3)
    model_path = './result/model/' + model_name + '.hdf5'
    checkpointer = utils.get_model_checkpoint(model_path,
                                              verbose=1,
                                              save_best_only=True)
    return [stopping, checkpointer]

def get_probabilities(best_model, x):
    probabilities = best_model.predict(x, batch_size=64)
    return probabilities

def get_classification_prediction(best_model, x, label):
    pred_probs = get_probabilities(best_model, x)
    preds = pred_probs.argmax(axis=-1)
    label = label.argmax(axis=-1)
    acc = accuracy_score(label, preds)
    mcc = matthews_corrcoef(label, preds)
    return [preds, pred_probs,acc,mcc,label]


def save_classification_prediction(prediction_class,prediction_prob,label):
    reports_dir = "./result/predict_file/"
    report_path =  reports_dir + model_name +  '_prediction.txt'
    with open(report_path,'w') as f:
        f.write('Predicted_Class'+'\t'+'P0'+'\t'+'P1'+'\t'+'label'+'\n')
        for i in range(0,len(prediction_class)):
            f.write(str(prediction_class[i])+'\t'+str(prediction_prob[i][0])+'\t'+str(prediction_prob[i][1])+'\t'+str(label[i])+'\n')
    f.close()


def make_data(seq, bi, tri, acc, acc20, ss, ss8, bio, label):

    x_seq = np.array(utils.pad_seq(seq, maxlen))
    x_bi = np.array(utils.pad_seq(bi, maxlen))
    x_tri = np.array(utils.pad_seq(tri, maxlen))

    x_acc = np.array(utils.pad_seq(acc, maxlen))
    x_acc20 = np.array(utils.pad_seq(acc20, maxlen))
    x_ss = np.array(utils.pad_seq(ss, maxlen))
    x_ss8 = np.array(utils.pad_seq(ss8, maxlen))

    x_bio = bio[:, :-1]
    y = utils.get_one_hot(label, num_classes)
    print("-----------------------------------------------")
    print("x_seq shape: {}".format(x_seq.shape))
    print("x_bi  shape: {}".format(x_bi.shape))
    print("x_tri shape: {}".format(x_tri.shape))
    print("x_acc shape: {}".format(x_acc.shape))
    print("x_acc20 shape: {}".format(x_acc20.shape))
    print("x_ss shape: {}".format(x_ss.shape))
    print("x_ss8 shape: {}".format(x_ss8.shape))
    print("x_bio shape: {}".format(x_bio.shape))
    print("label shape: {}".format(label.shape))
    return [x_seq, x_bi, x_tri, x_acc, x_acc20, x_ss, x_ss8, x_bio], y

def make_fold(x_full,y_full,train_index,test_index):
    x_train = []
    x_test = []
    for i in range(len(x_full)):
        # print(x_full[i].shape)
        x_train.append(x_full[i][train_index])
        x_test.append(x_full[i][test_index])
    
    return x_train,y_full[train_index],x_test,y_full[test_index]
    


def main():

    print("load data...")
    data = load_data()

    x_train_seq, x_train_bi, x_train_tri, x_train_acc, x_train_acc20, x_train_ss, x_train_ss8, x_train_bio, y_train = \
        data['train']['seq'], data['train']['bigram'], data['train']['trigram'], \
        data['train']['acc'], data['train']['acc20'], data['train']['ss'], data['train']['ss8'], \
        data['train']['src_bio'], data['train']['label']

    x_dev_seq, x_dev_bi, x_dev_tri, x_dev_acc, x_dev_acc20, x_dev_ss, x_dev_ss8, x_dev_bio, y_dev = \
        data['dev']['seq'], data['dev']['bigram'], data['dev']['trigram'], \
        data['dev']['acc'], data['dev']['acc20'], data['dev']['ss'], data['dev']['ss8'], \
        data['dev']['src_bio'], data['dev']['label']

    x_seq_full = np.concatenate((x_train_seq,x_dev_seq),axis=0)
    x_bi_full = np.concatenate((x_train_bi,x_dev_bi),axis=0)
    x_tri_full = np.concatenate((x_train_tri,x_dev_tri),axis=0)
    x_acc_full = np.concatenate((x_train_acc,x_dev_acc),axis=0)
    x_acc20_full = np.concatenate((x_train_acc20,x_dev_acc20),axis=0)
    x_ss_full = np.concatenate((x_train_ss,x_dev_ss),axis=0)
    x_ss8_full = np.concatenate((x_train_ss8,x_dev_ss8),axis=0)
    x_bio_full = np.concatenate((x_train_bio,x_dev_bio),axis=0)
    y_full = np.concatenate((y_train,y_dev),axis=0)


    print("make data...")
    x_full, y_oh_full = make_data(
        x_seq_full, x_bi_full, x_tri_full, x_acc_full, x_acc20_full, x_ss_full, x_ss8_full, x_bio_full, y_full)

    kf = KFold(n_splits=10,shuffle=True,random_state=2021)
    count = 1
    acc_list = []
    mcc_list = []
    for train_index,test_index in kf.split(y_oh_full):
        print('Starting CV Iteration: ',str(count))
        global model_name 
        model_name = "cv_fold_" + str(count)
        print(model_name)

        x_train,y_train,x_test,y_test = make_fold(x_full,y_oh_full,train_index,test_index)

        model = Models.EPSOL().get_model()
        model.compile(loss='binary_crossentropy',
                      optimizer=utils.get_adam_optim(), metrics=['accuracy'])
        # print(model.summary())

        model.fit(x_train, y_train, batch_size=64,
                  epochs=10,
                  validation_split=0.1,
                  callbacks=get_callbacks())
    
        filepath = './result/model/' + model_name + '.hdf5'

        best_model = utils.load_model(filepath)

        [pred_test,pred_prob_test,acc,mcc,label] = get_classification_prediction(best_model,x_test,y_test)
        print("-------------------------------------------------------")
        print("ACC of fold-{} cross-validation: {}".format(str(count),acc))
        print("MCC of fold-{} cross-validation: {}".format(str(count),mcc))
        acc_list.append(acc)
        mcc_list.append(mcc)
        save_classification_prediction(pred_test,pred_prob_test,label)
        print("save result ok!")
        count = count+1
    
    mean_acc = (1.0*sum(acc_list))/len(acc_list)
    mean_mcc = (1.0*sum(mcc_list))/len(mcc_list)
    print("-------------------------------------------------------")
    print("Mean ACC of 10 fold cross-validation: {}".format(mean_acc))
    print("Mean MCC of 10 fold cross-validation: {}".format(mean_mcc))


        





if __name__ == "__main__":
    set_seed(2021)

    maxlen = 1200
    num_classes = 2

    main()
