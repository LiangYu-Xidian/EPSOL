import pandas as pd
import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer
import tensorflow as tf
import keras
import os
import models.utils as utils
import sys

def tran_fasta_to_df(fasta_dir,type_name):
    f = open(fasta_dir)
    seq= []
    ids = []
    for line in f:
        if line.startswith('>'):
            name=line.replace('>','').split()
            ids.append(name[0])
        else:
            seq.append(line.replace('\n','').strip())
    assert(len(ids)==len(seq))
    f.close()
    df = pd.DataFrame(data={'id': ids ,type_name:seq}) 
    return df

def N_gram(seq,n):
    res = []
    for i in range(len(seq)-n+1):
        res.append(seq[i:i+n])
    return " ".join(res)


def gen_data(seq_file,ss_file,ss8_file,acc_file,acc20_file,src_bio_file):
    test_acc = tran_fasta_to_df( acc_file,"acc")
    test_acc20 = tran_fasta_to_df( acc20_file,"acc20")
    test_ss = tran_fasta_to_df( ss_file,"ss") 
    test_ss8 = tran_fasta_to_df( ss8_file,"ss8")
    test_seq = tran_fasta_to_df( seq_file,"seq")

    test_df = pd.concat([test_seq,test_acc["acc"],test_acc20["acc20"],test_ss["ss"],test_ss8["ss8"]],axis=1)

    test_df["bigram"] = test_df['seq'].apply(N_gram,n=2)
    test_df["trigram"] = test_df['seq'].apply(N_gram,n=3)

    tok = pickle.load(open("./utils/tok.pkl", 'rb'))
    tok_bigram = pickle.load(open("./utils/tok_bigram.pkl", 'rb'))
    tok_trigram = pickle.load(open("./utils/tok_trigram.pkl", 'rb'))
    tok_acc = pickle.load(open("./utils/tok_acc.pkl", 'rb'))
    tok_acc20 = pickle.load(open("./utils/tok_acc20.pkl", 'rb'))
    tok_ss = pickle.load(open("./utils/tok_ss.pkl", 'rb'))
    tok_ss8 = pickle.load(open("./utils/tok_ss8.pkl", 'rb'))

    test = {}
    test["seq"] = tok.texts_to_sequences(test_df["seq"].values)
    test["bigram"] = tok_bigram.texts_to_sequences(test_df["bigram"].values)
    test["trigram"] = tok_trigram.texts_to_sequences(test_df["trigram"].values)

    test["acc"] = tok_acc.texts_to_sequences(test_df["acc"].values)
    test["acc20"] = tok_acc20.texts_to_sequences(test_df["acc20"].values)
    test["ss"] = tok_ss.texts_to_sequences(test_df["ss"].values)
    test["ss8"] = tok_ss8.texts_to_sequences(test_df["ss8"].values)

    test["src_bio"] = np.loadtxt(src_bio_file)
    save_data = { "test": test }

    return save_data

def make_data(seq, bi, tri, acc, acc20, ss, ss8, bio):

    x_seq = np.array(utils.pad_seq(seq, maxlen))
    x_bi = np.array(utils.pad_seq(bi, maxlen))
    x_tri = np.array(utils.pad_seq(tri, maxlen))

    x_acc = np.array(utils.pad_seq(acc, maxlen))
    x_acc20 = np.array(utils.pad_seq(acc20, maxlen))
    x_ss = np.array(utils.pad_seq(ss, maxlen))
    x_ss8 = np.array(utils.pad_seq(ss8, maxlen))

    x_bio = bio[:, :-1]
    print("-----------------------------------------------")
    print("x_seq shape: {}".format(x_seq.shape))
    print("x_bi  shape: {}".format(x_bi.shape))
    print("x_tri shape: {}".format(x_tri.shape))
    print("x_acc shape: {}".format(x_acc.shape))
    print("x_acc20 shape: {}".format(x_acc20.shape))
    print("x_ss shape: {}".format(x_ss.shape))
    print("x_ss8 shape: {}".format(x_ss8.shape))
    print("x_bio shape: {}".format(x_bio.shape))
    return [x_seq, x_bi, x_tri, x_acc, x_acc20, x_ss, x_ss8, x_bio]

def get_probabilities(best_model, x):
    probabilities = best_model.predict(x, batch_size=64)
    return probabilities

def get_classification_prediction(best_model, x):
    pred_probs = get_probabilities(best_model, x)
    preds = pred_probs.argmax(axis=-1)
    return [preds, pred_probs]


def save_classification_prediction(prediction_class,prediction_prob,output_file):
    reports_dir = "./result/predict_file/"
    report_path =  reports_dir + output_file +  '_prediction.txt'
    with open(report_path,'w') as f:
        f.write('Predicted_Class'+'\t'+'P0'+'\t'+'P1'+'\n')
        for i in range(0,len(prediction_class)):
            f.write(str(prediction_class[i])+'\t'+str(prediction_prob[i][0])+'\t'+str(prediction_prob[i][1])+'\n')
    f.close()

def main(seq_file,ss_file,ss8_file,acc_file,acc20_file,src_bio_file,output_file):
    print("load data...")
    data = gen_data(seq_file,ss_file,ss8_file,acc_file,acc20_file,src_bio_file)

    x_test_seq, x_test_bi, x_test_tri, x_test_acc, x_test_acc20, x_test_ss, x_test_ss8, x_test_bio = \
        data['test']['seq'], data['test']['bigram'], data['test']['trigram'], \
        data['test']['acc'], data['test']['acc20'], data['test']['ss'], data['test']['ss8'], \
        data['test']['src_bio']

    print("make data...")

    x_test = make_data(
        x_test_seq, x_test_bi, x_test_tri, x_test_acc, x_test_acc20, x_test_ss, x_test_ss8, x_test_bio)
    

    filepath = './result/model/EPSOL.hdf5'

    best_model = utils.load_model(filepath)

    [pred_test,pred_prob_test] = get_classification_prediction(best_model,x_test)
    save_classification_prediction(pred_test,pred_prob_test,output_file)
    print("-----------------------------------------------")
    print("EPSOL prediction finished!")
    print("-----------------------------------------------")

if __name__ == "__main__":
    maxlen = 1200
    main(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],sys.argv[7])