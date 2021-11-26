import os
import sys

from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
from tensorflow.keras.optimizers import Adam
import utils
from keras.callbacks import EarlyStopping
from nettcr_architectures import nettcr_ab
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras.metrics
import random

# Options for Pandas DataFrame printing
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', -1)

from argparse import ArgumentParser

# ----------------------------输入参数----------------------------------------
parser = ArgumentParser(description="设定输入参数")
parser.add_argument("-c", "--chain", default="ab", help="默认使用ab链")

# 设置输入输出文件位置
parser.add_argument("-tr", "--trainfile", default="train_set.csv", help="设定训练集")
parser.add_argument("-te", "--testfile", default="gig_test.csv", help="设定测试集")
parser.add_argument("-o", "--outfile", default=sys.stdout, help="设定输出文件夹")

# 设置超参
parser.add_argument("-e", "--epochs", default=1, type=int, help="设定训练时期数")
parser.add_argument("-lr", "--learn_rate", default=0.001, type=float, help="设定优化算法学习率")
parser.add_argument("-bs", "--batch_size", default=128, type=int, help="批大小")

args = parser.parse_args()
LEARN_RATE = float(args.learn_rate)
EPOCHS = int(args.epochs)
BATCH_SIZE = int(args.batch_size)

output_path = 'data/model_output/lr'+str(LEARN_RATE)+'BS'+str(BATCH_SIZE)+'EP'+str(EPOCHS)+'/'

# ----------------------------数据处理----------------------------------------
print('Loading and encoding the data..')
train_data = pd.read_csv(args.trainfile)
test_data = pd.read_csv(args.testfile)

# Encode data
encoding = utils.blosum50_20aa
early_stop = EarlyStopping(monitor='loss', min_delta=0,
                           patience=10, verbose=0, mode='min', restore_best_weights=True)

pep_train = utils.enc_list_bl_max_len(train_data.peptide, encoding, 9)
tcra_train = utils.enc_list_bl_max_len(train_data.CDR3a, encoding, 30)
tcrb_train = utils.enc_list_bl_max_len(train_data.CDR3b, encoding, 30)
y_train = np.array(train_data.binder)

pep_test = utils.enc_list_bl_max_len(test_data.peptide, encoding, 9)
tcra_test = utils.enc_list_bl_max_len(test_data.CDR3a, encoding, 30)
tcrb_test = utils.enc_list_bl_max_len(test_data.CDR3b, encoding, 30)
train_inputs = [tcra_train, tcrb_train, pep_train]
test_inputs = [tcra_test, tcrb_test, pep_test]

# ----------- 训练模型 ----------
mdl = nettcr_ab()

METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
    keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
]

mdl.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=LEARN_RATE), metrics=METRICS)
print('Training..')

# history = mdl.fit(train_inputs, y_train,
#                   epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, callbacks=[early_stop])


def plot_graphs(historysum, string):
    history = []
    valhis = []
    for t in range(EPOCHS):
        history.append(historysum[t].history[string])
        valhis.append(historysum[t].history["val_"+string])

    plt.plot(history, 'r')
    plt.plot(valhis, 'b')

    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()


# plot_graphs(history, 'loss')

# htr = []

# # 可以在这里设置EPOCHS、BATCH的衰减
# for i in range(EPOCHS):
#     BATCH_SIZE = BATCH_SIZE - 1
#     history2 = mdl.fit(train_inputs, y_train, epochs=1, batch_size=BATCH_SIZE, verbose=1, validation_split=0.2,
#                        callbacks=[early_stop])
#     type(history2)
#     history2.history
#     htr.append(history2)



train_batches = []
y_train_batches = []
batch_his = []
epoch_his = []
history = []

# 区分正负样本
pos_index = np.nonzero(y_train)[0]
neg_index = np.nonzero(~y_train)[0]
rd = []
rate = len(pos_index)/len(y_train)

# TODO 把数据分批 每次训练一批 并记录批次
# 可以在这里设置EPOCHS、BATCH的衰减
for i in range(EPOCHS):
    for j in range(int(len(y_train)/BATCH_SIZE)):
        pos_index_batches = random.sample(pos_index.tolist(), int(BATCH_SIZE*rate))
        neg_index_batches = random.sample(neg_index.tolist(), int(BATCH_SIZE - BATCH_SIZE*rate))
        index_bathes = pos_index_batches + neg_index_batches
        train_batches = [train_inputs[0][np.array(index_bathes)], train_inputs[1][np.array(index_bathes)], train_inputs[2][np.array(index_bathes)]]
        y_train_batches = y_train[np.array(index_bathes)]
        history.append(mdl.fit(train_batches, y_train_batches, batch_size=BATCH_SIZE, verbose=1, callbacks=[early_stop]))
        mdl.save(output_path + 'trained_ep'+str(i)+'.tf2')

his_s = []
for his in history:
    his_s.append(his.history)
np.save(output_path+'history/his_s.npy', his_s, allow_pickle=True)

# print('Evaluating..')
#
# # --------------- 在测试集上使用模型 ----------------
# preds = mdl.predict(test_inputs, verbose=0)
# pred_df = pd.concat([test_data, pd.Series(np.ravel(preds), name='prediction')], axis=1)
#
# pred_df.to_csv(args.outfile, index=False)
