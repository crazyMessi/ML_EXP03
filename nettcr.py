import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from matplotlib import pyplot as plt
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
from argparse import ArgumentParser
import math

# Options for Pandas DataFrame printing
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', -1)

# -------------------------------------------------- 输入参数 ------------------------------------------------------------
parser = ArgumentParser(description="设定输入参数")
parser.add_argument("-n", "--sp_name", default="", help="模型命名")
# 设置输入输出文件位置
parser.add_argument("-tr", "--train_file", default="data/model_input/raw_data/train_set.csv", help="设定训练集")
parser.add_argument("-te", "--test_file", default="data/model_input/raw_data/gig_test.csv", help="设定测试集")
# 设置超参
parser.add_argument("-e", "--epochs", default=1, type=int, help="设定训练时期数")
parser.add_argument("-lr", "--learn_rate", default=0.001, type=float, help="设定优化算法学习率")
parser.add_argument("-bs", "--batch_size", default=128, type=int, help="批大小")

# 设置测试or训练
parser.add_argument("-t", "--if_skip_train", default=-1, type=int, help="如果大于等于0，则读取第参数批的模型进行测试")

args = parser.parse_args()
LEARN_RATE = float(args.learn_rate)
EPOCHS = int(args.epochs)
BATCH_SIZE = int(args.batch_size)
special_name = args.sp_name
if_skip_train = args.if_skip_train

output_path = 'data/model_output/' + special_name + 'lr' + str(LEARN_RATE) + 'bs' + str(BATCH_SIZE) + 'ep' + str(
    EPOCHS) + '/'
train_model_path = output_path + 'trained_model/'
his_path = output_path + 'history/'
if not os.path.exists(output_path):
    os.makedirs(output_path)
if not os.path.exists(train_model_path):
    os.mkdir(train_model_path)
if not os.path.exists(his_path):
    os.mkdir(his_path)


def plot_graphs(History, string):
    his_item = []
    val_his = []
    for t in range(EPOCHS):
        his_item.append(History[t].history[string])
        val_his.append(History[t].history["val_" + string])

    plt.plot(his_item, 'r')
    plt.plot(val_his, 'b')
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()


# --------------------------------------------------- 数据处理 -----------------------------------------------------------
print('Loading and encoding the data..')
train_data = pd.read_csv(args.train_file)
test_data = pd.read_csv(args.test_file)

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
y_test = np.array(test_data.binder)

train_inputs = [tcra_train, tcrb_train, pep_train]
test_inputs = [tcra_test, tcrb_test, pep_test]

mdl = []
# --------------------------------------------------- 训练模型 -----------------------------------------------------------
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
if if_skip_train < 0:
    mdl = nettcr_ab()
    mdl.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=LEARN_RATE), metrics=METRICS)
    print('Training..')
    train_batches = []
    y_train_batches = []
    # 每个时期的训练状况
    epoch_his = []
    history = []

    # 区分正负样本
    pos_index = np.nonzero(y_train)[0]
    neg_index = np.nonzero(~y_train)[0]
    # 划分验证集
    valid_rate = 0.2
    valid_pos_index_index = np.random.randint(0, np.size(pos_index) - 1, int(BATCH_SIZE * valid_rate / 2))
    valid_neg_index_index = np.random.randint(0, np.size(neg_index) - 1, int(BATCH_SIZE * valid_rate / 2))
    valid_pos_index = pos_index[valid_pos_index_index]
    valid_neg_index = neg_index[valid_neg_index_index]
    valid_index_batches = valid_pos_index.tolist() + valid_neg_index.tolist()
    valid_batches = [train_inputs[0][np.array(valid_index_batches)], train_inputs[1][np.array(valid_index_batches)],
                     train_inputs[2][np.array(valid_index_batches)]]
    pos_index = np.delete(pos_index, valid_pos_index_index)
    neg_index = np.delete(neg_index, valid_neg_index_index)

    # 可以在这里设置EPOCHS、BATCH的衰减
    for i in range(EPOCHS):
        for j in range(int(len(y_train) / BATCH_SIZE)):
            # 划分测试集
            pos_index_batches = random.sample(pos_index.tolist(), int(0.5*BATCH_SIZE))
            neg_index_batches = random.sample(neg_index.tolist(), BATCH_SIZE - int(0.5*BATCH_SIZE))
            index_bathes = pos_index_batches + neg_index_batches
            train_batches = [train_inputs[0][np.array(index_bathes)], train_inputs[1][np.array(index_bathes)],
                             train_inputs[2][np.array(index_bathes)]]
            y_train_batches = y_train[np.array(index_bathes)]
            # 训练一个批次，记录批次日志
            epoch_his.append(mdl.fit(train_batches, y_train_batches, batch_size=BATCH_SIZE, verbose=1,
                                     callbacks=[early_stop], validation_data=valid_batches).history)
        # 存储该时期模型
        mdl.save(train_model_path + 'trained_ep' + str(i) + '.tf2')
        # 存储该时期模型训练日志
        ep_his_path = his_path + 'ep' + str(i) + 'his.npy'
        with open(ep_his_path, 'wb') as f:
            np.save(f, epoch_his, allow_pickle=True)
        history.append(epoch_his)
        epoch_his = []

    his_path = his_path + 'final_his.npy'
    try:
        with open(his_path, 'wb') as f:
            np.save(f, history, allow_pickle=True)
    except ImportError as e:
        print("错误")


else:
    print("跳过训练，读取模型文件测试")
    for i in range(if_skip_train):
        model_name = train_model_path + "trained_ep" + str(i) + '.tf2'
        try:
            mdl.append(keras.models.load_model(model_name))
        except ImportError as e:
            print(model_name + "不存在")


# -------------------------------------------------- 测试模型 ------------------------------------------------------------
def eval_mdl(test_mdl):
    # --------------- 在测试集上使用模型 ----------------
    pred = test_mdl.predict(test_inputs, verbose=0)
    pred_df = pd.concat([test_data, pd.Series(np.ravel(pred), name='prediction')], axis=1)
    return pred
    # pred_df.to_csv(args.outfile, index=False)


if mdl:
    print('Evaluating..')
    pred_y = []
    if if_skip_train > 0:
        for i in range(if_skip_train):
            model_name = train_model_path + "trained_ep" + str(i) + '.tf2'
            print("测试读取的模型：" + model_name)
            pred_y.append(eval_mdl(mdl[i]))
    else:
        pred_y.append(eval_mdl(mdl))

# -------------------------------------------------- 分析测试表现 --------------------------------------------------------
print(y_test)

