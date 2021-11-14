import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from tensorflow.keras.optimizers import Adam
import utils
from keras.callbacks import EarlyStopping

from nettcr_architectures import nettcr_ab

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
parser.add_argument("-tr", "--trainfile", help="设定训练集")
parser.add_argument("-te", "--testfile", help="设定测试集")
parser.add_argument("-o", "--outfile", default=sys.stdout, help="设定输出文件夹")

# 设置超参
parser.add_argument("-e", "--epochs", default=100, type=int, help="设定训练时期数")
parser.add_argument("-lr", "--learn_rate", default=0.001, type=float, help="设定优化算法学习率")
parser.add_argument("-bs", "--batch_size", default=128, type=int, help="批大小")

args = parser.parse_args()
LR = float(args.learn_rate)
EPOCHS = int(args.epochs)
BATCH_SIZE = int(args.batch_size)

# ----------------------------载入数据----------------------------------------
print('Loading and encoding the data..')
train_data = pd.read_csv(args.trainfile)
test_data = pd.read_csv(args.testfile)

# Encode data
encoding = utils.blosum50_20aa
early_stop = EarlyStopping(monitor='loss', min_delta=0,
                           patience=10, verbose=0, mode='min', restore_best_weights=True)

# Call and compile the model
pep_train = utils.enc_list_bl_max_len(train_data.peptide, encoding, 9)
tcra_train = utils.enc_list_bl_max_len(train_data.CDR3a, encoding, 30)
tcrb_train = utils.enc_list_bl_max_len(train_data.CDR3b, encoding, 30)
y_train = np.array(train_data.binder)

pep_test = utils.enc_list_bl_max_len(test_data.peptide, encoding, 9)
tcra_test = utils.enc_list_bl_max_len(test_data.CDR3a, encoding, 30)
tcrb_test = utils.enc_list_bl_max_len(test_data.CDR3b, encoding, 30)
train_inputs = [tcra_train, tcrb_train, pep_train]
test_inputs = [tcra_test, tcrb_test, pep_test]

mdl = nettcr_ab()

mdl.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.001))
print('Training..')
# Train
history = mdl.fit(train_inputs, y_train,
                  epochs=EPOCHS, batch_size=128, verbose=1, callbacks=[early_stop])

print('Evaluating..')
# Predict on test data
preds = mdl.predict(test_inputs, verbose=0)
pred_df = pd.concat([test_data, pd.Series(np.ravel(preds), name='prediction')], axis=1)

pred_df.to_csv(args.outfile, index=False)
