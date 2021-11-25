import keras
from keras.models import Model
from keras.layers import Input, Dense, Conv1D, Dropout
from keras.layers import GlobalMaxPooling1D
from keras.layers import concatenate

kr = 0.001

def nettcr_ab():
    '''NetTCR ab with the "correct" pooling dimension, that is: the three towers are convoluted, then pooled
       and then concatenated for the FNN'''
    # 使用glorot_normal来初始化模型参数

    pep_in = Input(shape=(9, 20))
    cdra_in = Input(shape=(30, 20))
    cdrb_in = Input(shape=(30, 20))
    # 对peptide、cdr3ab连续做1、3、5、7、9的卷积
    # conv_size = [1, 3, 5, 7, 9]
    # pep_cat = []
    # for i in conv_size:
    #     conv = Conv1D(16, i, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(pep_in)
    #     pool = GlobalMaxPooling1D()(conv)
    #     pep_cat = concatenate([pep_cat, pool])



    pep_conv1 = Conv1D(16, 1, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(pep_in)
    pep_pool1 = GlobalMaxPooling1D()(pep_conv1)
    pep_conv3 = Conv1D(16, 3, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(pep_in)
    pep_pool3 = GlobalMaxPooling1D()(pep_conv3)
    pep_conv5 = Conv1D(16, 5, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(pep_in)
    pep_pool5 = GlobalMaxPooling1D()(pep_conv5)
    pep_conv7 = Conv1D(16, 7, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(pep_in)
    pep_pool7 = GlobalMaxPooling1D()(pep_conv7)
    pep_conv9 = Conv1D(16, 9, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(pep_in)
    pep_pool9 = GlobalMaxPooling1D()(pep_conv9)

    cdra_conv1 = Conv1D(16, 1, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(cdra_in)
    cdra_pool1 = GlobalMaxPooling1D()(cdra_conv1)
    cdra_conv3 = Conv1D(16, 3, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(cdra_in)
    cdra_pool3 = GlobalMaxPooling1D()(cdra_conv3)
    cdra_conv5 = Conv1D(16, 5, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(cdra_in)
    cdra_pool5 = GlobalMaxPooling1D()(cdra_conv5)
    cdra_conv7 = Conv1D(16, 7, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(cdra_in)
    cdra_pool7 = GlobalMaxPooling1D()(cdra_conv7)
    cdra_conv9 = Conv1D(16, 9, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(cdra_in)
    cdra_pool9 = GlobalMaxPooling1D()(cdra_conv9)

    cdrb_conv1 = Conv1D(16, 1, padding='same', activation='sigmoid', kernel_initializer='glorot_normal', kernel_regularizer = kr)(cdrb_in)
    cdrb_pool1 = GlobalMaxPooling1D()(cdrb_conv1)
    cdrb_conv3 = Conv1D(16, 3, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(cdrb_in)
    cdrb_pool3 = GlobalMaxPooling1D()(cdrb_conv3)
    cdrb_conv5 = Conv1D(16, 5, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(cdrb_in)
    cdrb_pool5 = GlobalMaxPooling1D()(cdrb_conv5)
    cdrb_conv7 = Conv1D(16, 7, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(cdrb_in)
    cdrb_pool7 = GlobalMaxPooling1D()(cdrb_conv7)
    cdrb_conv9 = Conv1D(16, 9, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(cdrb_in)
    cdrb_pool9 = GlobalMaxPooling1D()(cdrb_conv9)

    pep_cat = concatenate([pep_pool1, pep_pool3, pep_pool5, pep_pool7, pep_pool9])
    cdra_cat = concatenate([cdra_pool1, cdra_pool3, cdra_pool5, cdra_pool7, cdra_pool9])
    cdrb_cat = concatenate([cdrb_pool1, cdrb_pool3, cdrb_pool5, cdrb_pool7, cdrb_pool9])

    cat = concatenate([pep_cat, cdra_cat, cdrb_cat], axis=1)

    dense = Dense(32, activation='sigmoid')(cat)

    out = Dense(1, activation='sigmoid')(dense)

    model = (Model(inputs=[cdra_in, cdrb_in, pep_in], outputs=[out]))

    return model
