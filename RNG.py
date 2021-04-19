# new model: SE-ResNet

import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.regularizers import l2
se_chnl=32

brnp={#strong restriction should be used at the beginning of training
    'rmax': 1.25, 
    'rmin': 0.8,
    'dmax': 0.5
    # 'rmax': 3.125, 
    # 'rmin': 0.32,
    # 'dmax': 5.0
}

class RN_GM(tf.keras.Model):
    def __init__(self,nflter):
        super(RN_GM, self).__init__()
        #init conv
        self.conv01 = tf.keras.layers.Conv2D(nflter,3,padding='same',use_bias=False,kernel_regularizer=l2(1e-4))
        #seRN tower
        self.conv11 = tf.keras.layers.Conv2D(nflter,3,padding='same',use_bias=False,kernel_regularizer=l2(1e-4))
        self.conv12 = tf.keras.layers.Conv2D(nflter,3,padding='same',use_bias=False,kernel_regularizer=l2(1e-4))
        self.conv21 = tf.keras.layers.Conv2D(nflter,3,padding='same',use_bias=False,kernel_regularizer=l2(1e-4))
        self.conv22 = tf.keras.layers.Conv2D(nflter,3,padding='same',use_bias=False,kernel_regularizer=l2(1e-4))
        self.conv31 = tf.keras.layers.Conv2D(nflter,3,padding='same',use_bias=False,kernel_regularizer=l2(1e-4))
        self.conv32 = tf.keras.layers.Conv2D(nflter,3,padding='same',use_bias=False,kernel_regularizer=l2(1e-4))
        self.conv41 = tf.keras.layers.Conv2D(nflter,3,padding='same',use_bias=False,kernel_regularizer=l2(1e-4))
        self.conv42 = tf.keras.layers.Conv2D(nflter,3,padding='same',use_bias=False,kernel_regularizer=l2(1e-4))
        self.conv51 = tf.keras.layers.Conv2D(nflter,3,padding='same',use_bias=False,kernel_regularizer=l2(1e-4))
        self.conv52 = tf.keras.layers.Conv2D(nflter,3,padding='same',use_bias=False,kernel_regularizer=l2(1e-4))
        self.conv61 = tf.keras.layers.Conv2D(nflter,3,padding='same',use_bias=False,kernel_regularizer=l2(1e-4))
        self.conv62 = tf.keras.layers.Conv2D(nflter,3,padding='same',use_bias=False,kernel_regularizer=l2(1e-4))
        self.conv71 = tf.keras.layers.Conv2D(nflter,3,padding='same',use_bias=False,kernel_regularizer=l2(1e-4))
        self.conv72 = tf.keras.layers.Conv2D(nflter,3,padding='same',use_bias=False,kernel_regularizer=l2(1e-4))
        self.conv81 = tf.keras.layers.Conv2D(nflter,3,padding='same',use_bias=False,kernel_regularizer=l2(1e-4))
        self.conv82 = tf.keras.layers.Conv2D(nflter,3,padding='same',use_bias=False,kernel_regularizer=l2(1e-4))
        self.conv91 = tf.keras.layers.Conv2D(nflter,3,padding='same',use_bias=False,kernel_regularizer=l2(1e-4))
        self.conv92 = tf.keras.layers.Conv2D(nflter,3,padding='same',use_bias=False,kernel_regularizer=l2(1e-4))
        self.conva1 = tf.keras.layers.Conv2D(nflter,3,padding='same',use_bias=False,kernel_regularizer=l2(1e-4))
        self.conva2 = tf.keras.layers.Conv2D(nflter,3,padding='same',use_bias=False,kernel_regularizer=l2(1e-4))
        self.gap11 = tf.keras.layers.AveragePooling2D(15)
        self.fct11 = tf.keras.layers.Dense(se_chnl,activation='relu',kernel_regularizer=l2(1e-4),bias_regularizer=l2(2e-5))
        self.fct12 = tf.keras.layers.Dense(nflter,activation='sigmoid',kernel_regularizer=l2(1e-4),bias_regularizer=l2(2e-5))
        self.gap21 = tf.keras.layers.AveragePooling2D(15)
        self.fct21 = tf.keras.layers.Dense(se_chnl,activation='relu',kernel_regularizer=l2(1e-4),bias_regularizer=l2(2e-5))
        self.fct22 = tf.keras.layers.Dense(nflter,activation='sigmoid',kernel_regularizer=l2(1e-4),bias_regularizer=l2(2e-5))
        self.gap31 = tf.keras.layers.AveragePooling2D(15)
        self.fct31 = tf.keras.layers.Dense(se_chnl,activation='relu',kernel_regularizer=l2(1e-4),bias_regularizer=l2(2e-5))
        self.fct32 = tf.keras.layers.Dense(nflter,activation='sigmoid',kernel_regularizer=l2(1e-4),bias_regularizer=l2(2e-5))
        self.gap41 = tf.keras.layers.AveragePooling2D(15)
        self.fct41 = tf.keras.layers.Dense(se_chnl,activation='relu',kernel_regularizer=l2(1e-4),bias_regularizer=l2(2e-5))
        self.fct42 = tf.keras.layers.Dense(nflter,activation='sigmoid',kernel_regularizer=l2(1e-4),bias_regularizer=l2(2e-5))
        self.gap51 = tf.keras.layers.AveragePooling2D(15)
        self.fct51 = tf.keras.layers.Dense(se_chnl,activation='relu',kernel_regularizer=l2(1e-4),bias_regularizer=l2(2e-5))
        self.fct52 = tf.keras.layers.Dense(nflter,activation='sigmoid',kernel_regularizer=l2(1e-4),bias_regularizer=l2(2e-5))
        self.gap61 = tf.keras.layers.AveragePooling2D(15)
        self.fct61 = tf.keras.layers.Dense(se_chnl,activation='relu',kernel_regularizer=l2(1e-4),bias_regularizer=l2(2e-5))
        self.fct62 = tf.keras.layers.Dense(nflter,activation='sigmoid',kernel_regularizer=l2(1e-4),bias_regularizer=l2(2e-5))
        self.gap71 = tf.keras.layers.AveragePooling2D(15)
        self.fct71 = tf.keras.layers.Dense(se_chnl,activation='relu',kernel_regularizer=l2(1e-4),bias_regularizer=l2(2e-5))
        self.fct72 = tf.keras.layers.Dense(nflter,activation='sigmoid',kernel_regularizer=l2(1e-4),bias_regularizer=l2(2e-5))
        self.gap81 = tf.keras.layers.AveragePooling2D(15)
        self.fct81 = tf.keras.layers.Dense(se_chnl,activation='relu',kernel_regularizer=l2(1e-4),bias_regularizer=l2(2e-5))
        self.fct82 = tf.keras.layers.Dense(nflter,activation='sigmoid',kernel_regularizer=l2(1e-4),bias_regularizer=l2(2e-5))
        self.gap91 = tf.keras.layers.AveragePooling2D(15)
        self.fct91 = tf.keras.layers.Dense(se_chnl,activation='relu',kernel_regularizer=l2(1e-4),bias_regularizer=l2(2e-5))
        self.fct92 = tf.keras.layers.Dense(nflter,activation='sigmoid',kernel_regularizer=l2(1e-4),bias_regularizer=l2(2e-5))
        self.gapa1 = tf.keras.layers.AveragePooling2D(15)
        self.fcta1 = tf.keras.layers.Dense(se_chnl,activation='relu',kernel_regularizer=l2(1e-4),bias_regularizer=l2(2e-5))
        self.fcta2 = tf.keras.layers.Dense(nflter,activation='sigmoid',kernel_regularizer=l2(1e-4),bias_regularizer=l2(2e-5))
        #pv head
        self.convp1 = tf.keras.layers.Conv2D(nflter ,3,padding='same',kernel_regularizer=l2(1e-4),bias_regularizer=l2(2e-5))
        self.convp2 = tf.keras.layers.Conv2D(1 ,3,padding='same',kernel_regularizer=l2(1e-4),bias_regularizer=l2(2e-5))
        self.flat1 = tf.keras.layers.Flatten()
        self.flat2 = tf.keras.layers.Flatten()
        self.conca = tf.keras.layers.Concatenate()
        self.convv1 = tf.keras.layers.Conv2D(se_chnl ,3,padding='same',kernel_regularizer=l2(1e-4),bias_regularizer=l2(2e-5))
        self.dense1 = tf.keras.layers.Dense(128,use_bias=False,kernel_regularizer=l2(1e-4))
        self.dense2 = tf.keras.layers.Dense(1,activation='sigmoid',kernel_regularizer=l2(1e-4),bias_regularizer=l2(2e-5))
        #bn
        self.batnor01=tf.keras.layers.BatchNormalization(beta_regularizer=l2(2e-5),gamma_regularizer=l2(2e-5),renorm=True,renorm_clipping=brnp)
        self.batnor11=tf.keras.layers.BatchNormalization(beta_regularizer=l2(2e-5),gamma_regularizer=l2(2e-5),renorm=True,renorm_clipping=brnp)
        self.batnor12=tf.keras.layers.BatchNormalization(beta_regularizer=l2(2e-5),gamma_regularizer=l2(2e-5),renorm=True,renorm_clipping=brnp)
        self.batnor21=tf.keras.layers.BatchNormalization(beta_regularizer=l2(2e-5),gamma_regularizer=l2(2e-5),renorm=True,renorm_clipping=brnp)
        self.batnor22=tf.keras.layers.BatchNormalization(beta_regularizer=l2(2e-5),gamma_regularizer=l2(2e-5),renorm=True,renorm_clipping=brnp)
        self.batnor31=tf.keras.layers.BatchNormalization(beta_regularizer=l2(2e-5),gamma_regularizer=l2(2e-5),renorm=True,renorm_clipping=brnp)
        self.batnor32=tf.keras.layers.BatchNormalization(beta_regularizer=l2(2e-5),gamma_regularizer=l2(2e-5),renorm=True,renorm_clipping=brnp)
        self.batnor41=tf.keras.layers.BatchNormalization(beta_regularizer=l2(2e-5),gamma_regularizer=l2(2e-5),renorm=True,renorm_clipping=brnp)
        self.batnor42=tf.keras.layers.BatchNormalization(beta_regularizer=l2(2e-5),gamma_regularizer=l2(2e-5),renorm=True,renorm_clipping=brnp)
        self.batnor51=tf.keras.layers.BatchNormalization(beta_regularizer=l2(2e-5),gamma_regularizer=l2(2e-5),renorm=True,renorm_clipping=brnp)
        self.batnor52=tf.keras.layers.BatchNormalization(beta_regularizer=l2(2e-5),gamma_regularizer=l2(2e-5),renorm=True,renorm_clipping=brnp)
        self.batnor61=tf.keras.layers.BatchNormalization(beta_regularizer=l2(2e-5),gamma_regularizer=l2(2e-5),renorm=True,renorm_clipping=brnp)
        self.batnor62=tf.keras.layers.BatchNormalization(beta_regularizer=l2(2e-5),gamma_regularizer=l2(2e-5),renorm=True,renorm_clipping=brnp)
        self.batnor71=tf.keras.layers.BatchNormalization(beta_regularizer=l2(2e-5),gamma_regularizer=l2(2e-5),renorm=True,renorm_clipping=brnp)
        self.batnor72=tf.keras.layers.BatchNormalization(beta_regularizer=l2(2e-5),gamma_regularizer=l2(2e-5),renorm=True,renorm_clipping=brnp)
        self.batnor81=tf.keras.layers.BatchNormalization(beta_regularizer=l2(2e-5),gamma_regularizer=l2(2e-5),renorm=True,renorm_clipping=brnp)
        self.batnor82=tf.keras.layers.BatchNormalization(beta_regularizer=l2(2e-5),gamma_regularizer=l2(2e-5),renorm=True,renorm_clipping=brnp)
        self.batnor91=tf.keras.layers.BatchNormalization(beta_regularizer=l2(2e-5),gamma_regularizer=l2(2e-5),renorm=True,renorm_clipping=brnp)
        self.batnor92=tf.keras.layers.BatchNormalization(beta_regularizer=l2(2e-5),gamma_regularizer=l2(2e-5),renorm=True,renorm_clipping=brnp)
        self.batnora1=tf.keras.layers.BatchNormalization(beta_regularizer=l2(2e-5),gamma_regularizer=l2(2e-5),renorm=True,renorm_clipping=brnp)
        self.batnora2=tf.keras.layers.BatchNormalization(beta_regularizer=l2(2e-5),gamma_regularizer=l2(2e-5),renorm=True,renorm_clipping=brnp)
        self.batnorv1=tf.keras.layers.BatchNormalization(beta_regularizer=l2(2e-5),gamma_regularizer=l2(2e-5),renorm=True,renorm_clipping=brnp)

    def call(self, inputs, training=False):
        x=tf.nn.relu(self.batnor01(self.conv01(inputs),training=training))# init conv
        # se_RN tower
        y=self.conv12(tf.nn.relu(self.batnor11(self.conv11(x),training=training)))#block 1
        x=tf.nn.relu(x+self.batnor12(y*self.fct12(self.fct11(self.gap11(y))),training=training))
        y=self.conv22(tf.nn.relu(self.batnor21(self.conv21(x),training=training)))#block 2
        x=tf.nn.relu(x+self.batnor22(y*self.fct22(self.fct21(self.gap21(y))),training=training))
        y=self.conv32(tf.nn.relu(self.batnor31(self.conv31(x),training=training)))#block 3
        x=tf.nn.relu(x+self.batnor32(y*self.fct32(self.fct31(self.gap31(y))),training=training))
        y=self.conv42(tf.nn.relu(self.batnor41(self.conv41(x),training=training)))#block 4
        x=tf.nn.relu(x+self.batnor42(y*self.fct42(self.fct41(self.gap41(y))),training=training))
        y=self.conv52(tf.nn.relu(self.batnor51(self.conv51(x),training=training)))#block 5
        x=tf.nn.relu(x+self.batnor52(y*self.fct52(self.fct51(self.gap51(y))),training=training))
        y=self.conv62(tf.nn.relu(self.batnor61(self.conv61(x),training=training)))#block 6
        x=tf.nn.relu(x+self.batnor62(y*self.fct62(self.fct61(self.gap61(y))),training=training))
        y=self.conv72(tf.nn.relu(self.batnor71(self.conv71(x),training=training)))#block 7
        x=tf.nn.relu(x+self.batnor72(y*self.fct72(self.fct71(self.gap71(y))),training=training))
        y=self.conv82(tf.nn.relu(self.batnor81(self.conv81(x),training=training)))#block 8
        x=tf.nn.relu(x+self.batnor82(y*self.fct82(self.fct81(self.gap81(y))),training=training))
        y=self.conv92(tf.nn.relu(self.batnor91(self.conv91(x),training=training)))#block 9
        x=tf.nn.relu(x+self.batnor92(y*self.fct92(self.fct91(self.gap91(y))),training=training))
        y=self.conva2(tf.nn.relu(self.batnora1(self.conva1(x),training=training)))#block a
        x=tf.nn.relu(x+self.batnora2(y*self.fcta2(self.fcta1(self.gapa1(y))),training=training))
        #policy head
        pr=tf.math.softmax(self.flat2(self.convp2(self.convp1(x))))
        #value head
        vr=self.dense2(tf.nn.relu(self.batnorv1(self.dense1(self.flat1(self.convv1(x))),training=training)))
        return self.conca([pr,vr])


class RN_GM_v2(tf.keras.Model):
    def __init__(self,nflter,nblock):
        super(RN_GM_v2, self).__init__()
        self.nblk=nblock
        #init conv
        self.conv01 = tf.keras.layers.Conv2D(nflter,3,padding='same',use_bias=False,kernel_regularizer=l2(1e-4))
        #seRN tower
        self.convbs = [[tf.keras.layers.Conv2D(nflter,3,padding='same',use_bias=False,kernel_regularizer=l2(1e-4)),
                        tf.keras.layers.Conv2D(nflter,3,padding='same',use_bias=False,kernel_regularizer=l2(1e-4))]
                        for _ in range(nblock)]
        self.selys = [[tf.keras.layers.AveragePooling2D(15),
                       tf.keras.layers.Dense(se_chnl,activation='relu',kernel_regularizer=l2(1e-4),bias_regularizer=l2(2e-5)),
                       tf.keras.layers.Dense(nflter,activation='sigmoid',kernel_regularizer=l2(1e-4),bias_regularizer=l2(2e-5))]
                       for _ in range(nblock)]
        #pv head
        self.convp1 = tf.keras.layers.Conv2D(nflter ,3,padding='same',kernel_regularizer=l2(1e-4),bias_regularizer=l2(2e-5))
        self.convp2 = tf.keras.layers.Conv2D(1 ,3,padding='same',kernel_regularizer=l2(1e-4),bias_regularizer=l2(2e-5))
        self.flat1 = tf.keras.layers.Flatten()
        self.flat2 = tf.keras.layers.Flatten()
        self.conca = tf.keras.layers.Concatenate()
        self.convv1 = tf.keras.layers.Conv2D(se_chnl ,3,padding='same',kernel_regularizer=l2(1e-4),bias_regularizer=l2(2e-5))
        self.dense1 = tf.keras.layers.Dense(128,use_bias=False,kernel_regularizer=l2(1e-4))
        self.dense2 = tf.keras.layers.Dense(1,activation='sigmoid',kernel_regularizer=l2(1e-4),bias_regularizer=l2(2e-5))
        #bn
        self.batnor01=tf.keras.layers.BatchNormalization(beta_regularizer=l2(2e-5),gamma_regularizer=l2(2e-5),renorm=True,renorm_clipping=brnp)
        self.batnorbs = [(tf.keras.layers.BatchNormalization(beta_regularizer=l2(2e-5),gamma_regularizer=l2(2e-5),renorm=True,renorm_clipping=brnp),
                          tf.keras.layers.BatchNormalization(beta_regularizer=l2(2e-5),gamma_regularizer=l2(2e-5),renorm=True,renorm_clipping=brnp))
                        for _ in range(nblock)]
        self.batnorv1=tf.keras.layers.BatchNormalization(beta_regularizer=l2(2e-5),gamma_regularizer=l2(2e-5),renorm=True,renorm_clipping=brnp)

    def call(self, inputs, training=False):
        x=tf.nn.relu(self.batnor01(self.conv01(inputs),training=training))# init conv
        # se_RN tower
        for kn in range(self.nblk):
            y=self.convbs[kn][1](tf.nn.relu(self.batnorbs[kn][0](self.convbs[kn][0](x),training=training)))
            x=tf.nn.relu(x+self.batnorbs[kn][1](y*self.selys[kn][2](self.selys[kn][1](self.selys[kn][0](y))),training=training))
        #policy head
        pr=tf.math.softmax(self.flat2(self.convp2(self.convp1(x))))
        #value head
        vr=self.dense2(tf.nn.relu(self.batnorv1(self.dense1(self.flat1(self.convv1(x))),training=training)))
        return self.conca([pr,vr])
# tf.keras.backend.clear_session()

# class mybce(tf.keras.losses.Loss):
#     def __init__(self):
#         super(mybce, self).__init__()
#     def call(self, y_true, y_pred):
#         return tf.reduce_mean(-y_true*tf.math.log(y_pred)-(1-y_true)*tf.math.log(1-y_pred))

class MixedLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(MixedLoss, self).__init__()
        self.cce=tf.keras.losses.CategoricalCrossentropy()
        self.mse=tf.keras.losses.MeanSquaredError()
    def call(self, y_true, y_pred):
        return self.cce(y_true[:,:-1],y_pred[:,:-1])+4*self.mse(y_true[:,-1],y_pred[:,-1])


class AZP(tf.keras.metrics.Metric):
    def __init__(self):
        super(AZP, self).__init__()
        self.acup=tf.keras.metrics.TopKCategoricalAccuracy(k=1)
    def update_state(self, y_true, y_pred, sample_weight=None):
        yt=y_true[:,:-1]
        yp=y_pred[:,:-1]
        self.acup.update_state(yt,yp)
    def result(self):
        return self.acup.result()
    def reset_states(self):
        return self.acup.reset_states()

class AZV(tf.keras.metrics.Metric):
    def __init__(self):
        super(AZV, self).__init__()
        self.acuv=tf.keras.metrics.Accuracy()
    def update_state(self, y_true, y_pred, sample_weight=None):
        yt=tf.zeros_like(y_true[:,-1])
        yp=tf.nn.relu(tf.math.abs(y_pred[:,-1]-y_true[:,-1])-.5)
        self.acuv.update_state(yt,yp)
    def result(self):
        return self.acuv.result()
    def reset_states(self):
        return self.acuv.reset_states()

class A0_ENG(object):
    def __init__(self,nflter,wt_file,lgrt=1e-2,nblock=10):
        # self.optz=tf.keras.optimizers.SGD(learning_rate=lgrt,momentum=0.9,nesterov=True,clipvalue=1e-1)
        self.optz=tf.keras.optimizers.Adam() # step size bound with default params: 3.162*lr
        if(nblock==10):# use v2 ispossible, no v2 for compatibility when loading weights
            self.a0_eng = RN_GM(nflter)
        else:
            self.a0_eng = RN_GM_v2(nflter,nblock)

        self.gmloss=MixedLoss()
        self.gmpmetric=AZP()
        self.gmvmetric=AZV()

        self.a0_eng(np.array(np.zeros([16,15,15,2]), dtype='<f4'))
        self.a0_eng.compile(optimizer=self.optz,
                      loss=self.gmloss,
                      metrics=[self.gmpmetric,self.gmvmetric])
        self.a0_eng.summary()
        if(os.path.isfile(wt_file+".index")):
            self.a0_eng.load_weights(wt_file)
            print("loaded weights",wt_file)
        else:
            print("warning: weight file not found!")
        self.a0_eng.optimizer.lr.assign(lgrt)