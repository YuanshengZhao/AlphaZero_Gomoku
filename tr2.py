# training
import numpy as np
from tensorflow.python.ops.gen_math_ops import rsqrt, rsqrt_grad
import RNG as RN_Gomoku
import os
import time

a0eng=RN_Gomoku.A0_ENG(64,"./RNG64.tf")

if(os.path.isfile("./games/dat_trprev.npz")):
    npz_t=np.load("./games/dat_trprev.npz")
    x_tr=npz_t['arr_0']
    y_tr=npz_t['arr_1']
    print("load cached data: %d"%(len(y_tr)))
    npz_t=np.load("./games/dat_train1.npz")
    x_tr=np.concatenate((x_tr,npz_t['arr_0']))
    y_tr=np.concatenate((y_tr,npz_t['arr_1']))
else:
    npz_t=np.load("./games/dat_train1.npz")
    x_tr=npz_t['arr_0']
    y_tr=npz_t['arr_1']

npz_v=np.load("./games/dat_vlidn1.npz")
x_vl=npz_v['arr_0']
y_vl=npz_v['arr_1']

ii=2
while(True):
    nf="./games/dat_train"+str(ii)+".npz"
    if(os.path.exists(nf)):
        npz_t=np.load(nf)
        x_tr=np.concatenate((x_tr,npz_t['arr_0']))
        y_tr=np.concatenate((y_tr,npz_t['arr_1']))
    else:
        break
    nf="./games/dat_vlidn"+str(ii)+".npz"
    if(os.path.exists(nf)):
        npz_v=np.load(nf)
        x_vl=np.concatenate((x_vl,npz_v['arr_0']))
        y_vl=np.concatenate((y_vl,npz_v['arr_1']))
    ii+=1
print("loaded ",ii-1,"data files.")

# for i in range(len(y_tr)):
#     y_tr[i,:-1]=np.sqrt(y_tr[i,:-1])/np.sum(np.sqrt(y_tr[i,:-1]))
# for i in range(len(y_vl)):
#     y_vl[i,:-1]=np.sqrt(y_vl[i,:-1])/np.sum(np.sqrt(y_vl[i,:-1]))

lxtr,luxtr,luytr=len(x_tr),len(np.unique(x_tr,axis=0)),len(np.unique(y_tr,axis=0))
print(lxtr," training samples",luxtr,luytr,"unique")
print(len(x_vl)," validatng samples")
print("avg score: ",np.average(y_tr[:,-1]),np.average(y_vl[:,-1]))
print("draws: ",np.sum(y_tr[:,-1]==.5),np.sum(y_vl[:,-1]==.5))

btze=256
# rstr=a0eng.a0_eng(x_tr[-btze:],training=True).numpy()
# rsvl=a0eng.a0_eng(x_tr[-btze:]).numpy()
# print("v_rst:",a0eng.gmloss(y_tr[-btze:],rsvl).numpy(),rstr[-5:,-1],rsvl[-5:,-1],y_tr[-5:,-1])

hist=a0eng.a0_eng.fit(x_tr,y_tr,epochs=1,shuffle=False,batch_size=btze,validation_data=(x_vl,y_vl),steps_per_epoch=lxtr//btze)
a0eng.a0_eng.save_weights("./RNG64.tf")
nstep=a0eng.a0_eng.optimizer.get_weights()[-1]
print("steps trained:",nstep)

rstr=a0eng.a0_eng(x_tr[-btze:],training=True).numpy()
rsvl=a0eng.a0_eng(x_tr[-btze:]).numpy()
print("v_rst:",a0eng.gmloss(y_tr[-btze:],rsvl).numpy(),rstr[-5:,-1],rsvl[-5:,-1],y_tr[-5:,-1])

#do not waste any training data!
rmn=lxtr%btze
if(rmn>0):
    npz_t=np.savez("./games/dat_trprev.npz",np.concatenate((x_tr[-rmn:],x_vl)),np.concatenate((y_tr[-rmn:],y_vl)))
else:
    npz_t=np.savez("./games/dat_trprev.npz",x_vl,y_vl)
print("cached last %d samples and vlidn data"%(rmn))

fp=open("./trLog.log","a+")
fp.write(time.strftime("%m/%d/%Y %H:%M:%S",time.localtime())+" training > ")
fp.write("t: %6d s: %5d ux: %5d uy: %5d; ls: %.3f p: %.3f v: %.3f vls: %.3f vp: %.3f vv: %.3f\n"%(nstep,lxtr,luxtr,luytr,
                                     hist.history['loss'][0],hist.history['azp'][0],hist.history['azv'][0],
                                     hist.history['val_loss'][0],hist.history['val_azp'][0],hist.history['val_azv'][0]))
fp.close()
