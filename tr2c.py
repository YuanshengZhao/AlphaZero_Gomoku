# training
import numpy as np
import os
import time
import struct
import sys
import RNG
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(32)
tf.config.threading.set_intra_op_parallelism_threads(32)

try:
    nchl=int(sys.argv[1])
except:
    print("Error: Failed to get n_channel. Exiting.")
    quit()
if(nchl==64 or nchl==128):
    a0eng=RNG.A0_ENG(nchl,"./RNG%d.tf"%(nchl),1e-4 if nchl==64 else 1e-3)
    print("10 blk * %d flt."%(nchl))
elif(nchl==20):# 20 is actually blocks
    a0eng=RNG.A0_ENG(64,"./RNG%d.tf"%(nchl),1e-3,20)
    print("20 blk * 64 flt.")
else:
    print("Error: bad n cmd. Exiting.")
    quit()

time.sleep(30)


fn="games/gm1.x"
sz=os.stat(fn).st_size//4
fp=open(fn,"rb")
x_tr0=struct.unpack('f'*sz, fp.read(4*sz))
x_tr0=np.reshape(x_tr0,[sz//450,15,15,2])
fp.close()

fn="games/gm1.y"
sz=os.stat(fn).st_size//4
fp=open(fn,"rb")
y_tr0=struct.unpack('f'*sz, fp.read(4*sz))
y_tr0=np.reshape(y_tr0,[sz//226,226])
fp.close()

x_tr=x_tr0.copy()
y_tr=y_tr0.copy()

ii=2
while(True):
    fn="games/gm"+str(ii)+".x"
    if(os.path.exists(fn)):
        sz=os.stat(fn).st_size//4
        fp=open(fn,"rb")
        x_tr0=struct.unpack('f'*sz, fp.read(4*sz))
        x_tr0=np.reshape(x_tr0,[sz//450,15,15,2])
        fp.close()
        x_tr=np.concatenate((x_tr0,x_tr))
    else:
        break
    fn="games/gm"+str(ii)+".y"
    if(os.path.exists(fn)):
        sz=os.stat(fn).st_size//4
        fp=open(fn,"rb")
        y_tr0=struct.unpack('f'*sz, fp.read(4*sz))
        y_tr0=np.reshape(y_tr0,[sz//226,226])
        fp.close()
        y_tr=np.concatenate((y_tr0,y_tr))
    ii+=1
print("loaded ",ii-1,"data files.")


lxtr,luxtr,luytr=len(x_tr),len(np.unique(x_tr,axis=0)),len(np.unique(y_tr,axis=0))
print(lxtr,len(y_tr)," training samples",luxtr,luytr,"unique")
print("avg score: ",np.average(y_tr[:,-1]))
print("draws: ",np.sum(y_tr[:,-1]==.5))

# print("evaluation 1:")
# print(a0eng.a0_eng.evaluate(x_tr,y_tr))

indeces = np.random.permutation(lxtr)
x_tr=x_tr[indeces]
y_tr=y_tr[indeces]
print("data shuffled")


def flip_x(ps):
    return (14-ps//15)*15+ps%15
def flip_y(ps):
    return (ps//15)*15+(14-ps%15)
def trs_xy(ps):
    return (ps%15)*15+ps//15
idxes=[# here it should be the inverse operation!!!
    np.array([(ps                         if ps<225 else ps) for ps in range(226)]),
    np.array([(flip_x(ps)                 if ps<225 else ps) for ps in range(226)]),
    np.array([(flip_y(ps)                 if ps<225 else ps) for ps in range(226)]),
    np.array([(flip_x(flip_y(ps))         if ps<225 else ps) for ps in range(226)]),
    np.array([(trs_xy(ps)                 if ps<225 else ps) for ps in range(226)]),
    np.array([(trs_xy(flip_y(ps))         if ps<225 else ps) for ps in range(226)]),# !!!
    np.array([(trs_xy(flip_x(ps))         if ps<225 else ps) for ps in range(226)]),# !!!
    np.array([(trs_xy(flip_x(flip_y(ps))) if ps<225 else ps) for ps in range(226)])
]
mxfns=[
    lambda mxx: mxx,
    lambda mxx: np.flip(mxx,0),
    lambda mxx: np.flip(mxx,1),
    lambda mxx: np.flip(mxx,(0,1)),
    lambda mxx: np.transpose(mxx,(1,0,2)),
    lambda mxx: np.transpose(np.flip(mxx,0),(1,0,2)),
    lambda mxx: np.transpose(np.flip(mxx,1),(1,0,2)),
    lambda mxx: np.transpose(np.flip(mxx,(0,1)),(1,0,2))
]
def data_augmentor(datx,daty,rnd):
    return mxfns[rnd](datx),daty[idxes[rnd]]


# print("evaluation 2:")
# print(a0eng.a0_eng.evaluate(x_tr,y_tr))

btze=2048
prtt=int(lxtr/btze*.95)*btze
print("validation split:",prtt,"|",lxtr-prtt)
print("learning rate:",a0eng.a0_eng.optimizer.lr.numpy())
vloss=1e100
nepc=0
while(nepc<5):#max epochs here
    print("augmenting data...  ",end="")
    for kk in range(lxtr):
        x_tr[kk],y_tr[kk]=data_augmentor(x_tr[kk],y_tr[kk],np.random.randint(8))
    print("Done!")
    hist1=a0eng.a0_eng.fit(x_tr[:prtt],y_tr[:prtt],epochs=1,shuffle=True,batch_size=btze,validation_data=(x_tr[prtt:],y_tr[prtt:]))
    nvls=hist1.history['val_loss'][-1]
    if(nvls>=vloss):
        break
    else:
        nepc+=1
        hist=hist1
        vloss=nvls
        a0eng.a0_eng.save_weights("./RNG%d.tf"%(nchl))
        nstep=a0eng.a0_eng.optimizer.get_weights()[-1]
# a0eng.a0_eng.save("RNG")
if(not isinstance(nstep,(int,np.integer))): #first round only
    nstep=prtt//btze*nepc
print("steps trained:",nstep,"newly trained epochs:",nepc)


fp=open("./trLog%d.log"%(nchl),"a+")
fp.write(time.strftime("%m/%d/%Y %H:%M:%S",time.localtime())+" training > ")
fp.write("t: %6d s: %5d ux: %5d uy: %5d; ls: %.3f p: %.3f v: %.3f vls: %.3f vp: %.3f vv: %.3f\n"%(nstep,lxtr,luxtr,luytr,
                                     hist.history['loss'][-1],hist.history['azp'][-1],hist.history['azv'][-1],
                                     hist.history['val_loss'][-1],hist.history['val_azp'][-1],hist.history['val_azv'][-1]))
fp.close()
