# training
import numpy as np
import os
import time
import struct
import sys
import RNG

try:
    nchl=int(sys.argv[1])
except:
    print("Error: Failed to get n_channel. Exiting.")
    quit()
if(nchl==64 or nchl==128):
    a0eng=RNG.A0_ENG(nchl,"./RNG%d.tf"%(nchl),1e-3)
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

# print("evaluation 2:")
# print(a0eng.a0_eng.evaluate(x_tr,y_tr))

btze=2048
prtt=int(lxtr/btze*.95)*btze
print("validation split:",prtt,"|",lxtr-prtt)
vloss=1e100
nepc=0
while(True):
    hist1=a0eng.a0_eng.fit(x_tr[:prtt],y_tr[:prtt],epochs=1,shuffle=True,batch_size=btze,validation_data=(x_tr[prtt:],y_tr[prtt:]))
    nvls=hist1.history['val_loss'][-1]
    if(nvls>=vloss):
        break
    else:
        nepc+=1
        hist=hist1
        vloss=nvls
        a0eng.a0_eng.save_weights("./RNG%d.tf"%(nchl))
# a0eng.a0_eng.save("RNG")
nstep=a0eng.a0_eng.optimizer.get_weights()[-1]
if(not isinstance(nstep,(int,np.integer))): #first round only
    nstep=prtt//btze*nepc
print("steps trained:",nstep,"newly trained epochs:",nepc)


fp=open("./trLog%d.log"%(nchl),"a+")
fp.write(time.strftime("%m/%d/%Y %H:%M:%S",time.localtime())+" training > ")
fp.write("t: %6d s: %5d ux: %5d uy: %5d; ls: %.3f p: %.3f v: %.3f vls: %.3f vp: %.3f vv: %.3f\n"%(nstep,lxtr,luxtr,luytr,
                                     hist.history['loss'][-1],hist.history['azp'][-1],hist.history['azv'][-1],
                                     hist.history['val_loss'][-1],hist.history['val_azp'][-1],hist.history['val_azv'][-1]))
fp.close()
