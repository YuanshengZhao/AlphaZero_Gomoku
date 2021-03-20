# perform auto training loop

from types import DynamicClassAttribute
import psutil
import time
import datetime
import os
import random
import numpy as np
from six import int2byte

from tensorflow.python.keras.backend import dtype
import RNG as RN_Gomoku

# num of processes to launch
# need to clear games folder when modifying its value
n_proc=32
cpuFreeThreshold=1.0


start_time = time.time()

for ddir in ["./games","./eval"]:
    if(os.path.isdir(ddir)):
        print(ddir,"OK")
    else:
        os.system("mkdir "+ddir)
        print("created ",ddir)

def waitForComplete():
    freecount=0
    while(True):
        pst=psutil.cpu_percent()
        if(pst<cpuFreeThreshold):
            freecount+=1
        else:
            freecount=0
        print("current cpu:",pst,"free_count:",freecount,str(datetime.timedelta(seconds=time.time()-start_time)))
        if(freecount>=2):
            break
        time.sleep(5)
    print("finished!")

def getevalrst():
    st=0.0
    dbls=0
    for i in range(n_proc):
        fp=open("./eval/gm"+str(i+1)+"r.txt","r")
        pscr=float(fp.readline())
        st+=pscr
        if(pscr<0.1):
            dbls+=1
        fp.close()
    print("eval score = "+str(st))
    return st,dbls

def evaluate():
    print("evaluate...")
    os.system("python3 opening_generator.py")
    cmd="./Parallel_do.sh"
    rsed=random.sample(range(2147483629),n_proc)
    for i in range(n_proc):
        cmd+=" \"nice -n 19 ./ag.exe v gm"+str(i+1)+" op"+str(i+1)+" "+str(rsed[i])+"\""
    os.system(cmd)
    waitForComplete()

def saveMD(n1,n2):
    a0eng2=RN_Gomoku.A0_ENG(64,"./weights_run1/RNG64_%d.tf"%(n1),1e-1/(2.0**3))
    a0eng2.a0_eng.save("RNG_Old")
    a0eng=RN_Gomoku.A0_ENG(64,"./weights_run1/RNG64_%d.tf"%(n2),1e-1/(2.0**3))
    a0eng.a0_eng.save("RNG")

enging_list=np.array([47,46,45,44,43,41,31,21,11,1],dtype=int)
score_list=np.zeros(len(enging_list),dtype=float)

for i1 in range(len(enging_list)):
    for i2 in range(len(enging_list)):
        e1,e2=enging_list[i1],enging_list[i2]
        if(e2<=e1):
            continue
        saveMD(e1,e2)
        evaluate()
        est,dbls=getevalrst()
        score_list[i2]+=est
        score_list[i1]+=(2*n_proc-est)
        fp=open("./tounament_result.log","a+")
        fp.write(time.strftime("%m/%d/%Y %H:%M:%S",time.localtime())+" ")
        fp.write("%2d vs %2d -> %4.1f : %4.1f\n"%(e1,e2,2*n_proc-est,est))
        fp.close()

fp=open("./tounament_result.log","a+")
for scr,ege in sorted(zip(score_list,enging_list),reverse=True):
    fp.write("%2d -> %4.1f\n"%(ege,scr))
fp.close()
