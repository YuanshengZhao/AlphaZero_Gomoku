# perform auto training loop

import psutil
import time
import datetime
import os
import random

# num of processes to launch
# need to clear games folder when modifying its value
n_proc=32
cpuFreeThreshold=1.0

bigr,Nbgr=0,100
smlr,Nslr=0,1

start_time = time.time()

for ddir in ["./weights","./games","./eval"]:
    if(os.path.isdir(ddir)):
        print(ddir,"OK")
    else:
        os.system("mkdir "+ddir)
        print("created ",ddir)
os.system("nice -n 19 python3 SaveMD.py") # for safety

def waitForComplete():
    freecount=0
    while(True):
        pst=psutil.cpu_percent()
        if(pst<cpuFreeThreshold):
            freecount+=1
        else:
            freecount=0
        print("current cpu:",pst,"free_count:",freecount,"round:",bigr+1,"/",Nbgr,";",smlr+1,"/",Nslr,str(datetime.timedelta(seconds=time.time()-start_time)))
        if(freecount>=2):
            break
        time.sleep(60)
    print("finished!")
def selfPlay():
    print("self play...")
    cmd="./Parallel_do.sh"
    rsed=random.sample(range(2147483629),n_proc)
    for i in range(n_proc):
        # cmd+=" \"nice -n 19 ./ag.exe t 2000 gm"+str(i+1)+" "+str(rsed[i])+"\""
        cmd+=" \"nice -n 19 ./ag_batch.exe t 2000 gm"+str(i+1)+" "+str(rsed[i])+"\""
    print(cmd)
    os.system(cmd)
    waitForComplete()
    fp=open("./trLog64.log","a+")
    fp.write(time.strftime("%m/%d/%Y %H:%M:%S",time.localtime())+" finished self play\n")
    fp.close()

def training():
    os.system("nice -n 19 python3 tr2c.py 64")
    time.sleep(30)
    os.system("nice -n 19 python3 SaveMD.py n")

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

def updateNet():
    os.system("cp ./RNG64.tf.data-00000-of-00001 ./weights/RNG64.tf.data-00000-of-00001")
    os.system("cp ./RNG64.tf.index ./weights/RNG64.tf.index")
        
def checkpoints():
    ix=1
    while(os.path.isfile("./weights/RNG64_"+str(ix)+".tf.index")):
        ix+=1
    os.system("cp ./weights/RNG64.tf.data-00000-of-00001 ./weights/RNG64_"+str(ix)+".tf.data-00000-of-00001")
    os.system("cp ./weights/RNG64.tf.index ./weights/RNG64_"+str(ix)+".tf.index")    
    print("net saved to lv",ix)
    return ix

def discardNet():
    os.system("cp ./weights/RNG64.tf.data-00000-of-00001 ./RNG64.tf.data-00000-of-00001")
    os.system("cp ./weights/RNG64.tf.index ./RNG64.tf.index")

def evaluate():
    print("evaluate...")
    os.system("python3 opening_generator.py")
    cmd="./Parallel_do.sh"
    rsed=random.sample(range(2147483629),n_proc)
    for i in range(n_proc):
        cmd+=" \"nice -n 19 ./ag.exe v gm"+str(i+1)+" op"+str(i+1)+" "+str(rsed[i])+"\""
    os.system(cmd)
    waitForComplete()

start_date=time.localtime().tm_mday
discardNet() # safety

for bigr in range(Nbgr):
    noww=time.localtime()
    if(noww.tm_hour>=11 and noww.tm_mday!=start_date): break # stop after 11:00 next day
    print("training progress: ",bigr+1,"/",Nbgr)
    time.sleep(30)
    for smlr in range(Nslr):
        selfPlay()
        training()
        time.sleep(30)
    evaluate()
    est,dbls=getevalrst()
    fp=open("./trLog64.log","a+")
    fp.write(time.strftime("%m/%d/%Y %H:%M:%S",time.localtime())+" ")
    fp.write("test score: "+str(est)+" /"+str(n_proc*2)+" w/ "+str(dbls)+" dbls ")
    if(est-dbls*0.5 > n_proc):
        print("success")
        updateNet()
        vers=checkpoints()
        os.system("nice -n 19 python3 SaveMD.py")
        fp.write("success; net saved to lv "+str(vers)+"\n")
        fp.close()
        #on success train a 128-filter version and 20-block version
        # os.system("nice -n 19 python3 tr2c.py 128")
        # os.system("nice -n 19 python3 tr2c.py 20")
        #remove game files, since new files will append to old ones.
        os.system("rm ./games/gm*")
    else:
        print("fail")
        discardNet()
        fp.write("fail; net discarded\n")
        fp.close()