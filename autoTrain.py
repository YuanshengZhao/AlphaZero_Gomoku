# perform auto training loop

import psutil
import time
import os

# num of processes to launch
# need to clear games folder when modifying its value
n_proc=32
cpuFreeThreshold=1.0

bigr,Nbgr=0,6*7

print("test random:")
os.system("./Parallel_do.sh"+ " \"python3 rndCheck.py\""*n_proc)
time.sleep(10)

for ddir in ["./weights","./games","./eval"]:
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
        print("current cpu:",pst,"free_count:",freecount,"round:",bigr+1,"/",Nbgr)
        if(freecount>=6):
            break
        time.sleep(5)
    print("finished!")
def selfPlay():
    print("self play...")
    cmd="./Parallel_do.sh"
    for i in range(n_proc):
        cmd+=" \"python3 sfpl2.py "+str(i+1)+"\""
    os.system(cmd)
    waitForComplete()
    fp=open("./trLog.log","a+")
    fp.write(time.strftime("%m/%d/%Y %H:%M:%S",time.localtime())+" finished self play\n")
    fp.close()

def training():
    os.system("python3 tr2.py")

def getevalrst():
    st=0.0
    dbls=0
    for i in range(n_proc):
        fp=open("./eval/gm"+str(i+1)+".txt","r")
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
    cmd="./Parallel_do.sh"
    for i in range(n_proc):
        cmd+=" \"python3 sfvs.py "+str(i+1)+"\""
    os.system(cmd)
    waitForComplete()

for bigr in range(Nbgr):
    print("training progress: ",bigr+1,"/",Nbgr)
    time.sleep(30)
    selfPlay()
    training()
    time.sleep(30)
    evaluate()
    est,dbls=getevalrst()
    fp=open("./trLog.log","a+")
    fp.write(time.strftime("%m/%d/%Y %H:%M:%S",time.localtime())+" ")
    fp.write("test score: "+str(est)+" /"+str(n_proc*2)+" w/ "+str(dbls)+" dbls ")
    if(est-dbls/2 > n_proc):
        print("success")
        updateNet()
        vers=checkpoints()
        fp.write("success; net saved to lv "+str(vers)+"\n")
        fp.close()
        # break
    else:
        print("fail")
        # discardNet()
        fp.write("fail\n")
        fp.close()