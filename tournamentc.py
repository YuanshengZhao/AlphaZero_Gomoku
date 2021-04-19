# perform auto training loop

import psutil
import time
import datetime
import os
import random
import numpy as np

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
        if(freecount>=5):
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
    cmd="./Parallel_do.sh"
    for i in range(n_proc):
        cmd+=" \"nice -n 19 ./ag.exe v gm"+str(i+1)+" op"+str(i+1)+" "+str(rsed[i])+"\""
    os.system(cmd)
    waitForComplete()

def saveMD(ee1,ee2):
    p1,f1,b1=ee1
    p2,f2,b2=ee2
    a0eng=RN_Gomoku.A0_ENG(f1,p1,1e-3,b1)
    a0eng.a0_eng.save("RNG_Old")
    a0eng2=RN_Gomoku.A0_ENG(f2,p2,1e-3,b2)
    a0eng2.a0_eng.save("RNG")

def make_pair():
    neg=len(score_list)
    scre,ege_srt=zip(*sorted(zip(score_list, np.arange(neg)),reverse=True))
    ege_srt=list(ege_srt)
    egesetc=ege_srt.copy()
    pr=[]
    passed=False
    ege_srt_hist=[]
    while(not passed):
        ii=0
        while(ii<neg):
            jj=ii+1
            while(jj<neg):
                if(ege_srt[jj] not in pair_list[ege_srt[ii]]):
                    tp=ege_srt[ii+1]
                    ege_srt[ii+1]=ege_srt[jj]
                    ege_srt[jj]=tp
                    break
                jj+=1
            ii+=2
        passed=True
        # print(ege_srt)
        ii=1
        while(ii<neg):
            if(ege_srt[ii-1] in pair_list[ege_srt[ii]]): 
                # print(ege_srt[ii-1],ege_srt[ii])
                passed=False
                break
            ii+=2
        if(ege_srt in ege_srt_hist): return []
        else: 
            ege_srt_hist.append(ege_srt)
            # print(ege_srt_hist)
        ege_srt=ege_srt[::-1]
    ii=0
    while(ii<neg):
        pr.append([ege_srt[ii],ege_srt[ii+1]])
        pair_list[ege_srt[ii]].append(ege_srt[ii+1])
        pair_list[ege_srt[ii+1]].append(ege_srt[ii])
        ii+=2
    fp=open("./tounament_result.log","a+")
    for i in range(neg):
        fp.write("%2d %*s %.1f %s\n"%(egesetc[i],max_string,enging_list[egesetc[i]][0],score_list[egesetc[i]], str(pair_list[egesetc[i]])))
    fp.close()
    print(ege_srt,egesetc)
    return pr



enging_list=[
("./weights/RNG64_%d.tf"%(iii),64,10) for iii in [97,98,102,104,106,107,77,91]
]
random.shuffle(enging_list)
print(enging_list)
max_string=max(*[len(enging_list[k][0]) for k in range(len(enging_list))])
# form="RoundRobin"
form="Swiss"

# waitForComplete()

if(form =="RoundRobin"):
    score_list=np.zeros(len(enging_list),dtype=float)
    for i1 in range(len(enging_list)):
        for i2 in range(len(enging_list)):
            os.system("python3 opening_generator.py")
            rsed=random.sample(range(2147483629),n_proc)
            e1,e2=enging_list[i1],enging_list[i2]
            if(i1>=i2):
                continue
            saveMD(e1,e2)
            evaluate()
            est,dbls=getevalrst()
            score_list[i2]+=est
            score_list[i1]+=(2*n_proc-est)
            fp=open("./tounament_result.log","a+")
            fp.write(time.strftime("%m/%d/%Y %H:%M:%S",time.localtime())+" ")
            fp.write("%-*s vs %-*s -> %4.1f : %4.1f\n"%(max_string,e1[0],max_string,e2[0],2*n_proc-est,est))
            fp.close()
    fp=open("./tounament_result.log","a+")
    for scr,ege in sorted(zip(score_list,enging_list),reverse=True):
        fp.write("%-*s -> %4.1f\n"%(max_string,ege[0],scr))
    fp.close()
elif(form=="Swiss"):
    score_list=np.zeros(len(enging_list),dtype=float)
    pair_list=[[] for _ in range(len(enging_list))]
    rest_list=[[] for _ in range(len(enging_list))]
    nround=min(len(enging_list)-1,int(np.log2(len(enging_list)))+4)
    fp=open("./tounament_result.log","a+")
    fp.write(time.strftime("%m/%d/%Y %H:%M:%S",time.localtime())+" ")
    fp.write("swiss round %d\n"%(nround))
    fp.close()
    for nrd in range(nround):
        fp=open("./tounament_result.log","a+")
        fp.write("round %d\n"%(nrd))
        fp.close()
        os.system("python3 opening_generator.py")
        rsed=random.sample(range(2147483629),n_proc)
        prs=make_pair()
        for ppr in prs:
            i1,i2=ppr[0],ppr[1]
            e1,e2=enging_list[i1],enging_list[i2]
            saveMD(e1,e2)
            evaluate()
            est,dbls=getevalrst()
            # est=(33 if i1<i2 else 31)+np.random.choice([-1.5,-1,-.5,0,.5,1,1.5])
            rest_list[i2].append(est)
            rest_list[i1].append((2*n_proc-est))
            if(est==n_proc):
                score_list[i2]+=.5
                score_list[i1]+=.5
            elif(est>n_proc):
                score_list[i2]+=1
            else:
                score_list[i1]+=1
            fp=open("./tounament_result.log","a+")
            fp.write(time.strftime("%m/%d/%Y %H:%M:%S",time.localtime())+" ")
            fp.write("(%-2d) %-*s vs (%-2d) %-*s -> %4.1f : %4.1f | %4.1f : %4.1f\n"%(i1,max_string,e1[0],i2,max_string,e2[0],2*n_proc-est,est,score_list[i1],score_list[i2]))
            fp.close()
        

    fp=open("./tounament_result.log","a+")
    for scr,ege in sorted(zip(score_list,np.arange(len(enging_list))),reverse=True):
        fp.write("%-*s -> %4.1f | %4.1f\n"%(max_string,enging_list[ege][0],scr,sum(rest_list[ege])))
    fp.close()

else:
    print("Bad format!")
