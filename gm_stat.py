# show training/evaluation info
import os
import datetime
import sys
def sortfunc(zz):
    return -(zz[1])
def readX(nf):
    fp=open("./eval/gm%d.txt"%(nf+1),"r")
    tx=fp.readline()[:-1]
    fp.close()
    if(os.path.isfile("./eval/op%d.txt"%(nf+1))):
        fp=open("./eval/op%d.txt"%(nf+1),"r")
        tx1=fp.readline()[:-1]
        fp.close()
    else:
        tx1="opening not found"
    return float(tx),tx1
def toTxt(kk,ss):
    if(ss[0]>1.51):
        return "\033[94m eval %2d:  %.1f %-31s \033[0m"%(kk,ss[0],ss[1])
    elif(ss[0]>1.01):
        return "\033[92m eval %2d:  %.1f %-31s \033[0m"%(kk,ss[0],ss[1])
    elif(ss[0]<0.49):
        return "\033[91m eval %2d:  %.1f %-31s \033[0m"%(kk,ss[0],ss[1])
    elif(ss[0]<0.99):
        return "\033[93m eval %2d:  %.1f %-31s \033[0m"%(kk,ss[0],ss[1])
    else:
        return " eval %2d:  %.1f %-31s "%(kk,ss[0],ss[1])
if(sys.argv[-1]=="eval"):
    ddt=[[toTxt(i+1,readX(i)),(os.path.getmtime("./eval/gm%d.txt"%(i+1)))] for i in range(32)]
else:
    ddt=[["tr_dat %2d: "%(i+1),(os.path.getmtime("./games/dat_vlidn%d.npz"%(i+1)))] for i in range(32)]
sdt=sorted(ddt,key=sortfunc)
for i in range(32):
    print(sdt[i][0],end="")
    print(datetime.datetime.fromtimestamp(sdt[i][1]))