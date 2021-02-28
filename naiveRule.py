# a weak engine
import numpy as np
import random

drx=[[1,0],[0,1],[1,1],[1,-1]]

def boardXWalk(boardX,sx,sy,dx,dy,cr):
    bcr=1-cr
    cnt,mds,sbc,eds=0,0,0,0
    state=0
    cx,cy=sx+dx,sy+dy
    while(cx>=0 and cy>=0 and cx<15 and cy<15):
        if(state==0):
            if(boardX[cx][cy][cr]==1):
                cnt+=1
            elif(boardX[cx][cy][bcr]==1):
                break
            else:
                state=1
                mds+=1
        elif(state==1):
            if(boardX[cx][cy][cr]==1):
                state=2
                sbc+=1
            elif(boardX[cx][cy][bcr]==1):
                break
            else:
                mds+=1
        elif(state==2):
            if(boardX[cx][cy][cr]==1):
                sbc+=1
            elif(boardX[cx][cy][bcr]==1):
                break
            else:
                state=3
                eds+=1
        else:
            if(boardX[cx][cy][cr]==1):
                eds+=1
            elif(boardX[cx][cy][bcr]==1):
                break
            else:
                eds+=1
        cx,cy=cx+dx,cy+dy
    return cnt,mds,sbc,eds


def evalpos(boardX,xx,yy,cr):
    if(boardX[xx][yy][0]==1 or boardX[xx][yy][1]==1):
        return 0
    scr=0
    for dx in drx:
        cnt1,mds1,sbc1,eds1=boardXWalk(boardX,xx,yy,dx[0],dx[1],cr)
        cnt2,mds2,sbc2,eds2=boardXWalk(boardX,xx,yy,-dx[0],-dx[1],cr)
        cc=cnt1+cnt2+1
        if(cc>=5):
            scr+=10000
        elif(cc==4):
            if(mds1>0 and mds2>0):
                scr+=500
            elif(mds1>0 or mds2>0):
                scr+=56
        elif(cc==3):
            if(mds1>0 and mds2>0):
                scr+=30
            elif((mds1>0 or mds2>0) and (mds1+mds2>2)):
                 scr+=4
        elif(cc==2):
            if(mds1>0 and mds2>0 and mds1+mds2+sbc1+sbc2+eds1+eds2>2):
                scr+=4
                if((mds1==1 and sbc1>0) or (mds2==1 and sbc2>0)):
                    scr+=16
            elif((mds1>0 or mds2>0) and mds1+mds2+sbc1+sbc2+eds1+eds2>2):
                scr+=2
        elif(cc==1):
            if((mds1>0 or mds2>0) and mds1<2 and mds2<2 and mds1+mds2+sbc1+sbc2+eds1+eds2>3):
                scr+=2
                if(mds1<2 or mds2<2):
                    scr+=1
                    if((mds1==1 and sbc1>1) or (mds2==1 and sbc2>1)):
                        scr+=17
    return scr
def evalboardX(boardX):
    b1=np.array([[evalpos(boardX,xx,yy,1) for yy in range(15)] for xx in range(15)])
    b2=np.array([[evalpos(boardX,xx,yy,0) for yy in range(15)] for xx in range(15)])
    bf=b2+b1*.7+1.0e-8
    bf=bf/np.sum(bf)
    sc=np.sum(b2-b1*.7)
    # fig, ax = plt.subplots()
    # im = ax.imshow(bf)
    # for i in range(15):
    #     for j in range(15):
    #         text = ax.text(j, i, 
    #         toTex(i,j),
    #                    ha="center", va="center", color="w")
    # plt.show()
    return np.append(bf.flatten(),1/(1+np.exp(-sc/50.0)))
