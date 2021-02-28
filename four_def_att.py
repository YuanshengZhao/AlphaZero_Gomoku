# train to response to four
# no more used
import numpy as np
import random
import tensorflow as tf
import RNG as RN_Gomoku

board=np.array(np.zeros([15,15,2]), dtype='<f4')
rst=np.array(np.zeros([15*15+1]), dtype='<f4')
drx=[[1,0],[0,1],[1,1],[1,-1]]

def printBoard(bd):
    print('   ',end='')
    for ii in range(15):
        print(' %d ' %(ii%10), end='')
    print('')
    for ii in range(15):
        print(' %d ' %(ii%10), end='')
        for jj in range(15):
            if(bd[ii,jj,0]==0 and bd[ii,jj,1]==0):
                print(' - ',end='')
            elif(bd[ii,jj,0]==1 and bd[ii,jj,1]==0):
                print(' O ',end='')
            elif(bd[ii,jj,0]==0 and bd[ii,jj,1]==1):
                print(' X ',end='')
            else:
                print(' E ',end='')
        print('')

def fromrst(rrst):
    ii=0
    while(ii<225):
        if(rrst[ii]>.1):
            print(ii//15,ii%15,rst[-1])
            break
        ii+=1

def genpos4(cleanness=0):
    global board,rst
    board*=0
    rst*=0
    bx=random.randint(2,12)
    by=random.randint(2,12)
    bk=random.randint(-2,2)
    dr=random.randint(0,3)
    sd=random.randint(0,1)
    for ii in range(-2,3):
        board[bx+ii*drx[dr][0]][by+ii*drx[dr][1]][sd]=1
    for ii in range(random.randint(4,5)):
        while(True):
            cx=random.randint(0,14)
            cy=random.randint(0,14)
            if(board[cx][cy][sd]==0):
                board[cx][cy][1-sd]=1
                break
    for ii in range(random.randint(0,cleanness)):
        while(True):
            cx=random.randint(0,14)
            cy=random.randint(0,14)
            if(board[cx][cy][0]==0 and board[cx][cy][1]==0):
                board[cx][cy][0]=1
                break
        while(True):
            cx=random.randint(0,14)
            cy=random.randint(0,14)
            if(board[cx][cy][0]==0 and board[cx][cy][1]==0):
                board[cx][cy][1]=1
                break
    board[bx+bk*drx[dr][0]][by+bk*drx[dr][1]][sd]=0
    rst[-1]=1-sd
    rst[(bx+bk*drx[dr][0])*15+by+bk*drx[dr][1]]=1
    return bx+bk*drx[dr][0],by+bk*drx[dr][1],1-sd

def genpos2(cleanness=0):
    global board,rst
    board*=0
    rst*=0
    bx=random.randint(1,13)
    by=random.randint(1,13)
    bk=random.randint(-1,1)
    dr=random.randint(0,3)
    sd=random.randint(0,1)
    for ii in range(-1,2):
        board[bx+ii*drx[dr][0]][by+ii*drx[dr][1]][sd]=1
    for ii in range(random.randint(2,3)):
        while(True):
            cx=random.randint(0,14)
            cy=random.randint(0,14)
            if(board[cx][cy][sd]==0):
                board[cx][cy][1-sd]=1
                break
    for ii in range(random.randint(0,cleanness)):
        while(True):
            cx=random.randint(0,14)
            cy=random.randint(0,14)
            if(board[cx][cy][0]==0 and board[cx][cy][1]==0):
                board[cx][cy][0]=1
                break
        while(True):
            cx=random.randint(0,14)
            cy=random.randint(0,14)
            if(board[cx][cy][0]==0 and board[cx][cy][1]==0):
                board[cx][cy][1]=1
                break
    board[bx+bk*drx[dr][0]][by+bk*drx[dr][1]][sd]=0
    rst[-1]=.6-.2*sd
    rst[(bx+bk*drx[dr][0])*15+by+bk*drx[dr][1]]=1
    return bx+bk*drx[dr][0],by+bk*drx[dr][1],1-sd
# print(genpos(10))
# printBoard(board)

cln=30

print("generating training data...")
x_train=[]
y_train=[]
for _ in range(60000):
    if(not _%1000):
        print(_)
    if(_<48000):
        gr=genpos4(cln)
    else:
        gr=genpos2(cln//2)
    x_train.append(board.copy())
    y_train.append(rst.copy())

printBoard(x_train[-1])
fromrst(y_train[-1])

print("generating validation data...")
x_validate=[]
y_validate=[]
for _ in range(10000):
    if(not _%1000):
        print(_)
    if(_<8000):
        gr=genpos4(cln)
    else:
        gr=genpos2(cln//2)
    x_validate.append(board.copy())
    y_validate.append(rst.copy())

printBoard(x_validate[-1])
fromrst(y_validate[-1])

x_train=np.array(x_train)
y_train=np.array(y_train,dtype=int)
x_validate=np.array(x_validate)
y_validate=np.array(y_validate,dtype=int)


RN_Gomoku.a0_eng.fit(x_train,y_train,epochs=2,shuffle=True,batch_size=250,validation_data=(x_validate,y_validate))
RN_Gomoku.a0_eng.save_weights("RNG.tf")