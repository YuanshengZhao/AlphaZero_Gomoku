# Warning: This "gui app" is mainly for test play.
# Do "normal things" or the behavior is undefined.

import tkinter as tk
from tkinter import messagebox
import MCTS

MCTS.setNumSimul(50)
MCTS.setFPU(1.3,1.3)
root=tk.Tk(className='AzG')
boolvar=tk.IntVar()
autoPlay=tk.Checkbutton(root,text="auto play",variable=boolvar)
autoPlay.place(x=100,y=10)
autoPlay.toggle()
dx=32
pce=int(dx/2.3)
sar=int(dx/10)
hdx=dx//2
barcenter=int(dx*19.5)
barlength=int(dx*1.25)

mvlst=[]
nblst=[]
pllst=[]
evlst=[]
brlst=[]
filled=[[0 for _ in range(15)] for __ in range(15)]

root.geometry("%dx%d"%(18*dx+30,21*dx+60))
canvas=tk.Canvas(root,width=18*dx,height=21*dx,bg="gold")
canvas.place(x=10,y=40)

def redrawEval():
    global brlst
    for brs in brlst:
        canvas.delete(brs)
    nbs=len(evlst)
    brlst=[]
    if(nbs==0):
        return
    brwd=min(dx/2,dx*14/nbs)
    for ii in range(nbs):
        ddy=((evlst[ii]-0.5)*2*barlength) if ii%2==0 else ((0.5-evlst[ii])*2*barlength)
        if(ddy>=1 or ddy<= -1):
            brlst.append(canvas.create_rectangle(dx*2+brwd*ii,barcenter,dx*2+brwd*(ii+1),barcenter-ddy,fill="black" if ddy>0 else "white"))
        else:
            brlst.append(canvas.create_line(dx*2+brwd*ii,barcenter,dx*2+brwd*(ii+1),barcenter,fill="black"))

def takeBack():
    if(len(mvlst)<1):
        print("Cannot take back!")
        return
    canvas.delete(mvlst[-1])
    canvas.delete(nblst[-1])
    filled[pllst[-1][0]][pllst[-1][1]]=0
    mvlst.pop()
    pllst.pop()
    nblst.pop()
    evlst.pop()
    redrawEval()
    MCTS.takeBack()
    MCTS.printBoard(MCTS.board)

def playStoneXY(px,py,evalu=0.5):
    if(px<0 or px>14 or py<0 or py>14):
        return
    print("play",px,py)
    if(filled[px][py]):
        print("Non empty!")
        return 1
    else:
        filled[px][py]=1
        mvlst.append(canvas.create_oval(dx*(py+2)-pce,dx*(px+2)-pce,dx*(py+2)+pce,dx*(px+2)+pce,width=2,fill="white" if MCTS.move_count%2==1 else "black"))
        nblst.append(canvas.create_text(dx*(py+2),dx*(px+2),text=str(MCTS.move_count+1),fill="red" if MCTS.move_count%2==1 else "cyan"))
        pllst.append([px,py])
        evlst.append(evalu)
        redrawEval()
        MCTS.applyMove(px*15+py)
        MCTS.printBoard(MCTS.board)
        if(MCTS.winLossDraw()!=-1):
            print("game over!")
            root.update()
            messagebox.showinfo(master=root,message="Game Over!\nFurther operation undefined!")
            return 1
    return 0

def ComputerMove():
    print("Computer thinking...")
    MCTS.timeReset()
    mv,dpt=MCTS.run_mcts(MCTS.evaluatePositionA,False)
    if(MCTS.side2move==1):
        print("%3d X %2d %2d %.3f, d %2d"%(MCTS.move_count,mv[0]//15,mv[0]%15,mv[2],dpt))
    else:
        print("%3d O %2d %2d %.3f, d %2d"%(MCTS.move_count,mv[0]//15,mv[0]%15,mv[2],dpt))
    MCTS.printTime()
    playStoneXY(mv[0]//15,mv[0]%15,mv[2])

def playStone(event):
    px,py=(event.y+hdx)//dx-2, (event.x+hdx)//dx-2
    if(playStoneXY(px,py)==0):
        if(boolvar.get()):
            root.update()
            ComputerMove()
    

def changeNNode(xx):
    try:
        xx=int(xx)
    except:
        print("Bad cmd!")
    if(xx>0):
        MCTS.setNumSimul(xx)
    else:
        print("Illegal number of nodes!")

btmEval=tk.Button(root,text="Comp Play",command=ComputerMove)
btmEval.place(x=10,y=10)
btmBack=tk.Button(root,text="Take Back",command=takeBack)
btmBack.place(x=200,y=10)
entNod=tk.Entry(root,width=5)
entNod.place(x=400,y=10)
entNod.insert(0,"50")
btmSnd=tk.Button(root,text="Set N_Node",command=lambda:changeNNode(entNod.get()))
btmSnd.place(x=300,y=10)



canvas.bind("<Button-1>",playStone)
for i in range(15):
    canvas.create_line(dx*2,dx*(i+2),dx*16,dx*(i+2))
    canvas.create_line(dx*(i+2),dx*2,dx*(i+2),dx*16)
    canvas.create_text(dx,dx*(i+2),text=str(i))
    canvas.create_text(dx*17,dx*(i+2),text=str(i))
    canvas.create_text(dx*(i+2),dx,text=str(i))
    canvas.create_text(dx*(i+2),dx*17,text=str(i))
canvas.create_rectangle(dx*(7 +2)-sar,dx*(7 +2)-sar,dx*(7 +2)+sar,dx*(7 +2)+sar,fill="black")
canvas.create_rectangle(dx*(3 +2)-sar,dx*(3 +2)-sar,dx*(3 +2)+sar,dx*(3 +2)+sar,fill="black")
canvas.create_rectangle(dx*(3 +2)-sar,dx*(11+2)-sar,dx*(3 +2)+sar,dx*(11+2)+sar,fill="black")
canvas.create_rectangle(dx*(11+2)-sar,dx*(3 +2)-sar,dx*(11+2)+sar,dx*(3 +2)+sar,fill="black")
canvas.create_rectangle(dx*(11+2)-sar,dx*(11+2)-sar,dx*(11+2)+sar,dx*(11+2)+sar,fill="black")
for i in [-1,1]:
    canvas.create_line(dx*2,barcenter+i*barlength,dx*16,barcenter+i*barlength,fill="blue")
root.mainloop() 