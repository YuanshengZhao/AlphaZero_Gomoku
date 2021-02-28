# human play against engine

import MCTS

MCTS.setNumSimul(50)
MCTS.setFPU(1.5,1.5)

while(True):
    msg=input("input cmd > ")
    if(len(msg)<1):
        print("Bad cmd!")
        continue
    if(msg[0]=="p"):#human play
        try:
            xx,yy=map(int,msg[2:].split())
        except:
            print("Bad cmd!")
            continue
        hmv=xx*15+yy
        print("Play",xx,yy)
        if(xx>=0 and yy>=0 and xx<15 and yy<15 and MCTS.board[xx][yy][0]==0 and MCTS.board[xx][yy][1]==0):
            MCTS.applyMove(hmv)
            MCTS.printBoard(MCTS.board)
            if(MCTS.winLossDraw()!=-1):
                print("game over!")
                quit()
        else:
            print("Illegal move!")
    elif(msg[0]=="f"):#fpu
        try:
            xx,yy=map(float,msg[2:].split())
        except:
            print("Bad cmd!")
            continue
        MCTS.setFPU(xx,yy)
    elif(msg[0]=="n"):#change N_Node
        try:
            xx=int(msg[2:])
        except:
            print("Bad cmd!")
            continue
        if(xx>0):
            MCTS.setNumSimul(xx)
        else:
            print("Illegal number of nodes!")
    elif(msg[0]=="v"):#eng play
        print("Computer thinking...")
        MCTS.timeReset()
        mv,dpt=MCTS.run_mcts(MCTS.evaluatePositionA,False)
        if(MCTS.side2move==1):
            print("%3d X %2d %2d %.3f, d %2d"%(MCTS.move_count,mv[0]//15,mv[0]%15,mv[2],dpt))
        else:
            print("%3d O %2d %2d %.3f, d %2d"%(MCTS.move_count,mv[0]//15,mv[0]%15,mv[2],dpt))
        MCTS.applyMove(mv[0])
        MCTS.printBoard(MCTS.board)
        MCTS.printTime()
        if(MCTS.winLossDraw()!=-1):
            print("game over!")
            quit()
    elif(msg[0]=="b"):#takeback
        if(MCTS.move_count>0):
            MCTS.takeBack()
            MCTS.printBoard(MCTS.board)
        else:
            print("No moves back!")
    elif(msg[0]=="e"):#exit
        quit()
    else:
        print("Bad cmd!")
