import numpy as np
import matplotlib.pyplot as plt
import random
import time
import datetime
import RNG as RN_Gomoku
import naiveRule
import Zobrist

start_time = time.time()
def timeReset():
    global start_time
    start_time = time.time()
def printTime(tital=-1.0):
    ss=time.time()-start_time
    print("time used = " + str(datetime.timedelta(seconds=ss)))
    if(tital>ss):
        print("time remaining = " + str(datetime.timedelta(seconds=tital-ss)))
        return False
    else:
        return True


board=np.array(np.zeros([15,15,2]), dtype=int)
# board[7,7,0]=1
side2move=0
movelist=[7*16 for i in range(225)]
move_count=0
num_simul=800

def printBoard(bd):
    print('   ',end='')
    for ii in range(15):
        print(' %d ' %(ii%10), end='')
    print('')
    for ii in range(15):
        print(' %d ' %(ii%10), end='')
        for jj in range(15):
            if(bd[ii,jj,0]==0 and bd[ii,jj,1]==0):
                if((ii,jj)==(7,7) or (ii==3 or ii==11) and (jj==3 or jj==11)):
                    print(' + ',end='')
                else:
                    print(' - ',end='')
            elif(bd[ii,jj,0]==1 and bd[ii,jj,1]==0):
                print('\033[96m O \033[0m',end='')
            elif(bd[ii,jj,0]==0 and bd[ii,jj,1]==1):
                print('\033[91m X \033[0m',end='')
            else:
                print(' E ',end='')
        print(' %d' %(ii%10))
    print('   ',end='')
    for ii in range(15):
        print(' %d ' %(ii%10), end='')
    print('')

def toTex(bd,ii,jj):
    if(bd[ii,jj,0]==0 and bd[ii,jj,1]==0):
        return ' - '
    elif(bd[ii,jj,0]==1 and bd[ii,jj,1]==0):
        return ' O '
    elif(bd[ii,jj,0]==0 and bd[ii,jj,1]==1):
        return ' X '
    else:
        return ' E '
def showHeatMap(x_tr,y_tr,z_tr=None):
    if(z_tr is None):
        print(y_tr[-1])
        _, ax = plt.subplots()
        ax.imshow(np.reshape(y_tr[:-1],[15,15]))
        for ii in range(15):
            for jj in range(15):
                ax.text(jj, ii, 
                toTex(x_tr,ii,jj),
                           ha="center", va="center", color="w")
        ax.set_xticks([], [])
        ax.set_yticks([], [])
        plt.show()
    else:
        print(y_tr[-1],z_tr[-1])
        _, (ax1,ax2) = plt.subplots(1,2)
        ax1.imshow(np.reshape(y_tr[:-1],[15,15]))
        for ii in range(15):
            for jj in range(15):
                ax1.text(jj, ii, 
                toTex(x_tr,ii,jj),
                           ha="center", va="center", color="w")
        ax1.set_xticks([], [])
        ax1.set_yticks([], [])
        ax2.imshow(np.reshape(z_tr[:-1],[15,15]))
        for ii in range(15):
            for jj in range(15):
                ax2.text(jj, ii, 
                toTex(x_tr,ii,jj),
                           ha="center", va="center", color="w")
        ax2.set_xticks([], [])
        ax2.set_yticks([], [])
        plt.show()


class Node(object):
    def __init__(self,prior: float,sidem: int):
        self.visit_count=0
        self.to_play=sidem
        self.prior=prior
        self.value_sum=0.0
        self.actions=[]
        self.children=[]

    def set_state(self,prior: float,sidem: int):
        self.visit_count=0
        self.to_play=sidem
        self.prior=prior
        self.value_sum=0.0
        self.actions=[]
        self.children=[]

    def expanded(self):
        return len(self.children)>0

    def value(self):
        # if self.visit_count==0:
            # return 1.6-parantval
            # return 1.0
            #unvisited node treat as loss, should not visit
            #different sign in ucb formula results in 1 being here
        return self.value_sum/self.visit_count
        # error when calling at no visit

nodes=[Node(0.1,1) for _ in range(max((num_simul+1)*224,500))]
nodes_used=0
def setNumSimul(nsimul):
    global num_simul,nodes
    num_simul=nsimul
    if((num_simul+1)*224>len(nodes)):
        nodes=[Node(0.1,1) for _ in range(max((num_simul+1)*224,500))]
    print("num_simul set to",num_simul)


def leagal_actions():
    return [ii*15+jj for ii in range(15) for jj in range(15) if board[ii][jj][0]==0 and board[ii][jj][1]==0]

def winLossDraw():#return O wins or X wins
    sd=1-side2move
    posx,posy=movelist[move_count-1]//15,movelist[move_count-1]%15

    lx,lg=1,1
    while(posx-lx>=0 and board[posx-lx][posy][sd]==1):
        lg+=1
        lx+=1
    lx=1
    while(posx+lx<15 and board[posx+lx][posy][sd]==1):
        lg+=1
        lx+=1
    if(lg>=5):
        return side2move

    lx,lg=1,1
    while(posy-lx>=0 and board[posx][posy-lx][sd]==1):
        lg+=1
        lx+=1
    lx=1
    while(posy+lx<15 and board[posx][posy+lx][sd]==1):
        lg+=1
        lx+=1
    if(lg>=5):
        return side2move

    lx,lg=1,1
    while(posx-lx>=0 and posy-lx>=0 and board[posx-lx][posy-lx][sd]==1):
        lg+=1
        lx+=1
    lx=1
    while(posx+lx<15 and posy+lx<15 and board[posx+lx][posy+lx][sd]==1):
        lg+=1
        lx+=1
    if(lg>=5):
        return side2move

    lx,lg=1,1
    while(posx-lx>=0 and posy+lx<15 and board[posx-lx][posy+lx][sd]==1):
        lg+=1
        lx+=1
    lx=1
    while(posx+lx<15 and posy-lx>=0 and board[posx+lx][posy-lx][sd]==1):
        lg+=1
        lx+=1
    if(lg>=5):
        return side2move
    
    if(move_count==225):
        return 0.5
    return -1



side_id=0

# def evaluatePosition():
#     if(side_id==0):
#         if(side2move==1):
#             rstt=eng1(np.flip(board,-1))
#             rstt[-1]=1-rstt[-1]
#             # rstt[-1]=.5
#         else:
#             rstt=eng1(board.copy())
#             # rstt[-1]=.5
#         return rstt
#     else:
#         if(side2move==1):
#             rstt=eng2(np.flip(board,-1))
#             rstt[-1]=1-rstt[-1]
#         else:
#             rstt=eng2(board.copy())
#         return rstt
#     # return [0.5 for i in range(226)]

def evaluate(node: Node,evaluatePosition):
    global nodes_used
    wld=winLossDraw()
    if(wld!=-1):
        return wld
    policy_logits=evaluatePosition()
    node.actions=leagal_actions()
    policy=[policy_logits[i] for i in node.actions]
    policy_sum = sum(policy)
    num_p=len(node.actions)
    node.children=nodes[nodes_used:nodes_used+num_p]
    nodes_used+=num_p
    tp=1-node.to_play
    # print(num_p,len(node.children))
    for i in range(num_p):
        node.children[i].set_state(policy[i]/policy_sum,tp)
    return policy_logits[-1]

def add_exploration_noise(node: Node):
    noise=np.random.dirichlet(np.full(len(node.actions),.05))
    for i in range(len(node.actions)):
        node.children[i].prior = node.children[i].prior*.75+noise[i]*.25

fpuReduction=1.3
fpuReductionRoot=1.3
#fpu is modifyed in sfvs and sfpl. chaning it here no longer has any effect!
def setFPU(rdc,rdcr):
    global fpuReduction,fpuReductionRoot
    fpuReduction,fpuReductionRoot=rdc,rdcr
    print("FPU (normal/root) set to ",fpuReduction,fpuReductionRoot)

def ucb_score(parent: Node, child: Node,isnotroot=True):
    pb_c=np.log((parent.visit_count+19653)/19652)+1.25
    pb_c*=np.sqrt(parent.visit_count)/(child.visit_count+1)
    prior_score=pb_c*child.prior
    if(child.visit_count==0):
        return (prior_score-fpuReduction+parent.value()) if isnotroot else (prior_score-fpuReductionRoot+parent.value())
        # return prior_score-min(fpuReduction-parent.value(),1.0)
        #alpha zero use 0 and 1 in reality
        # up to lv9, the reduction is clipped between 0 and 1
    else:
        return prior_score-child.value()

def select_child(node: Node,isnotroot=True):
    _,action,child = max([(ucb_score(node,node.children[i],isnotroot)+np.random.uniform(-1e-4,1e-4),node.actions[i],node.children[i]) for i in range(len(node.actions))])
    return action,child

def applyMove(pos):
    global board,side2move,move_count,movelist
    board[pos//15][pos%15][side2move]=1
    side2move=1-side2move
    movelist[move_count]=pos
    move_count+=1

def takeBack():
    global board,side2move,move_count
    move_count-=1
    pos=movelist[move_count]
    side2move=1-side2move
    board[pos//15][pos%15][side2move]=0

def backpropagate(search_path, value: float):
    for node in search_path:
        node.value_sum += value if node.to_play == 0 else (1 - value)
        node.visit_count += 1

def select_action(root: Node,add_noise=True):
    if(move_count>=15 or (not add_noise)):
        visit_counts = [(root.children[i].visit_count+np.random.uniform(-1e-1,1e-1),root.actions[i]) for i in range(len(root.actions))]
        _, action = max(visit_counts)
    else:
        visit_counts = np.array([(root.children[i].visit_count)**1.25 for i in range(len(root.actions))])
        visit_counts_sum=np.sum(visit_counts)
        action = np.random.choice(root.actions,p=visit_counts/visit_counts_sum)

    pol_map=np.zeros([15,15])
    visit_counts = np.array([root.children[i].visit_count for i in range(len(root.actions))])
    visit_counts_sum=np.sum(visit_counts)
    for i in range(len(root.actions)):
        pol_map[root.actions[i]//15][root.actions[i]%15]=visit_counts[i]/visit_counts_sum
    return action,pol_map,root.value()

def run_mcts(evaluatePosition,add_noise=True):
    global nodes_used
    nodes_used=0
    root = Node(1.0,side2move)
    root.visit_count=1
    root.value_sum=evaluate(root,evaluatePosition)
    depth=0
    maxdepth=0
    if(add_noise):
        add_exploration_noise(root)

    for _ in range(num_simul):
        node=root
        search_path=[node]
        depth=0

        while node.expanded():
            depth+=1
            action,node=select_child(node,depth>1)
            applyMove(action)
            search_path.append(node)

        value=evaluate(node,evaluatePosition)
        backpropagate(search_path,value)
        for __ in range(len(search_path)-1):
            takeBack()

        maxdepth=max(maxdepth,depth)
    # for ii in range(len(root.actions)):
    #     print("%3d %3d %6f %6f %3d"%(root.actions[ii]//15,root.actions[ii]%15,root.children[ii].prior,root.children[ii].value(),root.children[ii].visit_count))
    return select_action(root,add_noise),maxdepth

a0eng =RN_Gomoku.A0_ENG(64,"./weights/RNG64.tf")
a0eng2=RN_Gomoku.A0_ENG(64,"./RNG64.tf")

hashTB=Zobrist.ZobristHash(500000)
hashTB2=Zobrist.ZobristHash(500000)

def evaluatePositionV2():# naive vs nn_new
    if(side_id==0):
        kkey,rstt=hashTB.getValue(board)
        if(len(rstt)>0):
            return rstt
        else:
            if(side2move==1):
                rstt=naiveRule.evalboardX(np.flip(board,-1))
                rstt[-1]=1-rstt[-1]
                # rstt[-1]=.5
            else:
                rstt=naiveRule.evalboardX(board.copy())
                # rstt[-1]=.5
            hashTB.setValue(kkey,board,rstt)
            return rstt
    else:
        kkey,rstt=hashTB2.getValue(board)
        if(len(rstt)>0):
            return rstt
        else:
            if(side2move==1):
                rstt=a0eng2.a0_eng(np.array([np.flip(board,-1)],dtype='<f4'),training=False)[0].numpy()
                rstt[-1]=1-rstt[-1]
            else:
                rstt=a0eng2.a0_eng(np.array([board.copy()],dtype='<f4'),training=False)[0].numpy()
            hashTB2.setValue(kkey,board,rstt)
            return rstt


def evaluatePositionV():# naive vs nn_old
    if(side_id==0):
        kkey,rstt=hashTB.getValue(board)
        if(len(rstt)>0):
            return rstt
        else:
            if(side2move==1):
                rstt=naiveRule.evalboardX(np.flip(board,-1))
                rstt[-1]=1-rstt[-1]
                # rstt[-1]=.5
            else:
                rstt=naiveRule.evalboardX(board.copy())
                # rstt[-1]=.5
            hashTB.setValue(kkey,board,rstt)
            return rstt
    else:
        kkey,rstt=hashTB2.getValue(board)
        if(len(rstt)>0):
            return rstt
        else:
            if(side2move==1):
                rstt=a0eng.a0_eng(np.array([np.flip(board,-1)],dtype='<f4'),training=False)[0].numpy()
                rstt[-1]=1-rstt[-1]
            else:
                rstt=a0eng.a0_eng(np.array([board.copy()],dtype='<f4'),training=False)[0].numpy()
            hashTB2.setValue(kkey,board,rstt)
            return rstt

# def randomPos(psx,psy):
#     if(psx<0):
#         while(True):
#             inx,iny=random.randint(0,14),random.randint(0,14)
#             if(board[inx,iny,0]==0 and board[inx,iny,1]==0):
#                 return inx,iny
#     else:
#         while(True):
#             inx,iny=random.randint(-1,1),random.randint(-1,1)
#             if((inx!=0 or iny !=0) and psx+inx>=0 and psx+inx<15 and psy+iny>=0 and psy+iny<15 and board[psx+inx,psy+iny,0]==0 and board[psx+inx,psy+iny,1]==0):
#                 return psx+inx,psy+iny
        

# def evaluatePositionVA():# nn_old vs nn_new
#     if(side_id==0):
#         if(side2move==1):
#             rstt=a0eng.a0_eng(np.array([np.flip(board,-1)],dtype='<f4'),training=False)[0].numpy()
#             rstt[-1]=1-rstt[-1]
#         else:
#             rstt=a0eng.a0_eng(np.array([board.copy()],dtype='<f4'),training=False)[0].numpy()
#         return rstt
#     else:
#         if(side2move==1):
#             rstt=a0eng2.a0_eng(np.array([np.flip(board,-1)],dtype='<f4'),training=False)[0].numpy()
#             rstt[-1]=1-rstt[-1]
#         else:
#             rstt=a0eng2.a0_eng(np.array([board.copy()],dtype='<f4'),training=False)[0].numpy()
#         return rstt

def evaluatePositionVA():# nn_old vs nn_new
    if(side_id==0):
        kkey,rstt=hashTB.getValue(board)
        if(len(rstt)>0):
            return rstt
        else:
            if(side2move==1):
                rstt=a0eng.a0_eng(np.array([np.flip(board,-1)],dtype='<f4'),training=False)[0].numpy()
                rstt[-1]=1-rstt[-1]
            else:
                rstt=a0eng.a0_eng(np.array([board.copy()],dtype='<f4'),training=False)[0].numpy()
            hashTB.setValue(kkey,board,rstt)
            return rstt
    else:
        kkey,rstt=hashTB2.getValue(board)
        if(len(rstt)>0):
            return rstt
        else:
            if(side2move==1):
                rstt=a0eng2.a0_eng(np.array([np.flip(board,-1)],dtype='<f4'),training=False)[0].numpy()
                rstt[-1]=1-rstt[-1]
            else:
                rstt=a0eng2.a0_eng(np.array([board.copy()],dtype='<f4'),training=False)[0].numpy()
            hashTB2.setValue(kkey,board,rstt)
            return rstt

def randomMove(maxn):
    return random.sample([[i,j] for i in range(15) for j in range(15)],random.randint(1,maxn))

def versus(num_games,engine):
    global board,side2move,side_id,move_count
    psr=0.0
    bookmove=[]
    for ngames in range(num_games):
        print(ngames+1,"/",num_games,"current score for 1:",ngames-psr)
        board=np.array(np.zeros([15,15,2]), dtype=int)
        #generate book move per 2 game.
        if(not (ngames%2)):
            bookmove=randomMove(3)
            # print(bookmove)
        for ix in range(len(bookmove)):
            board[bookmove[ix][0],bookmove[ix][1],ix%2]=1
            if(ix%2):
                print("    X - %2d %2d Book"%(bookmove[ix][0],bookmove[ix][1]))
            else:
                print("    O - %2d %2d Book"%(bookmove[ix][0],bookmove[ix][1]))
        side2move=len(bookmove)%2
        side_id=ngames%2
        move_count=len(bookmove)

        while(winLossDraw()==-1):
            timeReset()
            mv,dpt=run_mcts(engine,False)
            if(side2move==1):
                print("%3d X %d %2d %2d %.3f, d %2d "%(move_count,side_id,mv[0]//15,mv[0]%15,mv[2],dpt),end="")
                printTime()
            else:
                print("%3d O %d %2d %2d %.3f, d %2d "%(move_count,side_id,mv[0]//15,mv[0]%15,mv[2],dpt),end="")
                printTime()
            applyMove(mv[0])
            side_id=1-side_id
        grst=winLossDraw()
        printBoard(board)
        if(grst==1 or grst==0):
            psr+=side_id
            print(grst,1-side_id,"wins!")
        else:
            psr+=.5
            print(grst,"draw!")

    print(1,"rate",(num_games-psr)/num_games)
    print(0,"rate",psr/num_games)
    return num_games-psr



def evaluatePositionN():# naive
    kkey,rstt=hashTB.getValue(board)
    if(len(rstt)>0):
        return rstt
    else:
        if(side2move==1):
            rstt=naiveRule.evalboardX(np.flip(board,-1))
            rstt[-1]=1-rstt[-1]
            # rstt[-1]=.5
        else:
            rstt=naiveRule.evalboardX(board.copy())
            # rstt[-1]=.5
        hashTB.setValue(kkey,board,rstt)
        return rstt

def evaluatePositionA():# nn_old
    kkey,rstt=hashTB.getValue(board)
    if(len(rstt)>0):
        return rstt
    else:
        if(side2move==1):
            rstt=a0eng.a0_eng(np.array([np.flip(board,-1)],dtype='<f4'),training=False)[0].numpy()
            rstt[-1]=1-rstt[-1]
        else:
            rstt=a0eng.a0_eng(np.array([board.copy()],dtype='<f4'),training=False)[0].numpy()
        hashTB.setValue(kkey,board,rstt)
        return rstt
def evaluatePositionA2():# nn_new
    kkey,rstt=hashTB.getValue(board)
    if(len(rstt)>0):
        return rstt
    else:
        if(side2move==1):
            rstt=a0eng2.a0_eng(np.array([np.flip(board,-1)],dtype='<f4'),training=False)[0].numpy()
            rstt[-1]=1-rstt[-1]
        else:
            rstt=a0eng2.a0_eng(np.array([board.copy()],dtype='<f4'),training=False)[0].numpy()
        hashTB.setValue(kkey,board,rstt)
        return rstt


def selfPlay(num_games,engine,outfile,maxgames=40):
    global board,side2move,side_id,move_count
    x_tr=[]
    y_tr=[]
    totalTime=num_games*9*60.0
#at least 1 game will be played
    for ngames in range(maxgames):
        print("playing game",ngames+1)
        x_tr0=[]
        y_tr0=[]
        board=np.array(np.zeros([15,15,2]), dtype=int)
        bookmove=randomMove(3)
        lbm=len(bookmove)
        for ix in range(lbm):
            board[bookmove[ix][0],bookmove[ix][1],ix%2]=1
            if(ix%2):
                print("    X - %2d %2d Book"%(bookmove[ix][0],bookmove[ix][1]))
            else:
                print("    O - %2d %2d Book"%(bookmove[ix][0],bookmove[ix][1]))
        side2move=lbm%2
        side_id=ngames%2
        move_count=lbm

        while(winLossDraw()==-1):
            if(side2move==1):
                x_tr0.append(1.0*np.flip(board,-1).copy())
            else:
                x_tr0.append(1.0*board.copy())
            mv,dpt=run_mcts(engine,True)
            if(side2move==1):
                print("%3d X %d %2d %2d %.3f, d %2d"%(move_count,side_id,mv[0]//15,mv[0]%15,mv[2],dpt))
            else:
                print("%3d O %d %2d %2d %.3f, d %2d"%(move_count,side_id,mv[0]//15,mv[0]%15,mv[2],dpt))
            applyMove(mv[0])
            side_id=1-side_id
            y_tr0.append(np.append(mv[1].flatten(),[.5]))
        grst=winLossDraw()
        printBoard(board)
        print("result of game:",grst,"in",move_count,"moves")
        for i in range(len(y_tr0)):
            y_tr0[i][-1]=grst if (i+lbm)%2==0 else (1-grst) 
        # print(y_tr0[-1][-1])

        x_tr0=np.array(x_tr0,dtype='<f4')
        y_tr0=np.array(y_tr0,dtype='<f4')
        if(len(x_tr)==0):
            x_tr=x_tr0.copy()
            y_tr=y_tr0.copy()
        else:
            x_tr=np.concatenate((x_tr,x_tr0))
            y_tr=np.concatenate((y_tr,y_tr0))

        if(printTime(totalTime)):
            break

    print("played",ngames+1,"game(s)")
    np.savez(outfile,x_tr,y_tr)