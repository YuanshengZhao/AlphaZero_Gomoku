import MCTS

MCTS.setNumSimul(4000)

def apmv(x,y):
    MCTS.applyMove(x*15+y)

apmv(7,7)
apmv(6,8)
indir=[[5,9],[6,9],[7,9],[8,9],[9,9],
                   [7,8],[8,8],[9,8],
                         [8,7],[9,7],
                         [8,6],[9,6],
                               [9,5]]
nms=[["長星 Chosei   ","-= "],
["峡月 Kyogetsu ","1-0"],
["恒星 Kosei    ","1-0"],
["水月 Suigetsu ","1-0"],
["流星 Ryusei   ","-= "],
["雲月 Ungetsu  ","1-0"],
["浦月 Hogetsu  ","1-0"],
["嵐月 Rangetsu ","1-0"],
["銀月 Gingetsu ","+- "],
["明星 Myojo    ","1-0"],
["斜月 Shagetsu ","+= "],
["名月 Meigetsu ","+- "],
["彗星 Suisei   ","0-1"]]
for k in range(len(indir)):
    apmv(*(indir[k]))
    mv,dpt=MCTS.run_mcts(MCTS.evaluatePositionA,False)
    # MCTS.printBoard(MCTS.board)
    print("|",nms[k][0],"|",indir[k][0],indir[k][1]," |",nms[k][1],"   |",end="")
    print(" %d %d   | %.3f |"%(mv[0]//15,mv[0]%15,mv[2]))
    MCTS.takeBack()

print("")
MCTS.takeBack()
apmv(6,7)
indir=[[5,7],[5,8],[5,9],
             [6,8],[6,9],
             [7,8],[7,9],
       [8,7],[8,8],[8,9],
       [9,7],[9,8],[9,9]]
nms=[["寒星 Kansei   ","1-0"],
["渓月 Keigetsu ","1-0"],
["疎星 Sosei    ","=  "],
["花月 Kagetsu  ","1-0"],
["残月 Zangetsu ","+- "],
["雨月 Ugetsu   ","1-0"],
["金星 Kinsei   ","1-0"],
["松月 Shogetsu ","+- "],
["丘月 Kyugetsu ","+= "],
["新月 Shingetsu","+- "],
["瑞星 Zuisei   ","=  "],
["山月 Sangetsu ","+- "],
["遊星 Yusei    ","0-1"]]
for k in range(len(indir)):
    apmv(*(indir[k]))
    mv,dpt=MCTS.run_mcts(MCTS.evaluatePositionA,False)
    # MCTS.printBoard(MCTS.board)
    print("|",nms[k][0],"|",indir[k][0],indir[k][1]," |",nms[k][1],"   |",end="")
    print(" %d %d   | %.3f |"%(mv[0]//15,mv[0]%15,mv[2]))
    MCTS.takeBack()
