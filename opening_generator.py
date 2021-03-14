import random
# apmv(7,7)
# apmv(6,8)
indir=[[5,9],[6,9],[7,9],[8,9],[9,9],
                   [7,8],[8,8],[9,8],
                         [8,7],[9,7],
                         [8,6],[9,6],
                               [9,5]]
# nms=[["長星 Chosei   ","-= "],
# ["峡月 Kyogetsu ","1-0"],
# ["恒星 Kosei    ","1-0"],
# ["水月 Suigetsu ","1-0"],
# ["流星 Ryusei   ","-= "],
# ["雲月 Ungetsu  ","1-0"],
# ["浦月 Hogetsu  ","1-0"],
# ["嵐月 Rangetsu ","1-0"],
# ["銀月 Gingetsu ","+- "],
# ["明星 Myojo    ","1-0"],
# ["斜月 Shagetsu ","+= "],
# ["名月 Meigetsu ","+- "],
# ["彗星 Suisei   ","0-1"]]
idx=1
for k in range(len(indir)):
    fp=open("./eval/op%d.txt"%(idx),"w")
    fp.write("7 7 6 8 %d %d\n"%(indir[k][0],indir[k][1]))
    fp.close()
    idx+=1

# apmv(6,7)
indir=[[5,7],[5,8],[5,9],
             [6,8],[6,9],
             [7,8],[7,9],
       [8,7],[8,8],[8,9],
       [9,7],[9,8],[9,9]]
# nms=[["寒星 Kansei   ","1-0"],
# ["渓月 Keigetsu ","1-0"],
# ["疎星 Sosei    ","=  "],
# ["花月 Kagetsu  ","1-0"],
# ["残月 Zangetsu ","+- "],
# ["雨月 Ugetsu   ","1-0"],
# ["金星 Kinsei   ","1-0"],
# ["松月 Shogetsu ","+- "],
# ["丘月 Kyugetsu ","+= "],
# ["新月 Shingetsu","+- "],
# ["瑞星 Zuisei   ","=  "],
# ["山月 Sangetsu ","+- "],
# ["遊星 Yusei    ","0-1"]]
for k in range(len(indir)):
    fp=open("./eval/op%d.txt"%(idx),"w")
    fp.write("7 7 6 7 %d %d\n"%(indir[k][0],indir[k][1]))
    fp.close()
    idx+=1

flanks=[[ii,jj] for ii in range(15) for jj in range(15) if (ii>1 and ii<7 and jj>1 and jj<7 and ii>=jj) and not(ii>4 and jj>4)]
print(len(flanks))
sele=random.sample(flanks,6)
for k in range(6):
    fp=open("./eval/op%d.txt"%(idx),"w")
    fp.write("%d %d\n"%(sele[k][0],sele[k][1]))
    fp.close()
    idx+=1
print(sele)