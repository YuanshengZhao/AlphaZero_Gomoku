# evaluation

print("loading...")
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
import MCTS
import sys
import time

if(len(sys.argv)>3):
    MCTS.loadEngine(1,sys.argv[2])
    MCTS.loadEngine(2,sys.argv[3])
else:
    MCTS.loadEngine(1,"./weights/RNG64.tf")
    MCTS.loadEngine(2,"./RNG64.tf")
MCTS.setFPU(1.3,1.3)

rst=MCTS.versus(2,MCTS.evaluatePositionVA)

if(len(sys.argv)>1):
    fp=open("./eval/gm"+sys.argv[1]+".txt","w")
    fp.write(str(rst)+"\n")
    # fp.write(time.strftime("%H:%M:%S",time.localtime())+"\n")
    fp.close()