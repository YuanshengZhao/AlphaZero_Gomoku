# self play
print("loading...")
import numpy as np
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
import MCTS
import sys

if(len(sys.argv)<=1):
    print("Error! No argument given! Quiting.")
    quit()
    
MCTS.setFPU(1.1,0.9)

MCTS.timeReset()
MCTS.selfPlay(20,MCTS.evaluatePositionA,"./games/dat_train"+sys.argv[1]+".npz",40)
#vldn time is 2 as the argument is the total time from calling time reset
MCTS.selfPlay(23,MCTS.evaluatePositionA,"./games/dat_vlidn"+sys.argv[1]+".npz",10)

# npz_t=np.load("dat_train.npz")
# x_tr=npz_t['arr_0']
# y_tr=npz_t['arr_1']

# for i in range(len(y_tr)):
#     MCTS.showHeatMap(x_tr[i],y_tr[i])
