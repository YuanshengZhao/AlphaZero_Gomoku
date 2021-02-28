# hash table

import numpy as np
class ZobristHash(object):
    def __init__(self,size=100000):
        self.hash_size=1<<((size-1).bit_length()-1)
        self.table=np.random.randint(0,self.hash_size,size=(15,15,2))
        self.values=np.random.rand(self.hash_size,226)
        self.boardcheck=np.array(np.zeros([self.hash_size,15,15,2])-1, dtype=int)
        self.initkey=np.random.randint(self.hash_size)
        print("hash size:",self.hash_size)
    def getValue(self,boardX):
        key=self.initkey
        for ii in range(15):
            for jj in range(15):
                if(boardX[ii,jj,0]):
                    key^=self.table[ii,jj,0]
                if(boardX[ii,jj,1]):
                    key^=self.table[ii,jj,1]
        if(np.array_equal(boardX,self.boardcheck[key])):
            # print("Hash found!")
            return key,self.values[key]
        else:
            return key,[]
    def setValue(self,key,boardX,value):
        self.values[key]=value.copy()
        self.boardcheck[key]=boardX.copy()
