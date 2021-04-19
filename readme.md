# AzGomoku
Alpha (Go) (Zero) algorithm on Gomoku (Five In A Row) game. The code is largely based on the pseudo-code provided by Alpha Zero paper.
[Alpha Go Zero paper](https://www.nature.com/articles/nature24270?sf123103138=1)  
[Alpha Zero paper](https://science.sciencemag.org/content/362/6419/1140)

## Play against the engine
Use `hmplGUI.py` (Python), `hmplGUIC.py` (C++, fastest), or go to https://yuanshengzhao.github.io/AlphaZero_Gomoku/ (Javascript, similar speed with Python).

## Network
Reference to: [Lczero](https://lczero.org/dev/backend/nn/).

- Board: 15x15x2. maps: [stones-self, stones-opponent]. No need for color or history.
- Init: Conv2D, 3x3 64 filters   
- Body: 10 block [SE](https://arxiv.org/abs/1709.01507)-[ResNet](https://arxiv.org/abs/1512.03385) tower, 3x3, 64 filters, 32 SE_channels.  
- Body followed by:
    - P-Head: Conv2D(3,64) -- Conv2D(3,1)   
    - V-Head: Conv2D(3,32) -- Dense(128) --Dense(1)
- other value for nblock or nfilter is also supported.

## Parameters
- WDL encoding: [1/0.5/0]((https://lczero.org/blog/2018/12/alphazero-paper-and-lc0-v0191/)) (not +1/0/-1 stated in the paper)
- `cpuct_base`: 19652
- `cpuct`: 1.25
- `FPU`: reduction 0.3|0.1 for training

## Training
Run `autotrc.py` (set appropriate parameters (hash size, `n_prec`, etc.) first).
- `num_simul`: 800
- `dirichlet noise`: `alpha` 0.05 (~10/225), `fraction` 0.25
- `temperature`: 1 for 30 ply, 0 from 16th ply
- `opening book`: none, 1 random move or 26 canonical opening transforned
- `batch size`: 1024
- `optimizer`: Adam
- `learning rate`: 1e-3 to 1e-5

For pretrained weights, go to [releases](https://github.com/YuanshengZhao/AlphaZero_Gomoku/releases).