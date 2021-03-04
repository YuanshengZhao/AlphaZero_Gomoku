# AzGomoku
Alpha (Go) (Zero) algorithm on Gomoku (Five In A Row) game. The code is largely based on the pseudo-code provided by Alpha Zero paper.

[Alpha Go Zero paper](https://www.nature.com/articles/nature24270?sf123103138=1)  
[Alpha Zero paper](https://science.sciencemag.org/content/362/6419/1140)

## Network
Reference to: [Lczero](https://lczero.org/dev/backend/nn/).

- Board: 15x15x2. maps: [stones-self, stones-opponent]. No need for color or history.
- Init: Conv2D, 3x3 64 filters   
- Body: 10 block [SE](https://arxiv.org/abs/1709.01507)-[ResNet](https://arxiv.org/abs/1512.03385) tower, 3x3, 64 filters, 32 SE_channels.  
- Body followed by:
    - P-Head: Conv2D(3,64) -- Conv2D(3,1)   
    - V-Head: Conv2D(3,32) -- Dense(128) --Dense(1)

## Parameters
- WDL encoding: [1/0.5/0]((https://lczero.org/blog/2018/12/alphazero-paper-and-lc0-v0191/)) (not +1/0/-1 stated in the paper)
- `cpuct_base`: 19652
- `cpuct`: 1.25
- `FPU`: reduction 0.3

## Training
Run `autotr.py` (set appropriate parameters (hash size, `n_prec`, etc.) first; took several weeks on PC without GPU).
- `num_simul`: 800
- `dirichlet noise`: `alpha` 0.05 (~10/225), `fraction` 0.25
- `temperature`: 0.5~0.8 for 15 ply, 0 from 16th ply
- `opening book`: 1 to 3 random moves
- `batch size`: 256
- `optimizer`: SGD Nesterov
- `learning rate`: 0.01

At the beginning, to accelerate learning, I used an external engine (weak) to teach the NN (this is actually cheating!).

## Play against the engine
Use `hmplGUI.py`.

Warning: The whole code in this repository is for demonstration of Alpha (Go) Zero, and should be largely optimized for a good performance (e.g. parallel execution, batching the positions for NN evaluation)!!!

Currently, `num_simul = 50` is recommended on CPU (2 sec. thinking time using CPU). Increase this number for stronger playing strenth. 

The engine has certain playing strength, but can blunder in messy positions where both sides have attacking chances, which may be rooted in [a weakness of MCTS](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search#Advantages_and_disadvantages).

## Evaluation of [typical openings](https://en.wikipedia.org/wiki/Renju_opening_pattern)
8720 steps of training; 4K Nodes
### Indirect openings (first two moves are 7 7 and 6 8)
| Name           | Move | Theory | Reply | Score |
| -----------    | ---- | ------ | ----- | ----- |
| 長星 Chosei    | 5 9  | -=     | 6 7   | 0.397 |
| 峡月 Kyogetsu  | 6 9  | 1-0    | 8 9   | 0.362 |
| 恒星 Kosei     | 7 9  | 1-0    | 7 8   | 0.331 |
| 水月 Suigetsu  | 8 9  | 1-0    | 7 9   | 0.329 |
| 流星 Ryusei    | 9 9  | -=     | 8 8   | 0.382 |
| 雲月 Ungetsu   | 7 8  | 1-0    | 7 6   | 0.344 |
| 浦月 Hogetsu   | 8 8  | 1-0    | 6 6   | 0.302 |
| 嵐月 Rangetsu  | 9 8  | 1-0    | 8 7   | 0.318 |
| 銀月 Gingetsu  | 8 7  | +-     | 6 7   | 0.332 |
| 明星 Myojo     | 9 7  | 1-0    | 6 7   | 0.362 |
| 斜月 Shagetsu  | 8 6  | +=     | 8 8   | 0.350 |
| 名月 Meigetsu  | 9 6  | +-     | 6 9   | 0.341 |
| 彗星 Suisei    | 9 5  | 0-1    | 6 9   | 0.507 |

### Direct openings (first two moves are 7 7 and 6 7)
| Name           | Move | Theory | Reply | Score |
| -----------    | ---- | ------ | ----- | ----- |
| 寒星 Kansei    | 5 7  | 1-0    | 6 8   | 0.355 |
| 渓月 Keigetsu  | 5 8  | 1-0    | 6 8   | 0.327 |
| 疎星 Sosei     | 5 9  | =      | 6 8   | 0.399 |
| 花月 Kagetsu   | 6 8  | 1-0    | 8 6   | 0.305 |
| 残月 Zangetsu  | 6 9  | +-     | 7 9   | 0.327 |
| 雨月 Ugetsu    | 7 8  | 1-0    | 7 6   | 0.337 |
| 金星 Kinsei    | 7 9  | 1-0    | 7 8   | 0.326 |
| 松月 Shogetsu  | 8 7  | +-     | 7 8   | 0.332 |
| 丘月 Kyugetsu  | 8 8  | +=     | 6 6   | 0.379 |
| 新月 Shingetsu | 8 9  | +-     | 6 6   | 0.355 |
| 瑞星 Zuisei    | 9 7  | =      | 6 6   | 0.348 |
| 山月 Sangetsu  | 9 8  | +-     | 7 6   | 0.327 |
| 遊星 Yusei     | 9 9  | 0-1    | 6 6   | 0.404 |