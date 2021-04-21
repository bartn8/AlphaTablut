# AlphaTablut

Progetto per [Tablut Students Challenge | AI @ UniBO](http://ai.unibo.it/games/boardgamecompetition/tablut)

## Funzionamento
Il progetto è inspirato dal paper [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815). 

Rispetto ad esso, viene implementato una ricerca basata su MinMax invece che su MCTS, anche se nel paper è specificato che l'utilizzo di MCTS+RESNET è più efficace di MINMAX+RESNET, dati gli errori spuri della RESNET.

Tale modifica comporta anche una sostanziale modifica alla rete neurale utilizzata: è stata alleggerita di parecchio l'architettura (basata su **ResNet**) e si è utilizzato **TensorFlowLite** per fornire un'euristica molto rapida. Purtroppo rimane comunque circa **10 volte** più lenta della controparte _"old school"_

![Network](/images/network.png)

## Requirements

Risulta necessario un ambiente con Python 3.7. Si consiglia l'utilizzo di Anaconda.

## Build

Per prima cosa bisogna creare un nuovo ambiente virtuale: ```conda create -n alphatablut python=3.7```

In seguito si attiva l'ambiente ```conda activate alphatablut``` e si esegue l'installazione con PiP: ```python -m pip install -r requirements.txt```

Infine si effettua la build di Cython: ```python setup.py build_ext --inplace```

## Running

Per eseguire l'addestramento ```python main.py``` e seguire le opzioni del menu.

Per avviare il client ```python client.py -p [W, B] -t 60 -i localhost -v```

* ```-p```: Seleziona il giocatore **W**hite o **B**lack
* ```-t```: Imposta il timeout di ricerca
* ```-i```: Imposta l'host server
* ```-v```: Mostra le informazioni di debug
