# BERT-KG

Source code for "*Enriching Language Models Representations via Knowledge Graphs Regularisation*" paper (ESANN 2022, pending approval).

#### Astract
In this paper, we propose a novel method for augmenting the representations learned by Transformer-based language models with the symbolic information contained into knowledge graphs. We first compute the node embeddings of a knowledge graph via a deep graph network. We then add a new regularisation term to the loss of BERT that encourages the learned word embeddings to be similar to the node embeddings. We test our method on the challenging WordNet and Freebase knowledge graphs.
The results show that the regularised embeddings perform better than standard embeddings on the chosen probing tasks.

**Authors**: Matteo Medioli, Andrea Valenti, Davide Bacciu

**Credits**: Deepak Nathani, [*Learning Attention-based Embeddings for Relation Prediction in Knowledge Graphs*](https://github.com/deepakn97/relationPrediction)

## Requirements
```
pip install -r requirements.txt
```

## Data Preprocessing
```
sh setup.sh
```

## Learn Knowledge Graph Embeddings

### WN18RR
```
python kbgat/run_kbgat.py --get_2hop True --data kbgat/data/WN18RR/
```

### FB15K-237
```
python main.py --epochs_gat 3000 --epochs_conv 200 --weight_decay_gat 0.00001 --get_2hop True --partial_2hop True --batch_size_gat 272115 --margin 1 --out_channels 50 --drop_conv 0.3 --weight_decay_conv 0.000001 --output_folder ./checkpoints/fb/out/ --data kbgat/data/FB15k-237/
```

## BERT Training - Masked Language Modeling
```
sh train_bert.sh
```

## Probing Tasks
```
sh probe.sh
```
