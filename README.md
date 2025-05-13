## LexiSem
 
This is the implementation of the paper:LexiSem: A Re-Ranker Balancing Lexical and Semantic Quality for Enhanced Abstractive Summarization


[**LexiSem/paper_under_review**](https://).


## Abstract
Sequence-to-sequence neural networks have recently achieved significant success in abstractive summarization, especially through fine-tuning large pre-trained language models on downstream datasets. However, these models frequently suffer from exposure bias, which can impair their performance. To address this, re-ranking systems have been introduced, but their potential remains underexplored despite some demonstrated performance gains. Most prior work relies on ROUGE scores and aligned candidate summaries for ranking, exposing a substantial gap between semantic similarity and lexical overlap metrics. In this study, we demonstrate that a second-stage model can be trained to re-rank a set of summary candidates, significantly enhancing performance. Our novel approach leverages a re-ranker that balance lexical and semantic quality. Additionally, we introduce a new strategy for defining negative samples in ranking models. Through experiments on the CNN/DailyMail, XSum and Reddit TIFU  datasets, we show that our method effectively estimates the semantic content of summaries without compromising lexical quality. In particular, our method sets a new performance benchmark on the CNN/DailyMail dataset (48.18 R1, 24.46 R2, 45.05 RL) and on Reddit TIFU (30.37 R1,RL 23.87) .

## Architecture


## Dependency
```console
pip install -r requirements.txt
```



## Dataset
Obtain the original CNN dataset at [**this link**](https://github.com/abisee/cnn-dailymail).

## Step1: Segmentation 

CNN dataset has been preprocessed by using the model "SegmenT5-large". This setup encompasses.......

