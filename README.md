# BERT Regressor

Implementation of BERT for sequence classfication with Hugging face's transformers (Wolf et al, 2019). This model has great result in various tasks such as natural nangauage inference(Devlin et al, 2018), machine translation evaluation (Shimanaka et al, 2019). 

<p align="center">
<img src=https://user-images.githubusercontent.com/53220859/71612811-1b607480-2be6-11ea-9f3b-ca890eb39d76.png width=500pt>
</p>



## Installation

This code are depend on python >= 3.6.5.

```sh
git clone https://github.com/t080/bert-regressor.git
cd ./bert-regressor
pip install -r requirements.txt
```



## Usage

### Fine-tuning

Fine-tuning pre-trained model on your supervised data.  If you want to do classification task, please set `--num_labels > 1` (default: 1). When `num_labels > 1`,  cross-entropy loss is computed (default: mean-squre loss).

```shell
mkdir ./checkpoints # directory to save fine-tuned model
python run_regression.py \
  --do_train \
  --data ./sample_data/sample_train.tsv \
  --save_dir ./checkpoints \
  --model_type bert \
  --model_name_or_path bert-base-multilingual-cased \
  --num_labels 1 \
  --max_epoch 3 \
  --gpu
```



### Evaluation

Evaluation of fine-tuned model by F1 score.

```shell
python run_regression.py \
  --do_eval \
  --data ./sample_data/sample_eval.tsv \
  --model_type bert \
  --model_name_or_path ./checkpoints \
  --num_labels 1 \
  --threshold 0.5
```



### Test

Output score.

```shell
python run_regression.py \
  --do_test \
  --data ./sample_data/sample_test.tsv \
  --model_type bert \
  --model_name_or_path ./checkpoints \
  --num_labels 1
```



## Reference

- Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", arXiv:1810.04805 (2018)
- Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz† , Jamie Brew, "Transformers: State-of-the-art Natural Language Processing", arXiv:1910.03771v3 (2019)
- Hiroki Shimanaka, Tomoyuki Kajiwara, shimanaka hiroki, "Machine Translation Evaluation with BERT Regressor", arXiv:1907.12679 (2019)

