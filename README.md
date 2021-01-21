# TOD-ASR-BERT

This project aims to build a new SOTA model on the DSTC2 dataset 

## 0. Setup

```bash
pip3 install -r requirements.txt
```

## 1. Data

We conduct our experiments on a benchmark SLU dataset, **DSTC2**. Origin data can be obtained [HERE](http://camdial.org/~mh521/dstc/).

- Data preprocessing:
    ```bash
    python helpers/process_dstc2_with_SEP.py \
        --data_dir <input_dir> \
        --out_dir <output_dir>
    ```
    Note that you should replace the <input_dir> with the original DSTC2 data, and replace the <output_dir> with your own output directory.

The data is preprocessed and saved in `dstc2_data/processed_data/*`, where each line is a data sample in the form of:

```
[SYS] Hello you are here to book a table [USR] yes want a to book a table [SEP] yes book table 
```


## 2. Experiments

We use both discriminative (NC & STC) and generative (TFHD) approaches for the output module:

- NC (**N**aive **C**lassifier): All `act-slot-value` triplets are classified by one classifier. In this case, there may be multiple values for a certain `act-slot` pair. This method is not presented in the paper.
- STC (**S**emantic **T**uple **C**lassifier): We build two classifiers, the first one for `act-slot` pairs and the second one for `value`.
- TFHD (**T**rans**F**ormer-based **H**ierarchical **D**ecoder): We adopt the method from [this paper](https://arxiv.org/pdf/1904.04498.pdf). The hierarchical model builds classifiers for the `acts` and `slots`, and generate values with a sequence-to-sequence model with pointer network. We make two  changes as follows:
  - Change the backbone model from LSTM to Transformer;
  - Embed the `acts` and `slots` with BERT. 

Recommended hyper-parameters have been given in the scripts, and you can adjust them according to your needs. 

### 2.1 ASR-Hypothesis + System Act + STS

- train the model

  ```bash
  ./run/train/train_TOD_ASR_STC.sh
  ```


## 3. Results

Results of TOD-ASR-BERT and previous SOTA results:

| Model              | with system act | F1 score (%) | Acc. (%) |
| ---------------    | --------------- | ------------ | -------- |
| WCN-BERT + STC     | True            | 87.86        | 81.24    |
| WCN-BERT + STC     | False           | 86.71        | 79.68    |
| WCN-BERT + TFHD    | True            | 87.37        | 80.77    |
| WCN-BERT + TFHD    | False           | 86.15        | 79.12    |
| TOD-ASR-BERT + STC | True            | 87.4         | 81.8     |
| TOD-WCN-BERT + STC | True            | 88.6         | 81.x     |   

Results can be different due to various environments and hyper-parameter settings.



