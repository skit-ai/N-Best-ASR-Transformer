# WCN-BERT

Jointly encoding word confusion networks (WCNs) and dialogue context with BERT for spoken language understanding (SLU). The [paper](https://arxiv.org/pdf/2005.11640.pdf) is accepted by [INTERSPEECH 2020](http://www.interspeech2020.org/).

## 0. Setup

```bash
pip3 install -r requirements.txt
```

## 1. Data

We conduct our experiments on a benchmark SLU dataset, **DSTC2**. Origin data can be obtained [HERE](http://camdial.org/~mh521/dstc/).

- Data preprocessing:
    ```bash
    python helpers/process_dstc2_data.py \
        --data_dir <input_dir>
        --prun_opt rule \
        --prun_score_thres 1e-3 \
        --bin_norm \
        --rm_null \
        --subdir <output_dir>
    ```
    Note that you should replace the <input_dir> with the original DSTC2 data, and replace the <output_dir> with your own output directory.

The data is preprocessed and saved in `dstc2_data/processed_data/*`, where each line is a data sample in the form of:

```
<system act seq>\t<=>\t<wcn seq>\t<=>\t<labels>
```

The WCNs and system acts are flattened into a sequence as the figure below shows:

![](figs/input.png)

## 2. Experiments

For the experiments with only WCNs, we convert the WCNs into sequences and fed into BERT. For the experiments with WCNs and the last system act, they are flattened into one sequence (see above).

We use both discriminative (NC & STC) and generative (TFHD) approaches for the output module:

- NC (**N**aive **C**lassifier): All `act-slot-value` triplets are classified by one classifier. In this case, there may be multiple values for a certain `act-slot` pair. This method is not presented in the paper.
- STC (**S**emantic **T**uple **C**lassifier): We build two classifiers, the first one for `act-slot` pairs and the second one for `value`.
- TFHD (**T**rans**F**ormer-based **H**ierarchical **D**ecoder): We adopt the method from [this paper](https://arxiv.org/pdf/1904.04498.pdf). The hierarchical model builds classifiers for the `acts` and `slots`, and generate values with a sequence-to-sequence model with pointer network. We make two  changes as follows:
  - Change the backbone model from LSTM to Transformer;
  - Embed the `acts` and `slots` with BERT. 

Recommended hyper-parameters have been given in the scripts, and you can adjust them according to your needs. 

### 2.1 WCN + System Act + NC

- train the model

  ```bash
  ./run/train/train_wcn_NC_SA_BERT.sh
  ```

- test the model

  ```bash
  ./run/test/test_wcn_NC_SA_BERT.sh
  ```

### 2.2 WCN + System Act + STC

- train the model

  ```bash
  ./run/train/train_wcn_STC_SA_BERT.sh
  ```

- test the model

  ```bash
  ./run/test/test_wcn_STC_SA_BERT.sh
  ```

### 2.3 WCN + System Act + TFHD

- train the model

  ```bash
  ./run/train/train_wcn_TFHD_SA_BERT.sh
  ```

- test the model

  ```bash
  ./run/test/test_wcn_TFHD_SA_BERT.sh
  ```

### 2.4 WCN + NC

- train the model

  ```bash
  ./run/train/train_wcn_NC_BERT.sh
  ```

- test the model

  ```bash
  ./run/test/test_wcn_NC_BERT.sh
  ```

### 2.5 WCN + STC

- train the model

  ```bash
  ./run/train/train_wcn_STC_BERT.sh
  ```

- test the model

  ```bash
  ./run/test/test_wcn_STC_BERT.sh
  ```


### 2.6 WCN + TFHD

- train the model

  ```bash
  ./run/train/train_wcn_TFHD_BERT.sh
  ```

- test the model

  ```bash
  ./run/test/test_wcn_TFHD_BERT.sh
  ```

## 3. Results

You are expected to get the following results:

| Model           | with system act | F1 score (%) | Acc. (%) |
| --------------- | --------------- | ------------ | -------- |
| WCN-BERT + STC  | True            | 87.86        | 81.24    |
| WCN-BERT + STC  | False           | 86.71        | 79.68    |
| WCN-BERT + TFHD | True            | 87.37        | 80.77    |
| WCN-BERT + TFHD | False           | 86.15        | 79.12    |

Results can be different due to various environments and hyper-parameter settings.



## 4. Citation

If you use our models, please cite the following papers:

```
@article{liu2020jointly,
  title={Jointly Encoding Word Confusion Network and Dialogue Context with BERT for Spoken Language Understanding},
  author={Liu, Chen and Zhu, Su and Zhao, Zijian and Cao, Ruisheng and Chen, Lu and Yu, Kai},
  journal={arXiv preprint arXiv:2005.11640},
  year={2020}
}
```

