# Multi-Seq ASR BERT

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


## 2. Downstream classifier setting 

STC (**S**emantic **T**uple **C**lassifier): We build two classifiers, the first one for `act-slot` pairs and the second one for `value`.
 

Recommended hyper-parameters have been given in the scripts, and you can adjust them according to your needs. 

### 3. Run Script and parameters 

  Train and Evalute the Model

  ```bash
  ./run/train/train_MultiSeq_ASR_BERT_STC.sh
  ```
    
  Parameters: <br />
    -- **pre_trained_model**  `pre-trained model name to use among bert,roberta,xlm-roberta`  <br />
    -- **add_l2_loss**   `Flag used to set usage of MSE loss between asr and transcript hidden state`  <br />
    -- **tod_pre_trained_model** `Path to TOD pre-trained checkpoint Note: This will override pre_trained_model value if passed`  <br />
    -- **add_segment_ids** `Flag to decide to add segment ids`  <br />
    -- **without_system_act** `Flag to remove previous system act [In our case this is previous system utterance]`    

## 4. Results

Results of Multi-Seq ASR BERT and previous SOTA results:

| Model               | F1 score (%) | Acc. (%) |
| ---------------     | ------------ | -------- |
| Multi-Seq ASR BERT  | 87.4         | 81.9     |
| Multi-Seq ASR XLM-R | 87.8         | 81.8     |   

Results can be different due to various environments and hyper-parameter settings.



