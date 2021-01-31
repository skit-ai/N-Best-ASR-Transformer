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


## 2. Downstream classifier setting 

STC (**S**emantic **T**uple **C**lassifier): We build two classifiers, the first one for `act-slot` pairs and the second one for `value`.
 

Recommended hyper-parameters have been given in the scripts, and you can adjust them according to your needs. 

### 3. Run Script and parameters 

  Train and Evalute the Model

  ```bash
  ./run/train/train_TOD_ASR_STC.sh
  ```
    ### Parameters:
    -- pre_trained_model  pre-trained model name to use among bert,roberta,xlm-roberta
    -- add_l2_loss   flag used to set usage of MSE loss between asr and transcript hidden state
    -- tod_pre_trained_model Path to TOD pre-trained checkpoint Note: This will override pre_trained_model value if passed
    -- add_segment_ids Flag to decide to add segment ids
    -- without_system_act Flag to remove previous system act [In our case this is previous system utterance]
    

## 4. Results

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



