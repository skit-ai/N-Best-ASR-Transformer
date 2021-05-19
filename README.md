# Multi-Seq ASR BERT

This repository contains code for Multi-Seq ASR BERT: A simplified approach to model ASR confusion in SLU. The paper has been accepted to ACL-IJCNLP 2021.

## Introduction
### About Multi-Seq ASR BERT
Transformer models have achieved state-of-the-art generalization performance on various language understanding tasks but using them on raw Automatic Speech Recognition (ASR) output is sub-optimal because of transcription errors. Common approaches to mitigate this involve using richer output from ASR either in the form of transcription lattice or n-best hypotheses. Using lattices usually gives better performance at the cost of modifications in the architecture of models since they are designed to take plain text input. In our work, we use concatenated n-best ASR hypotheses as the input to the transformer encoder models like BERT. We show that this approach performs as well as the state-of-the-art approach on DSTC2 dataset. Since the input is closer in structure to text based transformers, our approach outperforms state-of-the-art WCN model in low data regimes. Additionally, since popular ASR APIs do not provide lattice level access, this simplification helps us to keep the downstream model relatively independent.  

### Architecture

[![arch-1.png](https://i.postimg.cc/bwds3pR9/arch-1.png)](https://postimg.cc/RW5S0ryW)

### About Data

## 1. Data

We conduct our experiments on a benchmark SLU dataset which ASR alternatives, **DSTC2**. Origin data can be obtained [here](http://camdial.org/~mh521/dstc/).

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

## Experiment

We build our experiments keeping [this](https://github.com/simplc/WCN-BERT) repository as our base code. 

The data preprocessing step mentioned above already converts the data into the desired input format. We use discriminative approaches for the output module. 

All act-slot-value triplets are classified by one classifier. In this case, there may be multiple values for a certain act-slot pair. This method is not presented in the paper.

### Downstream classifier setting 

We build two classifiers, the first one for `act-slot` pairs and the second one for `value`.

## Training Script and Parameters:

  Run command:

  ```bash
  ./run/train/train_MultiSeq_ASR_BERT_STC.sh
  ```
    
  Parameters: <br />

    -- **pre_trained_model** : pre-trained model name to use among `"bert"`,`"roberta"`,`"xlm-roberta"`  <br />

    -- **add_l2_loss**: Flag used to set usage of MSE loss between asr and transcript hidden state  <br />

    -- **tod_pre_trained_model**: Path to TOD pre-trained checkpoint Note: This will override pre_trained_model value if passed. <br>

    -- **add_segment_ids** : Flag to decide to add segment ids  <br>

    -- **without_system_act**: Flag to remove previous system act [In our case this is previous system utterance]    <br />

 Parameters to perform Sample Complexity related Experiments:  <br />

   -- **coverage**: Based on coverage percentage stratified data samples will be picked as a training set. Coverage = (0,1], where, coverage = 1 means you are including the whole data set for training, and, coverage < 1 refers to the percentage of samples you want to consider for training your model. For our work we test our model for sample complexity coverage of {0.05, 0.10, 0.20, 0.50}.  <br /> 
   
   -- **upsample_count**: Upsamples data set by X times. X is a real number.  <br />


## 4. Results

Results of Multi-Seq ASR BERT and previous SOTA results:

| Model               | F1 score (%) | Acc. (%) |
| ---------------     | ------------ | -------- |
| Multi-Seq ASR BERT  | 87.4         | 81.9     |
| Multi-Seq ASR XLM-R | 87.8         | 81.8     |   

Results are average after 5 runs on the dataset, each having a unique random seed.


## Citation

If you use our work, please cite our work as follows:

```

```
