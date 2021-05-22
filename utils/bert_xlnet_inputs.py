import torch


def prepare_inputs_for_roberta(raw_in, tokenizer, opt,device):
    '''
    @ input:
    - raw_in: list of strings
    @ output:
    - bert_inputs: padded Tensor
    '''

    bert_inputs = []
    seg_ids = []  
    seg_input_ids = None 
    sequence_a_segment_id=0
    sequence_b_segment_id=1
    #This stores one or two separator token according to 
    seq_separator = tokenizer.sep_token
    for seq in raw_in:    
        tok_seq = []
        tok_seq_a = []
        tok_seq_b = []
        #get index of [USR] word as we would like to get the 
        usr_idx = seq.index("[USR]")
        #Get the previous system response 
        seq_a = seq[2:usr_idx]
        #get user response 
        seq_b = seq[usr_idx+1:]

        if opt.tod_pre_trained_model:
            seq = ["[SYS]"] + seq_a + ["[USR]"] + seq_b
            #seq_a system utterance sequence 
            seq_a = ["[SYS]"] + seq_a
            #seq_b user utterance sequence 
            seq_b = ["[USR]"] + seq_b  
        
        if opt.pre_trained_model and opt.pre_trained_model=="xlm-roberta":
            #update sep token in sequence b having user utterance
            seq_separator = tokenizer.sep_token+tokenizer.sep_token 
            seq_b = [seq_separator if x=="[SEP]" else x for x in seq_b]
        else:
            #update sep token in sequence b having user utterance 
            seq_b = [tokenizer.sep_token if x=="[SEP]" else x for x in seq_b]    
        #tokenize words in seq_a
        #tokenize words in seq_a
        for word in seq_a:
            tok_word = tokenizer.tokenize(word)
            tok_seq_a += tok_word

        #tokenize words in seq_b
        for word in seq_b:
            tok_word = tokenizer.tokenize(word)
            tok_seq_b += tok_word

        if opt.tod_pre_trained_model:
            tok_seq_a = [tokenizer.cls_token] + tok_seq_a 
            #Add [SEP] token to end seq_b 
            tok_seq_b = tok_seq_b + [tokenizer.sep_token]
        
            #create one sequence
            tok_seq = tok_seq_a + tok_seq_b
            bert_inputs.append(tok_seq)
            seq_a_segments = [sequence_a_segment_id] * len(tok_seq_a)
            seq_b_segments = [sequence_b_segment_id] * len(tok_seq_b) 
            seg_ids.append(seq_a_segments + seq_b_segments)   

        #In case of pre-trained model like bert,roberta

        # without system act flag is set 
        elif opt.without_system_act:
            tok_seq = [tokenizer.cls_token] + tok_seq_b + [tokenizer.sep_token]
            bert_inputs.append(tok_seq)

        # this is usual case of [CLS] system_utterance [SEP] user hyp1 [SEP] user hyp2   
        else:           
            #create one sequence
            tok_seq_a = [tokenizer.cls_token] + tok_seq_a 
            #Add [SEP] token to end seq_b 
            tok_seq_b = [seq_separator] + tok_seq_b + [tokenizer.sep_token]            
            #create one sequence
            tok_seq = tok_seq_a + tok_seq_b
            bert_inputs.append(tok_seq)
            seq_a_segments = [sequence_a_segment_id] * len(tok_seq_a)
            seq_b_segments = [sequence_b_segment_id] * len(tok_seq_b) 
            seg_ids.append(seq_a_segments + seq_b_segments)
             
        #Add [SEP] token to end seq_b
        #bert_inputs.append([tokenizer.cls_token] + tok_seq + [tokenizer.sep_token])
        
        
    input_lens = [len(seq) for seq in bert_inputs]
    max_len = max(input_lens)
    bert_input_ids = [tokenizer.convert_tokens_to_ids(seq) + [tokenizer.pad_token_id] * (max_len - len(seq)) 
    for seq in bert_inputs]
    assert len(bert_input_ids[0]) == max_len
    bert_input_ids = torch.tensor(bert_input_ids, dtype=torch.long, device=device)
    

    if seg_ids!=[]:
        seg_input_ids = [seg_id + [0] * (max_len - len(seg_id)) for seg_id in seg_ids]
        assert len(seg_input_ids[0]) == max_len
        seg_input_ids = torch.tensor(seg_input_ids, dtype=torch.long, device=device)

    return bert_input_ids,seg_input_ids,input_lens