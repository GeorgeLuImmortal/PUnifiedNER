
from transformers import BertTokenizer,T5Tokenizer, T5ForConditionalGeneration,MT5ForConditionalGeneration, Text2TextGenerationPipeline,get_scheduler
import torch, argparse, json, gc, os, random
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib

font = {'size'   : 15}

matplotlib.rc('font', **font)




def read_seqlabel_data(file_json):
    """
    Args:
         json file of lines: [[text,[begin, end, token, category],[],...],...]
         
    Return:
    """
    
    categories = []
    data = []
    
    with open(file_json) as f:
        for line in f:
            line = json.loads(line)
            
            for begin, end, token, category in line[1:]:
                if category not in categories:
                    categories.append(category)
                    
            data.append(line)
    return categories, data

def get_f1_score_label(predictions, gold, label="organization"):
    """
    打分函数
    """
    # pre_lines = [json.loads(line.strip()) for line in open(pre_file) if line.strip()]
    # gold_lines = [json.loads(line.strip()) for line in open(gold_file) if line.strip()]
    TP = 0
    FP = 0
    FN = 0
    for pred, gold in zip(predictions, gold):
        pred = [item[1] for item in pred if item[0] == label]
        gold = [item[1] for item in gold if item[0] == label]
        for i in pred:
            if i in gold:
                TP += 1
            else:
                FP += 1
        for i in gold:
            if i not in pred:
                FN += 1
    if TP != 0:
        p = TP / (TP + FP)
        r = TP / (TP + FN)
        f = 2 * p * r / (p + r)
        print(p, r, f)
        return p,r,f
    else:
        print(0, 0, 0)
        return 0,0,0
    
def get_f1_score(predictions, gold,labels):

    score = {}

    sum = 0
    for idx,label in enumerate(labels):

        p,r,f = get_f1_score_label(predictions, gold, label=label)
        score[label] = {'precision':p,'recall':r,'f1':f}
        
        sum += f
    avg = sum / len(labels)
    return score, avg


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='search for best template according to dev set')
    parser.add_argument('--max_source_length', default=256, type=int, help="max source sequence length")
    parser.add_argument('--max_target_length', default=200, type=int, help="max target sequence length")
    parser.add_argument('--batch_size', default=8, type=int, help="batch size")
    parser.add_argument('--epoch', default=100, type=int, help="training epoches")
    parser.add_argument('--eval_steps', default=1000, type=int, help="eval per steps")
    parser.add_argument('--warm_up_step', default=1000, type=int, help="warm up steps")
    parser.add_argument('--lr', default=1e-5, type=float, help="learning rate")
    parser.add_argument('--model', default='./models/my_t5_base/', type=str, help="pretrained model")
    parser.add_argument('--tokenizer', default='./models/my_t5_base/', type=str, help="tokenizer")
    parser.add_argument('--method', default='1+2', type=str, help="training data construction method")
    # parser.add_argument('--model', default='uer/t5-v1_1-small-chinese-cluecorpussmall', type=str, help="pretrained model")
    # parser.add_argument('--tokenizer', default='uer/t5-v1_1-small-chinese-cluecorpussmall', type=str, help="tokenizer")
    parser.add_argument('--train_dir', default='../cws-dev/dataset/cluener/ml_train.json', type=str, help="training set")
    parser.add_argument('--dev_dir', default='../cws-dev/dataset/cluener/ml_test_all.json', type=str, help="development set")
    parser.add_argument('--save_dir', default='./my_trained_models/', type=str, help="save trained model dir")
    args = parser.parse_args()
    print(args)

    if os.path.exists(args.save_dir):
        pass
    else:
        os.mkdir(args.save_dir)





    train_categories, train_data = read_seqlabel_data(args.train_dir)
    dev_categories, dev_data = read_seqlabel_data(args.dev_dir)





    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    mapping = {'name':'姓名','company':'公司','game':'游戏','organization':'组织','movie':'电影','address':'地址','position':'职位','government':'政府','scene':'景点','book':'书籍'}




    def dataset_construct(data, categories, method='1'):
        
        dataloader = []
        num_categories = len(categories)
        
        for example in data:
            
            ori_text = example[0]
                 
            ## with exact tags
            if '1' in method:
                prefix_tags = []
                target_seq = "("
                source_seq = ''
                
                for item in example[1:]:
                    label = mapping[item[3]]
                    token = item[2]
                    target_seq = target_seq+f"({label}:{token}),"
                    if label not in prefix_tags:
                        prefix_tags.append(label)

                target_seq = target_seq[:-1]+')'

                for tag in prefix_tags:
                    source_seq = source_seq+f"<实体>{tag}"

                source_seq = source_seq+f"<文本>{ori_text}"

                dataloader.append({'input_seq':source_seq,'output_seq':target_seq})
              
            ## 2. with random tags
            if '2' in method:
                
                target_seq = "("
                source_seq = ''
                
                num_tags = random.randint(1, num_categories)
                prefix_tags = list(np.random.choice(categories,num_tags,replace=False))
                
                exist_tags = []
                for item in example[1:]:
                    label = mapping[item[3]]
                    token = item[2]
                    if label in prefix_tags:
                        target_seq = target_seq+f"({label}:{token}),"
                        exist_tags.append(label)
                    else:
                        pass
                
                target_tags = list(set(prefix_tags) - set(exist_tags))
                for label in target_tags:
                     target_seq = target_seq+f"({label}:null),"
                
                target_seq = target_seq[:-1]+')'

                for tag in prefix_tags:
                    source_seq = source_seq+f"<实体>{tag}"

                source_seq = source_seq+f"<文本>{ori_text}"
                
                
                dataloader.append({'input_seq':source_seq,'output_seq':target_seq})

        
        return dataloader



    ## 3. with some tags in an exmaple but are not extracted
    ## 4. mix of 2 and 3





    categories = list(mapping.values())
    print(categories)

    train_dataloader_ori = dataset_construct(train_data,categories,method='1')
    train_dataloader_aug = dataset_construct(train_data,categories,method='2')
    dev_dataloader = dataset_construct(dev_data,categories,method='1')

    print(len(train_dataloader), len(dev_dataloader))





    ## construct train batches
    batches = []

    for idx in range(0,len(train_dataloader),args.batch_size):
        batch = []
        try:
            for index in range(idx, idx+args.batch_size):
                batch.append(train_dataloader[index])
        except Exception:
            pass
        
        batches.append(batch)
        
    eval_batches = []
    for idx in range(0,len(dev_dataloader),args.batch_size):
        batch = []
        try:
            for index in range(idx, idx+args.batch_size):
                batch.append(dev_dataloader[index])
        except Exception:
            pass
        
        eval_batches.append(batch)
      

    ## eval labels for evaluation
    dev_labels = []
    for line in dev_data:
        label = []
        for item in line[1:]:
            label.append([mapping[item[3]],item[2]])
            # label.append((mapping[item[3]],item[2]))
            
        dev_labels.append(label)


    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    model = MT5ForConditionalGeneration.from_pretrained(args.model).to(device)





    optimizer = AdamW(model.parameters(), lr=args.lr)
    # the following 2 hyperparameters are task-specific
    max_source_length = args.max_source_length
    max_target_length = args.max_target_length

    num_training_steps = args.epoch * len(batches)
    print(f'### total number of training steps is {num_training_steps}')
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=args.warm_up_step, num_training_steps=num_training_steps
    )





    losses = [] ## training loss
    eval_losses = [] ## eval loss
    steps = 0 ## num of training step
    avgs = [] ## avg f-score
    x_axis = [] ## x-axis for plotting eval performance

    for epoch in range(args.epoch):

        for batch in tqdm(batches):
            model.train()
            steps += 1
            
            ## set grad to zeros
            optimizer.zero_grad()
            
            input_sequences = []
            output_sequences = []

            for example in batch:

                input_sequences.append(example['input_seq'])
                output_sequences.append(example['output_seq'])


            # encode the inputs
            encoding = tokenizer( input_sequences, padding="longest", max_length=max_source_length, 
                                 truncation=True, return_tensors="pt",)
            input_ids, attention_mask = encoding.input_ids.to(device), encoding.attention_mask.to(device)


            # encode the targets
            target_encoding = tokenizer(
                output_sequences, padding="longest", max_length=max_target_length, truncation=True,
                return_tensors='pt'
            )

            labels = target_encoding.input_ids.to(device)

            # replace padding token id's of the labels by -100
            labels = labels.clone().detach()
            labels[labels == tokenizer.pad_token_id] = -100

            # forward pass
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss

            loss.backward()
            loss = loss.detach().cpu().clone()
            losses.append(loss/args.batch_size)
            
            with open(f"{args.save_dir}training_loss.txt",'a') as fout:
                fout.write(f'{loss.item()}\n')

            plt.plot(losses)
            plt.xlabel('Step')
            plt.savefig(f'{args.save_dir}train_loss.pdf', bbox_inches='tight')
            plt.clf()
            
            # update parameters
            optimizer.step()
            lr_scheduler.step()
            

            del loss, labels, input_ids, attention_mask
            gc.collect()
            torch.cuda.empty_cache()
            
            if steps%args.eval_steps == 0:
                
                print(' ### starting evaluation ###')
                x_axis.append(steps)
                eval_loss = 0.0
                predictions = []
                model.eval()
                
                
                for batch in tqdm(eval_batches):
                    input_sequences = []
                    output_sequences = []

                    for example in batch:
                        input_sequences.append(example['input_seq'])
                        output_sequences.append(example['output_seq'])

                    # encode the inputs
                    encoding = tokenizer(
                        input_sequences,
                        padding="longest",
                        max_length=max_source_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                    input_ids, attention_mask = encoding.input_ids.to(device), encoding.attention_mask.to(device)

                    # encode the targets
                    target_encoding = tokenizer(
                        output_sequences, padding="longest", max_length=max_target_length, truncation=True,
                        return_tensors='pt'
                    )

                    labels = target_encoding.input_ids.to(device)

                    # replace padding token id's of the labels by -100
                    labels = labels.clone().detach()
                    labels[labels == tokenizer.pad_token_id] = -100

                    # compute loss
                    loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
                    eval_loss += loss.detach().cpu().clone()
                    
                    # inference
                    outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask,do_sample=False,eos_token_id=tokenizer.sep_token_id,
                                    decoder_start_token_id=tokenizer.cls_token_id)
                                   
        
                    for pred in tokenizer.batch_decode(outputs, skip_special_tokens=True):
                        pred = ''.join(pred.split(' '))
                        predictions.append(pred)
                        
                   
                    
                    del loss, labels, input_ids, attention_mask
                    gc.collect()
                    torch.cuda.empty_cache()
                
                # plotting eval loss
                eval_losses.append(eval_loss/len(dev_dataloader))
                
                fig, ax = plt.subplots(figsize=(10,6))
                x = x_axis
                y = eval_losses
                ax.set_ylabel('Eval loss')
                ax.set_xlabel('steps')
                ax.plot(x,y)
                plt.savefig(f'{args.save_dir}eval_loss.pdf',bbox_inches='tight')
                plt.clf()
                
                with open(f"{args.save_dir}eval_loss.txt",'a') as fout:
                    fout.write(f"{steps}:{eval_loss/len(dev_dataloader)}\n")
                
                # compute and plotting eval f-score
                
                postprocess_preds = []
                for pred in predictions:
                    pred = pred.split(',')
                    pred = [item.replace(')','').replace('(','').split(':') for item in pred]
    #                 pred = [item.replace(']','').replace('[','').split(':') for item in pred]
                    pred = [item for item in pred if len(item)>1]
                    postprocess_preds.append(pred)

                with open(f"{args.save_dir}{steps}_preds.txt", 'w', encoding='utf8') as fout:
                    for line in postprocess_preds:
                        fout.write(f"{line}\n")
                
                score,avg = get_f1_score(postprocess_preds,dev_labels,categories)
                print(f"{steps}:{avg}")
                avgs.append(avg)
                
                fig, ax = plt.subplots(figsize=(10,6))
                x = x_axis
                y = avgs
                ax.set_ylabel('Eval F-score')
                ax.set_xlabel('steps')
                ax.plot(x,y)
                plt.savefig(f'{args.save_dir}eval_f_score.pdf',bbox_inches='tight')
                plt.clf()
                
                with open(f"{args.save_dir}eval_fscore.txt",'a') as fout:
                    fout.write(f"{steps}:{avg}\n")
                
                
                # saving model checkpoints
                print('### staring saving model ###')
                model.save_pretrained(f'{args.save_dir}{steps}')          





