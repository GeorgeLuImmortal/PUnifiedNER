seed  = 1
from torch.nn.parallel import DistributedDataParallel
from transformers import set_seed, BertTokenizer,T5Tokenizer, T5ForConditionalGeneration,MT5ForConditionalGeneration, Text2TextGenerationPipeline,get_scheduler
from torch.utils.data import DataLoader
import torch, argparse, json, gc, os, random, math, builtins, ast
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
import torch.distributed as dist
set_seed(seed)

print(f'### num of gpus detect is {torch.cuda.device_count()} ###')
font = {'size'   : 15}
matplotlib.rc('font', **font)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: val[idx].detach().clone() for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)

def read_seqlabel_data(file_json:str) -> (list,list):
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

def get_f1_score_label(predictions:list, gold:list, label:str="organization") -> (float,float,float):
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
    
def get_f1_score(predictions:list, gold:list,labels:list) -> (dict, float):

    score = {}

    sum = 0
    for idx,label in enumerate(labels):

        p,r,f = get_f1_score_label(predictions, gold, label=label)
        score[label] = {'precision':p,'recall':r,'f1':f}
        
        sum += f
    avg = sum / len(labels)
    return score, avg

def dataset_construct(data:list, categories:list, method:str ='1') -> list:
        
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

                dataloader.append({'input_seq':source_seq.lower(),'output_seq':target_seq.lower()})
              
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
                
                
                dataloader.append({'input_seq':source_seq.lower(),'output_seq':target_seq.lower()})

        
        return dataloader

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='search for best template according to dev set')
    parser.add_argument('--max_source_length', default=256, type=int, help="max source sequence length")
    parser.add_argument('--max_target_length', default=200, type=int, help="max target sequence length")
    parser.add_argument('--train_batch_size_per_gpu', default=4, type=int, help="train batch size")
    parser.add_argument('--dev_batch_size_per_gpu', default=4, type=int, help="eval batch size")
    parser.add_argument('--epoch', default=100, type=int, help="training epoches")
    parser.add_argument('--eval_steps', default=1000, type=int, help="eval per steps")
    parser.add_argument('--warm_up_step', default=1000, type=int, help="warm up steps")
    parser.add_argument('--lr', default=1e-5, type=float, help="learning rate")
    parser.add_argument('--model', default='./models/my_t5_base/', type=str, help="pretrained model")
    parser.add_argument('--tokenizer', default='./models/my_t5_base/', type=str, help="tokenizer")
    parser.add_argument('--method', default='1+2', type=str, help="training data construction method")
    parser.add_argument('--do_eval', default=True, help="whether do evaluation")
    # parser.add_argument('--model', default='uer/t5-v1_1-small-chinese-cluecorpussmall', type=str, help="pretrained model")
    # parser.add_argument('--tokenizer', default='uer/t5-v1_1-small-chinese-cluecorpussmall', type=str, help="tokenizer")
    parser.add_argument('--train_dir', default='../cws-dev/dataset/cluener/ml_train.json', type=str, help="training set")
    parser.add_argument('--dev_dir', default='../cws-dev/dataset/cluener/ml_test_all.json', type=str, help="development set")
    parser.add_argument('--save_dir', default='./my_trained_models/', type=str, help="save trained model dir")
    args = parser.parse_args()

    args.distributed = True
    args.device ="cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    print(args)

    

    if args.distributed:
        local_rank = int(os.environ['SLURM_LOCALID'])
        args.local_rank=local_rank
        port = "29501"#自己指定0-65535之间
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']


        if '[' in node_list:
            beg = node_list.find('[')
            pos1 = node_list.find('-', beg)
            if pos1 < 0:
                pos1 = 1000
            pos2 = node_list.find(',', beg)
            if pos2 < 0:
                pos2 = 1000
            node_list = node_list[:min(pos1, pos2)].replace('[', '')

        addr = node_list[8:].replace('-', '.')
        os.environ['MASTER_PORT'] = port
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(args.local_rank)

        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])


        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        args.dist_backend = 'nccl'
        host_addr_full = 'tcp://' + addr + ':' + port
        torch.distributed.init_process_group(backend=args.dist_backend, init_method=host_addr_full,world_size=args.world_size, rank=args.rank)#


    # suppress printing if not on master gpu
    if args.rank!=0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    ## create directory only on master node
    if args.rank == 0:
        print(os.path.exists(args.save_dir))
        if os.path.exists(args.save_dir):
            pass
        else:
            os.mkdir(args.save_dir)

    train_categories, train_data = read_seqlabel_data(args.train_dir)
    dev_categories, dev_data = read_seqlabel_data(args.dev_dir)



    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    mapping = {'name':'姓名','company':'公司','game':'游戏','organization':'组织','movie':'电影','address':'地址','position':'职位','government':'政府','scene':'景点','book':'书籍'}


    ## valid labels for computing f score
    valid_labels = [] ## normalized ground truths
    for line in dev_data:
        norm_label = []
        for item in line[1:]:
            new_item = [mapping[item[3]],item[2].lower()]
            if new_item not in norm_label:
                norm_label.append(new_item)
        
        valid_labels.append(norm_label)

    categories = list(mapping.values())
    print(categories)

    train_dataset = dataset_construct(train_data,categories,method=args.method)
    dev_dataset = dataset_construct(dev_data,categories,method='1')
    

    print('### starting encoding text ###')
    ## encoding training/valid inputs/outputs
    train_inputs, train_labels = [line['input_seq'] for line in train_dataset], [line['output_seq'] for line in train_dataset]
    dev_inputs, dev_labels = [line['input_seq'] for line in dev_dataset], [line['output_seq'] for line in dev_dataset]

    train_inputs = tokenizer(train_inputs, padding="longest", max_length=args.max_source_length, truncation=True, return_tensors="pt")
    train_labels = tokenizer(train_labels, padding="longest", max_length=args.max_target_length, truncation=True,return_tensors='pt')
    train_inputs['labels'] = train_labels.input_ids.detach().clone()

    dev_inputs = tokenizer(dev_inputs, padding="longest", max_length=args.max_source_length, truncation=True, return_tensors="pt")
    dev_labels = tokenizer(dev_labels, padding="longest", max_length=args.max_target_length, truncation=True,return_tensors='pt')
    dev_inputs['labels'] = dev_labels.input_ids.detach().clone()
    dev_inputs['sample_id'] = torch.tensor([i for i in range(len(dev_dataset))])

    train_dataset, dev_dataset = MyDataset(train_inputs), MyDataset(dev_inputs)

    print(f'### training example nums {len(train_dataset)}, dev example nums {len(dev_dataset)}')

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True) if args.distributed else None
    dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_dataset, shuffle=False) if args.distributed else None
    # dev_sampler = None

    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size_per_gpu, shuffle=(not args.distributed))#如果用分布式训练这里shuffle=False
    dev_loader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=args.dev_batch_size_per_gpu, shuffle=False)
    

    model = MT5ForConditionalGeneration.from_pretrained(args.model).to(args.device)
    model.config.max_length = 512

    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    

    optimizer = AdamW(model.parameters(), lr=args.lr)
    # the following 2 hyperparameters are task-specific
    max_source_length = args.max_source_length
    max_target_length = args.max_target_length

    num_training_steps = args.epoch * len(train_loader)
    print(f'### total number of training steps is {num_training_steps}')
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=args.warm_up_step, num_training_steps=num_training_steps
    )
    # Creates a GradScaler once at the beginning of training.
    scaler = GradScaler()

    print(f'## total number of training steps {num_training_steps} ##')





    losses = [] ## training loss
    eval_losses = [] ## eval loss
    steps = 0 ## num of training step
    avgs = [] ## avg f-score
    x_axis = [] ## x-axis for plotting eval performance

    for epoch in range(args.epoch):

        set_seed(epoch)
        # fix sampling seed such that each gpu gets different part of dataset
        if args.distributed: 
            train_loader.sampler.set_epoch(epoch)

        for batch in tqdm(train_loader):
            model.train()
            steps += 1
            
            ## set grad to zeros
            optimizer.zero_grad()
            
            
            input_ids, attention_mask = batch['input_ids'].to(args.device), batch['attention_mask'].to(args.device)
            labels = batch['labels'].to(args.device)

            # replace padding token id's of the labels by -100
            labels = labels.clone().detach()
            labels[labels == tokenizer.pad_token_id] = -100

            # forward pass
            with autocast():
                loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss

            scaler.scale(loss).backward()
       
            loss = loss.detach().cpu().clone()

            ## write loss of each rank to local
            with open(f"{args.save_dir}{args.rank}_batch_loss.txt",'w') as fout:
                fout.write(f'{loss.item()}')

            dist.barrier()

            if args.rank == 0:
                batch_loss = 0.0 ## total loss of all gpus
                ## read loss of each rank from local
                for i in range(ntasks):
                    i = str(i)
                    with open(f"{args.save_dir}{i}_batch_loss.txt",'r') as fin:
                        rank_loss = float(fin.readlines()[0])
                        batch_loss+=rank_loss

                losses.append(batch_loss)
            
                with open(f"{args.save_dir}training_loss.txt",'a') as fout:
                    fout.write(f'{steps}:{batch_loss}\n')

                plt.plot(losses)
                plt.xlabel('Step')
                plt.savefig(f'{args.save_dir}train_loss.pdf', bbox_inches='tight')
                plt.clf()
            
            # update parameters
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            

            del loss, labels, input_ids, attention_mask
            gc.collect()
            torch.cuda.empty_cache()
            
            if steps%args.eval_steps == 0 and args.do_eval:

                print(' ### starting evaluation ###')
                x_axis.append(steps)
                eval_loss = 0.0
                predictions = [] ## predictions of this sampler
                sampler_ids = [] ## index of this sampler
                model.eval()
                
                
                for batch in tqdm(dev_loader):

                    sampler_ids.extend(batch['sample_id'].tolist())

                    input_ids, attention_mask = batch['input_ids'].to(args.device), batch['attention_mask'].to(args.device)
                    labels = batch['labels'].to(args.device)

                    # replace padding token id's of the labels by -100
                    labels = labels.clone().detach()
                    labels[labels == tokenizer.pad_token_id] = -100

                    # compute loss
                    with torch.no_grad():
                        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss

                    eval_loss += loss.detach().cpu().clone()
                    
                    # inference
                    outputs = model.module.generate(input_ids=input_ids,num_beams=10, max_length=512, eos_token_id=tokenizer.sep_token_id)
                                   
        
                    for pred in tokenizer.batch_decode(outputs, skip_special_tokens=True):
                        pred = ''.join(pred.split(' '))
                        predictions.append(pred)
                        
                   
                    
                    del loss, labels, input_ids, attention_mask
                    gc.collect()
                    torch.cuda.empty_cache()

               ## write loss of each rank to local
                with open(f"{args.save_dir}{args.rank}_eval_batch_loss.txt",'w') as fout:
                    fout.write(f'{eval_loss}')

                dist.barrier()
                # plotting eval loss
                if args.rank == 0:
                    batch_loss = 0.0
                    ## read loss of each rank from local
                    for i in range(ntasks):
                        i = str(i)
                        with open(f"{args.save_dir}{i}_eval_batch_loss.txt",'r') as fin:
                            rank_loss = float(fin.readlines()[0])
                            batch_loss+=rank_loss

                    eval_losses.append(batch_loss)
                    
                    fig, ax = plt.subplots(figsize=(10,6))
                    x = x_axis
                    y = eval_losses
                    ax.set_ylabel('Eval loss')
                    ax.set_xlabel('steps')
                    ax.plot(x,y)
                    plt.savefig(f'{args.save_dir}eval_loss.pdf',bbox_inches='tight')
                    plt.clf()
                    
                    with open(f"{args.save_dir}eval_loss.txt",'a') as fout:
                        fout.write(f"{steps}:{batch_loss}\n")
                
                # compute and plotting eval f-score
                
                postprocess_preds = []
                for pred in predictions:
                    pred = pred.split(',')
                    pred = [item.replace(')','').replace('(','').split(':') for item in pred]
    #                 pred = [item.replace(']','').replace('[','').split(':') for item in pred]

                    ## normalize predictions
                    new_pred = []
                    for item in pred:
                        if item not in new_pred and len(item)>1 and item[1]!='null':
                            new_pred.append(item)
                    # pred = [item for item in pred if len(item)>1]
                    postprocess_preds.append(new_pred)

                with open(f"{args.save_dir}{args.rank}_{steps}_preds.txt", 'w', encoding='utf8') as fout:
                    for line in postprocess_preds:
                        fout.write(f"{line}\n")

                sampler_labels = []
                for idx,line in enumerate(valid_labels):
                    if idx in sampler_ids:
                        sampler_labels.append(line)


                with open(f"{args.save_dir}{args.rank}_{steps}_labels.txt", 'w', encoding='utf8') as fout:
                    for line in sampler_labels:
                        fout.write(f"{line}\n")



                dist.barrier()

                if args.rank == 0:

                    postprocess_preds = []
                    gold_labels = []

                    for i in range(ntasks):
                        i = str(i)
                        with open(f"{args.save_dir}{i}_{steps}_preds.txt",'r') as fin:
                            for line in fin.readlines():
                                postprocess_preds.append(ast.literal_eval(line))

                    for i in range(ntasks):
                        i = str(i)
                        with open(f"{args.save_dir}{i}_{steps}_labels.txt",'r') as fin:
                            for line in fin.readlines():
                                gold_labels.append(ast.literal_eval(line))

                    with open(f"{args.save_dir}{steps}_preds.txt", 'w', encoding='utf8') as fout:
                        for line in postprocess_preds:
                            fout.write(f"{line}\n")

                    with open(f"{args.save_dir}gold_labels.txt", 'w', encoding='utf8') as fout:
                        for line in gold_labels:
                            fout.write(f"{line}\n")
                
                    score,avg = get_f1_score(postprocess_preds,gold_labels,categories)
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
                    model.module.save_pretrained(f'{args.save_dir}{steps}')          





