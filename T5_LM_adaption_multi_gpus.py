seed = 1
from torch.nn.parallel import DistributedDataParallel
from transformers import set_seed,BertTokenizer,T5Tokenizer, T5ForConditionalGeneration,MT5ForConditionalGeneration, Text2TextGenerationPipeline,get_scheduler
from torch.utils.data import DataLoader
import torch, argparse, json, gc, os, random, math, builtins
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
import torch.distributed as dist
set_seed(1)

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

def prefix_lm_data_construct(data: list) -> list:
    
    dataset = []
    
    for example in tqdm(data):
        
        ori_text = example[0]
        if len(ori_text)!=0:
            random_split = random.randint(0,len(ori_text))
            source_seq = ori_text[:random_split]
            target_seq = ori_text[random_split:]

            dataset.append({'input_seq':source_seq.lower(),'output_seq':target_seq.lower()})

    
    return dataset

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='search for best template according to dev set')
    parser.add_argument('--max_source_length', default=512, type=int, help="max source sequence length")
    parser.add_argument('--max_target_length', default=512, type=int, help="max target sequence length")
    parser.add_argument('--train_batch_size_per_gpu', default=4, type=int, help="train batch size")
    parser.add_argument('--dev_batch_size_per_gpu', default=4, type=int, help="eval batch size")
    parser.add_argument('--epoch', default=100, type=int, help="training epoches")
    parser.add_argument('--eval_steps', default=500, type=int, help="eval per steps")
    parser.add_argument('--lr', default=1e-5, type=float, help="learning rate")
    parser.add_argument('--model', default='./models/my_t5_small/', type=str, help="pretrained model")
    parser.add_argument('--tokenizer', default='./models/my_t5_small/', type=str, help="tokenizer")
    parser.add_argument('--cut_off', default=0.9, type=float, help="cut off of training/dev")
    parser.add_argument('--save_dir', default='./my_trained_models/', type=str, help="save trained model dir")
    parser.add_argument('--data_dir', default='./datasets/', type=str, help="ner data dir")
    args = parser.parse_args()

    args.distributed = True
    args.device ="cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    print(args)

    


    if args.distributed:
        local_rank = int(os.environ['SLURM_LOCALID'])
        args.local_rank=local_rank
        port = "29502"#自己指定0-65535之间
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

    all_data = []

    print('### loading weibo data ###')
    weibo_train_categories, weibo_train_data = read_seqlabel_data(f'{args.data_dir}weibo_train.json')
    weibo_dev_categories, weibo_dev_data = read_seqlabel_data(f'{args.data_dir}weibo_dev.json')
    print(len(weibo_train_data), len(weibo_dev_data))
    all_data.extend(weibo_train_data)
    all_data.extend(weibo_dev_data)

    print('### loading ontonote data ###')
    ontonote_train_categories, ontonote_train_data = read_seqlabel_data(f'{args.data_dir}ontonotes_train.json')
    ontonote_dev_categories, ontonote_dev_data = read_seqlabel_data(f'{args.data_dir}ontonotes_dev.json')
    print(len(ontonote_train_data), len(ontonote_dev_data))
    all_data.extend(ontonote_train_data)
    all_data.extend(ontonote_dev_data)

    print('### loading msra data ###')
    msra_train_categories, msra_train_data = read_seqlabel_data(f'{args.data_dir}MSRA_train.json')
    print(len(msra_train_data))
    all_data.extend(msra_train_data)


    print('### loading resume data ###')
    resume_train_categories, resume_train_data = read_seqlabel_data(f'{args.data_dir}resume_train.json')
    resume_dev_categories, resume_dev_data = read_seqlabel_data(f'{args.data_dir}resume_dev.json')
    print(len(resume_train_data), len(resume_dev_data))
    all_data.extend(resume_train_data)
    all_data.extend(resume_dev_data)

    print('### loading cluener data ###')
    cluener_train_categories, cluener_train_data = read_seqlabel_data(f'{args.data_dir}ml_train.json')
    cluener_dev_categories, cluener_dev_data = read_seqlabel_data(f'{args.data_dir}ml_test_all.json')
    print(len(cluener_train_data), len(cluener_dev_data))
    all_data.extend(cluener_train_data)
    all_data.extend(cluener_dev_data)

    print('### loading people daily data ###')
    pd_train_categories, pd_train_data = read_seqlabel_data(f'{args.data_dir}people_daily.json')
    print(len(pd_train_data))
    ## people daily size is too big, subsample
    pd_train_data = random.sample(pd_train_data, int(len(pd_train_data)/10))
    all_data.extend(pd_train_data)

    print('### loading boson data ###')
    boson_train_categories, boson_train_data = read_seqlabel_data(f'{args.data_dir}boson.json')
    print(len(boson_train_data))
    all_data.extend(boson_train_data)

    print('### loading ecommerce data ###')
    ecommerce_train_categories, ecommerce_train_data = read_seqlabel_data(f'{args.data_dir}ecommerce_train.json')
    ecommerce_dev_categories, ecommerce_dev_data = read_seqlabel_data(f'{args.data_dir}ecommerce_dev.json')
    print(len(ecommerce_train_data), len(ecommerce_dev_data))
    all_data.extend(ecommerce_train_data)
    all_data.extend(ecommerce_dev_data)

    print('### loading chinese_address data ###')
    chinese_address_train_categories, chinese_address_train_data = read_seqlabel_data(f'{args.data_dir}chinese_address_train.json')
    chinese_address_dev_categories, chinese_address_dev_data = read_seqlabel_data(f'{args.data_dir}chinese_address_dev.json')
    print(len(chinese_address_train_data), len(chinese_address_dev_data))
    all_data.extend(chinese_address_train_data)
    all_data.extend(chinese_address_dev_data)

    print(len(all_data))



    sz = len(all_data)
    cut = int(args.cut_off * sz) #cut-off% of the list
    shuffled_l = random.shuffle(all_data)
    train_data = all_data[:cut] # first cut-of% of shuffled list
    dev_data = all_data[cut:] # last cut-of% of shuffled list


    print('### starting prefix LM data split ###')
    train_dataset = prefix_lm_data_construct(train_data)
    dev_dataset = prefix_lm_data_construct(dev_data)


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

    train_dataset, dev_dataset = MyDataset(train_inputs), MyDataset(dev_inputs)

    print(f'### training example nums {len(train_dataset)}, dev example nums {len(dev_dataset)}')

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True) if args.distributed else None
    dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_dataset, shuffle=False) if args.distributed else None
    # dev_sampler = None

    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size_per_gpu, shuffle=(not args.distributed))#如果用分布式训练这里shuffle=False
    dev_loader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=args.dev_batch_size_per_gpu, shuffle=False)

    

    model = MT5ForConditionalGeneration.from_pretrained(args.model).to(args.device)

    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)



    optimizer = AdamW(model.parameters(), lr=args.lr)
    # the following 2 hyperparameters are task-specific
    max_source_length = args.max_source_length
    max_target_length = args.max_target_length

    num_training_steps = args.epoch * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=int(num_training_steps*0.06), num_training_steps=num_training_steps
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

            with autocast():
            # forward pass
                loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
            # loss.backward()

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
            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            

            del loss, labels, input_ids, attention_mask
            gc.collect()
            torch.cuda.empty_cache()
            
            if steps%args.eval_steps == 0: # only val and save on master node
  
                print(' ### starting evaluation ###')
                x_axis.append(steps)
                eval_loss = 0.0
                predictions = []
                model.eval()
                
                
                for batch in tqdm(dev_loader):
                    
                    input_ids, attention_mask = batch['input_ids'].to(args.device), batch['attention_mask'].to(args.device)
                    labels = batch['labels'].to(args.device)

                    # replace padding token id's of the labels by -100
                    labels = labels.clone().detach()
                    labels[labels == tokenizer.pad_token_id] = -100

                    # compute loss
                    ## eval does not need to calculate graduate
                    with torch.no_grad():
                        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
                     
                    eval_loss += loss.detach().cpu().clone()
                    
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
                
                    # saving model checkpoints
                    print('### staring saving model ###')
                    model.module.save_pretrained(f'{args.save_dir}{steps}')  

    print('### Training finished ###')    

