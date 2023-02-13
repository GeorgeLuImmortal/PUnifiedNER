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
    
    with open(file_json, encoding='utf8') as f:
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

def dataset_construct(mapping:dict, data:list, categories:list, max_entities: int, method:str ='1') -> list:
        
    dataloader = []
    # num_categories = len(categories)
    
    for example in data:
        
        ori_text = example[0]
             
        ## with exact tags
        if '1' in method:
            prefix_tags = []
            target_seq = "("
            source_seq = ''
            
            for item in example[1:]:
                # if item[3]!='o' and item[3]!='product_name':
                if item[3]!='o':
                    label = mapping[item[3]]
                    token = item[2]
                    target_seq = target_seq+f"({label}:{token}),"
                    if label not in prefix_tags:
                        prefix_tags.append(label)

        
            if len(target_seq)==1:
                target_seq = '()'
            else:
                target_seq = target_seq[:-1]+')'

            for tag in prefix_tags:
                source_seq = source_seq+f"<实体>{tag}"

            source_seq = source_seq+f"<文本>{ori_text}"

            dataloader.append({'input_seq':source_seq.lower(),'output_seq':target_seq.lower()})
          
        ## 2. with random tags
        if '2' in method:
            
            target_seq = "("
            source_seq = ''
            
            num_tags = random.randint(0, max_entities)
            prefix_tags = list(np.random.choice(categories,num_tags,replace=False))## select random tags from the pool
            
            exist_tags = []
            for item in example[1:]:
                # if item[3]!='o' and item[3]!='product_name':
                if item[3]!='o':
                    label = mapping[item[3]]
                    token = item[2]
                    if label in prefix_tags:
                        target_seq = target_seq+f"({label}:{token})," ## for hit tags add ground truth
                        exist_tags.append(label)
                    else:
                        pass
            
            ## for excluded tags add null
            target_tags = list(set(prefix_tags) - set(exist_tags))
            for label in target_tags:
                 target_seq = target_seq+f"({label}:null),"
            
            ## if not any tags
            if len(target_seq)==1:
                target_seq = '()'
            else:
                target_seq = target_seq[:-1]+')'
            

            for tag in prefix_tags:
                source_seq = source_seq+f"<实体>{tag}"

            source_seq = source_seq+f"<文本>{ori_text}"
            
            
            dataloader.append({'input_seq':source_seq.lower(),'output_seq':target_seq.lower()})

        ## 3. with all tags
        if '3' in method:

            target_seq = "("
            source_seq = ''

    
            prefix_tags = categories ## select all tags

            
            exist_tags = []
            for item in example[1:]:

                if item[3]=='GPE':
                    label = mapping[item[3]]
                    token = item[2]
                    if label in prefix_tags:
                        target_seq = target_seq+f"({label}:{token})," ## for hit tags add ground truth
                        exist_tags.append(label)
                    else:
                        pass

                    ##GPE is GPE as well as location
                    label = '地点'
                    token = item[2]
                    if label in prefix_tags:
                        target_seq = target_seq+f"({label}:{token})," ## for hit tags add ground truth
                        exist_tags.append(label)
                    else:
                        pass


                # elif item[3]!='o' and item[3]!='product_name':
                elif item[3]!='o':
                    label = mapping[item[3]]
                    token = item[2]
                    if label in prefix_tags:
                        target_seq = target_seq+f"({label}:{token})," ## for hit tags add ground truth
                        exist_tags.append(label)
                    else:
                        pass


            ## for excluded tags add null
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
    parser.add_argument('--steps', default=100000, type=int, help="training steps")
    parser.add_argument('--start_step', default=0, type=int, help="starting steps for resuming training")
    parser.add_argument('--eval_steps', default=1000, type=int, help="eval per steps")
    parser.add_argument('--warm_up_step', default=1000, type=int, help="warm up steps")
    parser.add_argument('--lr', default=1e-5, type=float, help="learning rate")
    parser.add_argument('--model', default='./models/my_t5_base/', type=str, help="pretrained model")
    parser.add_argument('--tokenizer', default='./models/my_t5_base/', type=str, help="tokenizer")
    parser.add_argument('--method', default='1+2', type=str, help="training data construction method")
    parser.add_argument('--do_eval', default=True, help="whether do evaluation")
    parser.add_argument('--data_dir', default='./ner_datasets/', type=str, help="ner data dir")
    parser.add_argument('--save_dir', default='./my_trained_models/', type=str, help="save trained model dir")
    parser.add_argument('--num_gpus', default=16, type=int, help="num gpus used")
    parser.add_argument('--beam_width', default=5, type=int, help="beam search width")
    parser.add_argument('--decode_max_len', default=512, type=int, help="valid decoder max length")
    parser.add_argument('--model_max_len', default=512, type=int, help="model max length")
    parser.add_argument('--max_entities', default=20, type=int, help="max num entities prepended")
    parser.add_argument('--random_seed', default=20, type=int, help="random seed")
    args = parser.parse_args()

    args.distributed = True
    args.device ="cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    print(args)

    

    if args.distributed:
        local_rank = int(os.environ['SLURM_LOCALID'])
        args.local_rank=local_rank
        set_seed(args.random_seed)
        # port = "29512"#自己指定0-65535之间
        port = str(random.randint(0,65534))
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


    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  

    ## for evaluation
    ## entities of each dataset
    all_datasets_mapping = {
                            # 0:{'PER.NAM':'名称','PER.NOM':'人称泛指','GPE.NAM':'地缘政治实体','GPE.NOM':'地缘政治实体泛指','LOC.NOM':'地点泛指','LOC.NAM':'地点','ORG.NOM':'组织泛指','ORG.NAM':'组织'},
                            1:{'HP':'品牌','HC':'商品'},
                            2:{'LOC':'地点','PER':'名称','ORG':'组织'},
                            3:{'GPE':'地缘政治实体','LOC':'地点','PER':'名称','ORG':'组织'},
                            4:{'NAME':'名称', 'CONT':'国籍', 'RACE':'民族', 'TITLE':'职位', 'EDU':'学历', 'ORG':'公司', 'PRO':'专业', 'LOC':'籍贯'},
                            5:{'name':'名称','company':'公司','game':'游戏','organization':'组织','movie':'电影', 'address':'地点','position':'职位','government':'政府','scene':'景点','book':'书籍'},
                            # 6:{'product_name':'产品', 'time':'时间', 'person_name':'名称', 'org_name':'组织', 'location':'地点', 'company_name':'公司'},
                            6:{'product_name':'产品','time':'时间', 'person_name':'名称', 'org_name':'组织', 'location':'地点', 'company_name':'公司'},
                            7:{'LOC':'地点','PER':'名称','ORG':'组织','T':'时间'},
                            8:{'prov':'省份', 'city':'城市', 'district':'区', 'town':'街道', 'community':'社区', 'poi':'兴趣点', 'road':'路', 'roadno':'路号',\
           'subpoi':'次兴趣点', 'devzone':'产业园', 'houseno':'楼号', 'intersection':'路口', 'assist':'方位', 'cellno':'单元', 'floorno':'楼层', 'distance':'距离', 'village_group':'村组'}}


    # datasetname_mapping = {0:'weibo',1:'ecommerce',2:'msra',3:'ontonote',4:'resume',\
    #                   5:'cluener',6:'boson',7:'pd',8:'chinese_address'}

    ## mappings of id to datasetname
    datasetname_mapping = {1:'ecommerce',2:'msra',3:'ontonote',4:'resume',\
                      5:'cluener',6:'boson',7:'pd',8:'chinese_address'}

    # categories = list(set(list(mapping.values())+list(resume_mapping.values())))
    categories = []
    for key, subdict in all_datasets_mapping.items():
        categories.extend(list(subdict.values()))

    categories = list(set(categories))

    print(len(categories))
    print(categories)

   

    # print('### loading weibo data ###')
    # weibo_train_categories, weibo_train_data = read_seqlabel_data(f'{args.data_dir}weibo_train.json')
    # weibo_dev_categories, weibo_dev_data = read_seqlabel_data(f'{args.data_dir}weibo_dev.json')
    # print(len(weibo_train_data), len(weibo_dev_data))
   

    print('### loading ontonote data ###')
    ontonote_train_categories, ontonote_train_data = read_seqlabel_data(f'{args.data_dir}ontonotes_train.json')
    ontonote_dev_categories, ontonote_dev_data = read_seqlabel_data(f'{args.data_dir}ontonotes_dev.json')
    print(len(ontonote_train_data), len(ontonote_dev_data))
   

    print('### loading msra data ###')
    msra_train_categories, msra_train_data = read_seqlabel_data(f'{args.data_dir}MSRA_train.json')
    msra_dev_categories, msra_dev_data = read_seqlabel_data(f'{args.data_dir}MSRA_test.json')
    print(len(msra_train_data),len(msra_dev_data))
    

    print('### loading resume data ###')
    resume_train_categories, resume_train_data = read_seqlabel_data(f'{args.data_dir}resume_train.json')
    resume_dev_categories, resume_dev_data = read_seqlabel_data(f'{args.data_dir}resume_dev.json')
    print(len(resume_train_data), len(resume_dev_data))
    

    print('### loading cluener data ###')
    cluener_train_categories, cluener_train_data = read_seqlabel_data(f'{args.data_dir}ml_train.json')
    cluener_dev_categories, cluener_dev_data = read_seqlabel_data(f'{args.data_dir}ml_test_all.json')
    print(len(cluener_train_data), len(cluener_dev_data))
    

    print('### loading people daily data ###')
    pd_train_categories, pd_train_data = read_seqlabel_data(f'{args.data_dir}people_daily_train.json')
    pd_dev_categories, pd_dev_data = read_seqlabel_data(f'{args.data_dir}people_daily_dev.json')
    print(len(pd_train_data), len(pd_dev_data))
    ## people daily size is too big, subsample
    # pd_train_data = random.sample(pd_train_data, int(len(pd_train_data)/10))
    

    print('### loading boson data ###')
    boson_train_categories, boson_train_data = read_seqlabel_data(f'{args.data_dir}boson_train.json')
    boson_dev_categories, boson_dev_data = read_seqlabel_data(f'{args.data_dir}boson_dev.json')
    print(len(boson_train_data),len(boson_dev_data))
    

    print('### loading ecommerce data ###')
    ecommerce_train_categories, ecommerce_train_data = read_seqlabel_data(f'{args.data_dir}ecommerce_train.json')
    ecommerce_dev_categories, ecommerce_dev_data = read_seqlabel_data(f'{args.data_dir}ecommerce_dev.json')
    print(len(ecommerce_train_data), len(ecommerce_dev_data))
    

    print('### loading chinese_address data ###')
    chinese_address_train_categories, chinese_address_train_data = read_seqlabel_data(f'{args.data_dir}chinese_address_train.json')
    chinese_address_dev_categories, chinese_address_dev_data = read_seqlabel_data(f'{args.data_dir}chinese_address_dev.json')
    print(len(chinese_address_train_data), len(chinese_address_dev_data))


    ## datasets
    # train_datasets = {0:weibo_train_data,1:ecommerce_train_data,2:msra_train_data,3:ontonote_train_data,4:resume_train_data,\
    #                   5:cluener_train_data,6:boson_train_data,7:pd_train_data,8:chinese_address_train_data}
    # dev_datasets = {0:weibo_dev_data,1:ecommerce_dev_data,2:msra_dev_data,3:ontonote_dev_data,4:resume_dev_data,\
    #                   5:cluener_dev_data,6:boson_train_data,7:pd_train_data,8:chinese_address_dev_data}

    train_datasets = {1:ecommerce_train_data,2:msra_train_data,3:ontonote_train_data,4:resume_train_data,\
                      5:cluener_train_data,6:boson_train_data,7:pd_train_data,8:chinese_address_train_data}
    dev_datasets = {1:ecommerce_dev_data,2:msra_dev_data,3:ontonote_dev_data,4:resume_dev_data,\
                      5:cluener_dev_data,6:boson_dev_data,7:pd_dev_data,8:chinese_address_dev_data}

    ## reconstruct examples
    new_train_datasets = {}
    # new_train_datasets_aug = {}
    # for i in range(1,len(train_datasets)):
    for dataset_id in datasetname_mapping.keys():
        subcategories = list(all_datasets_mapping[dataset_id].values())
        # if i!=4:
        #     new_train_datasets[i] = dataset_construct(mapping,train_datasets[i],categories,args.max_entities,method='1')
        #     new_train_datasets_aug[i] = dataset_construct(mapping,train_datasets[i],categories,args.max_entities,method='2')
        # else:
        new_train_datasets[dataset_id] = dataset_construct(all_datasets_mapping[dataset_id],train_datasets[dataset_id],subcategories,args.max_entities,method=args.method)
        # new_train_datasets_aug[i] = dataset_construct(resume_mapping,train_datasets[i],categories,args.max_entities,method='2')

    

    new_dev_datasets = {}
    for dataset_id in datasetname_mapping.keys():
        subcategories = list(all_datasets_mapping[dataset_id].values())
        # if i!=4:
        #     new_dev_datasets[i] = dataset_construct(mapping,dev_datasets[i],categories,args.max_entities,method='1')
        # else:
        new_dev_datasets[dataset_id] = dataset_construct(all_datasets_mapping[dataset_id],dev_datasets[dataset_id],subcategories,args.max_entities,method=args.method)


    ## construct valid data for each datasets
    dev_loaders = {}
    valid_labels_set = {}
    for dataset_id in datasetname_mapping.keys():
        dev_dataset = new_dev_datasets[dataset_id]

        dev_inputs, dev_labels = [line['input_seq'] for line in dev_dataset], [line['output_seq'] for line in dev_dataset]
        dev_inputs = tokenizer(dev_inputs, padding="longest", max_length=args.max_source_length, truncation=True, return_tensors="pt")
        dev_labels = tokenizer(dev_labels, padding="longest", max_length=args.max_target_length, truncation=True, return_tensors='pt')
        dev_inputs['labels'] = dev_labels.input_ids.detach().clone()
        dev_inputs['sample_id'] = torch.tensor([i for i in range(len(dev_dataset))])

        dev_dataset = MyDataset(dev_inputs)
        dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_dataset, shuffle=False) if args.distributed else None
        dev_loader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=args.dev_batch_size_per_gpu, shuffle=False)
        dev_loaders[dataset_id] = dev_loader

        valid_labels = [] ## ground truths
        for line in dev_datasets[dataset_id]:
            label = []
            for item in line[1:]:
                # if dataset_id != 4:
                # if item[3]!='o' and item[3]!='product_name':
                if item[3]!='o':
                    new_item = [all_datasets_mapping[dataset_id][item[3]],item[2].lower()]
                    label.append(new_item)
                # else:
                #     new_item = [resume_mapping[item[3]],item[2].lower()]
                #     label.append(new_item)
                
            valid_labels.append(label)

        valid_labels_set[dataset_id] = valid_labels
    
    ## demonstration training examples
    for dataset_id in all_datasets_mapping.keys():
        d_name = datasetname_mapping[dataset_id]
        print(f'## example demonstration {d_name}')
        print(f'## train example {new_train_datasets[dataset_id][0]}')
        print(f'## dev example {new_dev_datasets[dataset_id][0]}')
    

    model = MT5ForConditionalGeneration.from_pretrained(args.model).to(args.device)
    model.config.max_length = args.model_max_len

    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    

    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    print(f'### total number of training steps is {args.steps}')
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=args.warm_up_step, num_training_steps=args.steps
    )
    # Creates a GradScaler once at the beginning of training.
    scaler = GradScaler()
    num_instance_per_batch = args.num_gpus*args.train_batch_size_per_gpu
    print(f'### total number of batch size is {num_instance_per_batch}')



    losses = [] ## training loss
    eval_losses = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]} ## eval loss
    avgs = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]} ## avg f-score
    x_axis = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]} ## x-axis for plotting eval performance for each dataset
    dataset_keys = list(datasetname_mapping.keys())

    for step in tqdm(range(args.start_step,args.steps)):

        set_seed(step)
        # fix sampling seed such that each gpu gets different part of dataset

        ## training batch construct
        batch_example = []
        for i in range(num_instance_per_batch):
            dataset_id = random.sample(dataset_keys,1)[0]
            ## one example has exact match tags
            example = random.sample(new_train_datasets[dataset_id],1)
            ## one example has random tags
            # example_aug = random.sample(new_train_datasets_aug[dataset_id],1)[0]
            batch_example.extend(example)
            # batch_example.append(example_aug)

        train_inputs, train_labels = [line['input_seq'] for line in batch_example], [line['output_seq'] for line in batch_example]
        train_inputs = tokenizer(train_inputs, padding="longest", max_length=args.max_source_length, truncation=True, return_tensors="pt")
        train_labels = tokenizer(train_labels, padding="longest", max_length=args.max_target_length, truncation=True, return_tensors='pt')
        train_inputs['labels'] = train_labels.input_ids.detach().clone()


        train_dataset = MyDataset(train_inputs)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True) if args.distributed else None
        train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=num_instance_per_batch, shuffle=(not args.distributed))#如果用分布式训练这里shuffle=False


        if args.distributed: 
            train_loader.sampler.set_epoch(step)

        for batch in train_loader:
            model.train()
            
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
                    fout.write(f'{step}:{batch_loss}\n')

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
            
            if step%args.eval_steps == 0 and args.do_eval and step!=args.start_step:


                for dataset_id in all_datasets_mapping.keys():

                    print(f' ### starting evaluation for dataset {datasetname_mapping[dataset_id]} ###')
                    x_axis[dataset_id].append(step)
                    eval_loss = 0.0 ## total loss of this step
                    predictions = [] ## predictions of this sampler
                    sampler_ids = [] ## index of this sampler
                    model.eval()
                
                
                    for batch in tqdm(dev_loaders[dataset_id]):

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
                        outputs = model.module.generate(input_ids=input_ids,num_beams=args.beam_width, max_length=args.decode_max_len, eos_token_id=tokenizer.sep_token_id)
                                       
            
                        for pred in tokenizer.batch_decode(outputs, skip_special_tokens=True):
                            pred = ''.join(pred.split(' '))
                            predictions.append(pred)
                            

                        del loss, labels, input_ids, attention_mask
                        gc.collect()
                        torch.cuda.empty_cache()

                   ## write loss of each rank to local
                    with open(f"{args.save_dir}{args.rank}_{datasetname_mapping[dataset_id]}_eval_batch_loss.txt",'w') as fout:
                        fout.write(f'{eval_loss}')

                    dist.barrier()
                    # plotting eval loss
                    if args.rank == 0:
                        batch_loss = 0.0
                        ## read loss of each rank from local
                        for i in range(ntasks):
                            i = str(i)
                            with open(f"{args.save_dir}{i}_{datasetname_mapping[dataset_id]}_eval_batch_loss.txt",'r') as fin:
                                rank_loss = float(fin.readlines()[0])
                                batch_loss+=rank_loss

                        eval_losses[dataset_id].append(batch_loss)
                        
                        fig, ax = plt.subplots(figsize=(10,6))
                        x = x_axis[dataset_id]
                        y = eval_losses[dataset_id]
                        ax.set_ylabel('Eval loss')
                        ax.set_xlabel('steps')
                        ax.plot(x,y)
                        plt.savefig(f'{args.save_dir}{datasetname_mapping[dataset_id]}_eval_loss.pdf',bbox_inches='tight')
                        plt.clf()
                        
                        with open(f"{args.save_dir}{datasetname_mapping[dataset_id]}_eval_loss.txt",'a') as fout:
                            fout.write(f"{step}:{batch_loss}\n")
                    
                    # compute and plotting eval f-score
                    
                    postprocess_preds = []
                    for pred in predictions:
                        pred = pred.split(',')
                        pred = [item.replace(')','').replace('(','').split(':') for item in pred]
        #                 pred = [item.replace(']','').replace('[','').split(':') for item in pred]

                        ## normalize predictions
                        new_pred = []
                        for item in pred:
                            if len(item)>1 and item[1]!='null':
                                new_pred.append(item)
                        # pred = [item for item in pred if len(item)>1 and item[1]!='null']
                        postprocess_preds.append(new_pred)

                    with open(f"{args.save_dir}{args.rank}_{datasetname_mapping[dataset_id]}_preds.txt", 'w', encoding='utf8') as fout:
                        for line in postprocess_preds:
                            fout.write(f"{line}\n")

                    dist.barrier()

                    sampler_labels = []
                    valid_labels = valid_labels_set[dataset_id]
                    for idx,line in enumerate(valid_labels):
                        if idx in sampler_ids:
                            sampler_labels.append(line)


                    with open(f"{args.save_dir}{args.rank}_{datasetname_mapping[dataset_id]}_labels.txt", 'w', encoding='utf8') as fout:
                        for line in sampler_labels:
                            fout.write(f"{line}\n")



                    dist.barrier()

                    if args.rank == 0:

                        postprocess_preds = []
                        gold_labels = []

                        for i in range(ntasks):
                            i = str(i)
                            with open(f"{args.save_dir}{i}_{datasetname_mapping[dataset_id]}_preds.txt",'r') as fin:
                                for line in fin.readlines():
                                    postprocess_preds.append(ast.literal_eval(line))

                        for i in range(ntasks):
                            i = str(i)
                            with open(f"{args.save_dir}{i}_{datasetname_mapping[dataset_id]}_labels.txt",'r') as fin:
                                for line in fin.readlines():
                                    gold_labels.append(ast.literal_eval(line))

                        with open(f"{args.save_dir}{datasetname_mapping[dataset_id]}_{step}_preds.txt", 'w', encoding='utf8') as fout:
                            for line in postprocess_preds:
                                fout.write(f"{line}\n")

                        with open(f"{args.save_dir}{datasetname_mapping[dataset_id]}_gold_labels.txt", 'w', encoding='utf8') as fout:
                            for line in gold_labels:
                                fout.write(f"{line}\n")
                    
                        sub_categories = list(all_datasets_mapping[dataset_id].values())
                        score,avg = get_f1_score(postprocess_preds,gold_labels,sub_categories)
                        print(f"{step}:{avg}")
                        avgs[dataset_id].append(avg)
                        with open(f"{args.save_dir}{datasetname_mapping[dataset_id]}_eval_f_score_details.txt",'w') as fout:
                            for line,value in score.items():
                                fout.write(f"{line}:{value}\n")
                        
                        fig, ax = plt.subplots(figsize=(10,6))
                        x = x_axis[dataset_id]
                        y = avgs[dataset_id]
                        ax.set_ylabel('Eval F-score')
                        ax.set_xlabel('steps')
                        ax.plot(x,y)
                        plt.savefig(f'{args.save_dir}{datasetname_mapping[dataset_id]}_eval_f_score.pdf',bbox_inches='tight')
                        plt.clf()
                        
                        with open(f"{args.save_dir}{datasetname_mapping[dataset_id]}_eval_f_score.txt",'a') as fout:
                            fout.write(f"{step}:{avg}\n")
                    
                    
                        # saving model checkpoints
                        
                        if not os.path.exists(f'{args.save_dir}{step}'):
                            print('### staring saving model ###')
                            model.module.save_pretrained(f'{args.save_dir}{step}')          



    print('Training Finishing!!!!')
