
from transformers import BertTokenizer,T5Tokenizer, T5ForConditionalGeneration, MT5ForConditionalGeneration, Text2TextGenerationPipeline
import torch, argparse, json, gc, os, random
from tqdm import tqdm
from torch.optim import AdamW
from pathlib import Path
import matplotlib.pyplot as plt
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

def dataset_construct(mapping:dict, data:list, categories:list, max_entities: int,method:str ='1', is_train:bool = True) -> list:
        
    dataloader = []
    # num_categories = len(categories)

    for example in data:

        ori_text = example[0]

        ## with exact tags
        if '1' in method:
            prefix_tags = []
            target_seq = "("
            source_seq = ''

            if is_train:
                for item in example[1:]:
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

            if is_train:
                dataloader.append({'input_seq':source_seq.lower(),'output_seq':target_seq.lower()})
            else:
                dataloader.append({'input_seq':source_seq.lower()})

        ## 2. with random tags
        if '2' in method:

            target_seq = "("
            source_seq = ''

            num_tags = random.randint(0, max_entities)
            prefix_tags = list(np.random.choice(categories,num_tags,replace=False))## select random tags from the pool

            if is_train:
                exist_tags = []
                for item in example[1:]:
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


            if is_train:
                dataloader.append({'input_seq':source_seq.lower(),'output_seq':target_seq.lower()})
            else:
                dataloader.append({'input_seq':source_seq.lower()})


        ## 3. with all tags
        if '3' in method:

            target_seq = "("
            source_seq = ''

    
            prefix_tags = categories ## select all tags

            if is_train:
                exist_tags = []
                for item in example[1:]:
                    if item[3]!='o' and item[3]!='product_name':
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


            if is_train:
                dataloader.append({'input_seq':source_seq.lower(),'output_seq':target_seq.lower()})
            else:
                dataloader.append({'input_seq':source_seq.lower()})
            
                    
    return dataloader





parser = argparse.ArgumentParser(description='search for best template according to dev set')
parser.add_argument('--max_source_length', default=128, type=int, help="max source sequence length")
parser.add_argument('--max_target_length', default=64, type=int, help="max target sequence length")
parser.add_argument('--batch_size', default=1, type=int, help="batch size")
parser.add_argument('--epoch', default=20, type=int, help="training epoches")
parser.add_argument('--model', default='./my_trained_models/30000/', type=str, help="pretrained model")
parser.add_argument('--tokenizer', default='./models/my_t5_base/', type=str, help="tokenizer")
parser.add_argument('--dataset_name', default='resume', type=str, help="dataset name")
parser.add_argument('--dev_dir', default='./ner_datasets/resume_test.json', type=str, help="development set")
parser.add_argument('--eval_result_dir', default='./results/', type=str, help="development set")
parser.add_argument('--beam_width', default='5_6_7_8_9_10', type=str, help="beam_width_list")
args = parser.parse_args()
print(args)

if not os.path.exists(args.eval_result_dir):
    os.mkdir(args.eval_result_dir)


# In[34]:


dev_categories, dev_data = read_seqlabel_data(args.dev_dir)


# In[35]:


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = "cpu"
# mapping = {## cluener
#            'name':'名称','company':'公司','game':'游戏','organization':'组织','movie':'电影', 'address':'地点','position':'职位','government':'政府','scene':'景点','book':'书籍',\
#            # 'name':'姓名','company':'公司','game':'游戏','organization':'组织','movie':'电影','address':'地址','position':'职位','government':'政府','scene':'景点','book':'书籍',\
#            ## ecommerce
#            'HP':'品牌','HC':'商品',\
#            ## msra, ontonote, people_daily
#            'GPE':'地点','LOC':'地点','PER':'名称','ORG':'组织','T':'时间',\
#            ## webio
#            # 'PER.NAM':'名称','PER.NOM':'名称泛指','GPE.NAM':'地缘政治实体','GPE.NOM':'地缘政治实体泛指','LOC.NOM':'地点泛指','LOC.NAM':'地点','ORG.NOM':'组织泛指','ORG.NAM':'组织',\
#             ## boson
#            'time':'时间', 'person_name':'名称', 'org_name':'组织', 'location':'地点', 'company_name':'公司',\
#             ## chinese_address
#            'prov':'省份', 'city':'城市', 'district':'区', 'town':'街道', 'community':'社区', 'poi':'兴趣点', 'road':'路', 'roadno':'路号',\
#            'subpoi':'次兴趣点', 'devzone':'产业园', 'houseno':'楼号', 'intersection':'路口', 'assist':'方位', 'cellno':'单元', 'floorno':'楼层', 'distance':'距离', 'village_group':'村组'}
                      
# resume_mapping = {## resume
# 'NAME':'名称', 'CONT':'国籍', 'RACE':'民族', 'TITLE':'职位', 'EDU':'学历', 'ORG':'公司', 'PRO':'专业', 'LOC':'籍贯'} 

## for evaluation
all_datasets_mapping = {
# 'weibo':{'PER.NAM':'名称','PER.NOM':'名称泛指','GPE.NAM':'地缘政治实体','GPE.NOM':'地缘政治实体泛指','LOC.NOM':'地点泛指','LOC.NAM':'地点','ORG.NOM':'组织泛指','ORG.NAM':'组织'},
                        'ecommerce':{'HP':'品牌','HC':'商品'},
                        # 'ecommerce':{'HP':'品牌','HC':'产品'},
                        'msra':{'LOC':'地点','PER':'名称','ORG':'组织'},
                        'ontonote':{'LOC':'地点','PER':'名称','ORG':'组织','GPE':'地点'},
                        'resume':{'NAME':'名称', 'CONT':'国籍', 'RACE':'民族', 'TITLE':'职位', 'EDU':'学历', 'ORG':'公司', 'PRO':'专业', 'LOC':'籍贯'},
                        'cluener':{'name':'名称','company':'公司','game':'游戏','organization':'组织','movie':'电影', 'address':'地点','position':'职位','government':'政府','scene':'景点','book':'书籍'},
                        # 'cluener':{'name':'姓名','company':'公司','game':'游戏','organization':'组织','movie':'电影','address':'地址','position':'职位','government':'政府','scene':'景点','book':'书籍'},
                        'boson':{'time':'时间', 'person_name':'名称', 'org_name':'组织', 'location':'地点', 'company_name':'公司'},
                        'pd':{'LOC':'地点','PER':'名称','ORG':'组织','T':'时间'},
                        'chinese_address':{'prov':'省份', 'city':'城市', 'district':'区', 'town':'街道', 'community':'社区', 'poi':'兴趣点', 'road':'路', 'roadno':'路号',\
       'subpoi':'次兴趣点', 'devzone':'产业园', 'houseno':'楼号', 'intersection':'路口', 'assist':'方位', 'cellno':'单元', 'floorno':'楼层', 'distance':'距离', 'village_group':'村组'}}


# datasetname_mapping = {0:'weibo',1:'ecommerce',2:'msra',3:'ontonote',4:'resume', 5:'cluener',6:'boson',7:'pd',8:'chinese_address'}
datasetname_mapping = {1:'ecommerce',2:'msra',3:'ontonote',4:'resume', 5:'cluener',6:'boson',7:'pd',8:'chinese_address'}

# categories = list(set(list(mapping.values())+list(resume_mapping.values())))



# if args.dataset_name!='resume':
categories = list(all_datasets_mapping[args.dataset_name].values())
dev_datasets = dataset_construct(all_datasets_mapping[args.dataset_name],dev_data,categories,40,method='3',is_train=True)
# else:
#     categories = list(all_datasets_mapping[args.dataset_name].values())
#     dev_datasets = dataset_construct(resume_mapping,dev_data,categories,40,method='3',is_train=True)
    
eval_batches = []
for idx in range(0,len(dev_datasets),args.batch_size):
    batch = []
    try:
        for index in range(idx, idx+args.batch_size):
            batch.append(dev_datasets[index])
    except Exception:
        pass
    
    eval_batches.append(batch)


print(eval_batches[0])


valid_labels = [] ## ground truths
for line in dev_data:
    label = []
    for item in line[1:]:
        # if args.dataset_name != 'resume':
        if item[3]!='o' and item[3]!='product_name':
            new_item = [all_datasets_mapping[args.dataset_name][item[3]],item[2].lower()]
            label.append(new_item)
        # else:
            
        #     new_item = [resume_mapping[item[3]],item[2].lower()]
        #     label.append(new_item)

    valid_labels.append(label)


print(valid_labels[:5])


tokenizer = BertTokenizer.from_pretrained(args.tokenizer)

avgs = []
for beam_width in args.beam_width.split('_'):
    beam_width = int(beam_width)
    
    print(f"### staring load model {args.model} ###")
    model = MT5ForConditionalGeneration.from_pretrained(args.model).to(device)
    predictions = []
    for batch in tqdm(eval_batches):

        input_sequences = []
        output_sequences = []

        for example in batch:
            input_sequences.append(example['input_seq'])
            output_sequences.append(example['output_seq'])

        # encode the inputs
        encoding = tokenizer(input_sequences,padding="longest", max_length=args.max_source_length, truncation=True, return_tensors="pt",
        )
        input_ids, attention_mask = encoding.input_ids.to(device), encoding.attention_mask.to(device)

        # inference
        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, do_sample=False,eos_token_id=tokenizer.sep_token_id,num_beams = beam_width, max_length= 512,
                                decoder_start_token_id=tokenizer.cls_token_id)


        del input_ids, attention_mask
        gc.collect()
        torch.cuda.empty_cache()

        for pred in tokenizer.batch_decode(outputs, skip_special_tokens=True):
            pred = ''.join(pred.split(' '))
            predictions.append(pred)

    postprocess_preds = []
    for pred in predictions:
        pred = pred.split(',')
        pred = [item.replace(')','').replace('(','').split(':') for item in pred]

        new_pred =[]
        for item in pred:
            if len(item)>1 and item[1]!='null':
                new_pred.append(item)
    #                 pred = [item for item in pred if len(item)>1]
        postprocess_preds.append(new_pred)

    with open(f"{args.eval_result_dir}{args.dataset_name}_preds_{beam_width}.txt", 'w', encoding='utf8') as fout:
        for line in postprocess_preds:
            fout.write(f"{line}\n")
    subcategories = list(all_datasets_mapping[args.dataset_name].values())
    score,avg = get_f1_score(postprocess_preds,valid_labels,subcategories)
    
    with open(f"{args.eval_result_dir}{args.dataset_name}_preds_details_{beam_width}.txt", 'w', encoding='utf8') as fout:
        for key,value in score.items():
            fout.write(f"{key}:{value}\n")
    print(f"{args.eval_result_dir}_{beam_width}:{avg}")
    avgs.append(avg)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    
with open(f"{args.eval_result_dir}{args.dataset_name}_f_score.txt", 'w', encoding='utf8') as fout:
    for line in avgs:
        fout.write(f"{line}\n")


