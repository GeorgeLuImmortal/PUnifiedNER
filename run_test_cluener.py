
from transformers import BertTokenizer,T5Tokenizer, T5ForConditionalGeneration, MT5ForConditionalGeneration, Text2TextGenerationPipeline
import torch, argparse, json, gc, os, ast
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

def dataset_construct(mapping:dict, data:list, categories:list) -> list:
        
    dataloader = []
    # num_categories = len(categories)

    for example in data:

        ori_text = example['text']
        source_seq = ''


        prefix_tags = categories ## select all tags


        for tag in prefix_tags:
            source_seq = source_seq+f"<实体>{tag}"

        source_seq = source_seq+f"<文本>{ori_text}"

        dataloader.append({'input_seq':source_seq.lower()})
            
                    
    return dataloader


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='search for best template according to dev set')
    parser.add_argument('--max_source_length', default=128, type=int, help="max source sequence length")
    parser.add_argument('--max_target_length', default=64, type=int, help="max target sequence length")
    parser.add_argument('--batch_size', default=2, type=int, help="batch size")
    parser.add_argument('--epoch', default=20, type=int, help="training epoches")
    parser.add_argument('--model', default='../v1.3/', type=str, help="pretrained model")
    parser.add_argument('--version', default='1.3', type=str, help="model version")
    parser.add_argument('--tokenizer', default='./models/my_t5_base/', type=str, help="tokenizer")
    parser.add_argument('--dataset_name', default='cluener', type=str, help="dataset name")
    parser.add_argument('--dev_dir', default='../cws-dev/dataset/cluener/test.json', type=str, help="development set")
    parser.add_argument('--eval_result_dir', default='./results/', type=str, help="development set")
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.eval_result_dir):
        os.mkdir(args.eval_result_dir)






    test_path = args.dev_dir
    test_data = []
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            test_data.append(ast.literal_eval(line))
            
    len(test_data)





    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # device = "cpu"
    mapping_reverse = {## cluener
    '名称':'name','公司':'company','游戏':'game','组织':'organization','电影':'movie', '地点':'address','职位':'position','政府':'government','景点':'scene','书籍':'book'}
                           
    mapping = {'name':'名称','company':'公司','game':'游戏','organization':'组织','movie':'电影', 'address':'地点','position':'职位','government':'政府','scene':'景点','book':'书籍'}

    categories = list(mapping.values())



    # test_data = test_data[:50]






    test_datasets = dataset_construct(mapping,test_data,categories)





    test_batches = []
    for index in range(0,len(test_datasets),args.batch_size):
        batch = []
        start = index
        end = index+args.batch_size
        
        for line in tqdm(test_datasets[start:end]):
            batch.append(line)
        
        
        test_batches.append(batch)




    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)

    avgs = []
    for beam_width in [10]:
        
        print(f"### staring load model {args.model} ###")
        model = MT5ForConditionalGeneration.from_pretrained(args.model).to(device)
        predictions = []
        for batch in tqdm(test_batches):

            input_sequences = []
            output_sequences = []

            for example in batch:
                input_sequences.append(example['input_seq'])
    #             output_sequences.append(example['output_seq'])

            # encode the inputs
            encoding = tokenizer(input_sequences,padding="longest", max_length=args.max_source_length, truncation=True, return_tensors="pt",
            )
            input_ids, attention_mask = encoding.input_ids.to(device), encoding.attention_mask.to(device)

            # inference
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask,                             do_sample=False,eos_token_id=tokenizer.sep_token_id,num_beams = beam_width, max_length= args.max_target_length,
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





    output_result = []
    for item, preds in zip(test_data,postprocess_preds):
        idx = item['id']
        text = item['text']
        lower_text = text.lower()
        ner_dict = {}
        for pred in preds:
        
            start = lower_text.find(pred[1])
            if start==-1:
                pass
    #             print(pred[1])
    #             print(text)
            else:
                tag = mapping_reverse[pred[0]]
                if tag not in ner_dict:
                    end = start+len(pred[1])
                    tag_text = text[start:end]
                    tag = mapping_reverse[pred[0]]
                    ner_dict[tag] = {tag_text:[[start,end-1]]}
                else:
                    end = start+len(pred[1])
                    tag_text = text[start:end]
                    
                    if tag_text in ner_dict[tag]:
                        old_end = ner_dict[tag][tag_text][-1][1]
    #                     print(idx)
                        new_start = lower_text[old_end:].find(pred[1])
                        new_start = new_start+old_end
                        new_end = new_start+len(pred[1])
                        ner_dict[tag][tag_text].append([new_start,new_end-1])
                    else:
                        ner_dict[tag].update({tag_text:[[start,end-1]]})
                    
                 
           
        output_result.append({"id":idx,"label":ner_dict})





    print(output_result[:10])





    with open(f"cluener_predict_{args.version}.json","w",encoding="utf8") as file:
        for line in output_result:
            json.dump(line,file,ensure_ascii=False)
            file.write("\n")







