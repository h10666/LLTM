import json
import random
def readjson(jsonpath):#读取json
    with open(jsonpath,'r') as f:
        temp =json.load(f)
    return temp

def shuffle_vid(vid):#打乱sth-else的所有vid
    shuffle_train_vid=[]
    shuffle_valid_vid=[]
    s = random.sample(range(len(vid)), len(vid))
    for i in range(0, len(vid)//2):
        shuffle_train_vid.append(vid[s[i]])
    for i in vid:
        if i not in shuffle_train_vid:
            shuffle_valid_vid.append(i)

    return shuffle_train_vid, shuffle_valid_vid
        

def shuffle_train(all_json,shuffle_train_vid):#划分train
    # print(len(all_json))
    for i in all_json:
        if i['id'] in shuffle_train_vid:
            continue
        else:
            all_json.remove (i)
    with open('/home/10601001/wsm/sth_else/sth_train_7.json','w') as f:
        f.write((json.dumps(all_json)))
    
def shuffle_valid(all_json,shuffle_valid_vid):#划分val
    for i in all_json:
        if str(i['id']) not in shuffle_valid_vid:
            all_json.remove(i)
        else:
            continue
    with open('/home/10601001/wsm/sth_else/sth_val_6.json','w') as f:
        f.write((json.dumps(all_json))) 





if __name__ == '__main__':
    # mini_train_json = 'tools/anno_STHELSE/COM/mini_train.json'
    train_json='/home/10102007/TSM_NEW_CHECK3/tools/anno_STHELSE/SHUFFLED/train.json'#shuffle-train
    valid_json = '/home/10102007/TSM_NEW_CHECK3/tools/anno_STHELSE/SHUFFLED/validation.json'#shuffle-val
    sthall_json = '/home/10102007/TSM_NEW_CHECK3/tools/anno_STHELSE/SHUFFLED/sth_all.json'#train+val
    # mini_tr_json = readjson(mini_train_json)
    tr_json=readjson(train_json)
    print('tr_json', len(tr_json))
    va_json=readjson(valid_json)
    print('va_json', len(va_json))
    all_json=readjson(sthall_json)
    print('all_json', len(all_json))
    # exit(0) 
    all_vid=[]
    # for i in mini_tr_json:
    #     all_vid.append(i['id'])
    # print(all_vid)    
    # exit(0)
    for i in all_json:
        all_vid.append(i['id'])
    print(len(all_vid))    
        
    shuffle_train_vid, shuffle_valid_vid=shuffle_vid(all_vid)
    print(len(shuffle_train_vid))# 56397
    print(len(shuffle_valid_vid))#56398
    print(len(all_vid))#
    train = []
    val = []
    for i in all_json:
        if i['id'] in shuffle_train_vid:
            train.append(i)
        else:
            val.append(i)
    print('new_train:\t',len(train))
    print('new_val:\t', len(val))
    
            
    # for i in all_json:
    #     if i['id'] in shuffle_train_vid:
    #         continue
    #     else:
    #         all_json.remove(i)
    with open('/home/10102007/TSM_NEW_CHECK3/tools/anno_STHELSE/SHUFFLED/train3.json', 'w') as f:
            f.write((json.dumps(train)))
    with open('/home/10102007/TSM_NEW_CHECK3/tools/anno_STHELSE/SHUFFLED/val3.json', 'w') as f:
            f.write((json.dumps(val)))






    # print(len(all_vid))
    # shuffle_train(all_json, shuffle_train_vid)
    # shuffle_valid(all_json, shuffle_valid_vid)
    # train=[]
    


