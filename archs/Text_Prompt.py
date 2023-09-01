from numpy.lib.function_base import append
import torch
# import clip
from CLIP import clip
# from archs.two_stream_new import Two_stream
import os
# from archs.model_builder import get_tokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "true"
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
def text_prompt(label_text):
    text_aug = [f"a photo of action {{}}", f"a picture of action {{}}", f"Human action of {{}}", f"{{}}, an action",
                f"{{}} this is an action", f"{{}}, a video of action", f"Playing action of {{}}", f"{{}}",
                f"Playing a kind of action, {{}}", f"Doing a kind of action, {{}}", f"Look, the human is {{}}",
                f"Can you recognize the action of {{}}?", f"Video classification of {{}}", f"A video of {{}}",
                f"The man is {{}}", f"The woman is {{}}"]
    text_dict = {}
    # num_text_aug = len(text_aug)

    for ii, txt in enumerate(text_aug):
        # text_dict[ii] = torch.cat([clip.tokenize(txt.format(c)) for i, c in data.classes])
        text_dict[ii] = torch.cat([clip.tokenize(txt.format(c)) for c in label_text])
        
    # print('text_dict:\t', text_dict[0].size()) # <174, 77>   
    # print(label_text)
    classes = torch.cat([v for k, v in text_dict.items()])
    # print(classes.size(), classes)
    # exit(0)
    return text_dict, classes

def captions_prompt(captions, text_id):
    text_aug = [f"a photo of action {{}}", f"a picture of action {{}}", f"Human action of {{}}", f"{{}}, an action",
                f"{{}} this is an action", f"{{}}, a video of action", f"Playing action of {{}}", f"{{}}",
                f"Playing a kind of action, {{}}", f"Doing a kind of action, {{}}", f"Look, the human is {{}}",
                f"Can you recognize the action of {{}}?", f"Video classification of {{}}", f"A video of {{}}",
                f"The man is {{}}", f"The woman is {{}}"]
    # text_dict = {}
    num_text_aug = len(text_aug)
    texture = []
    for c in captions:    
        for txt in text_aug:         
            caption = txt.format(c)
            texture.append(caption)

    text_input = clip.tokenize(texture)

    text_input = text_input.view(-1, num_text_aug, text_input.size(-1))
    b, _, _ = text_input.size()
    order = torch.arange(b)
    text_input = torch.stack([text_input[i][j,:] for i,j in zip(order,text_id)])

    return text_input

# from transformers import RobertaModel, RobertaTokenizerFast
# from transformers import AutoTokenizer, AutoModel
def tags_prompt(tags, text_id, tokenizer, do_prompt, do_kl):
    #  "obj1, obj2"
    if not do_kl:
        text_aug = [f"A photo contains {{}}", f"A video contains {{}}"]
    else:   
        text_aug = [f"{{}}", f"{{}}"]
    
    
    # text_dict = {}
    num_text_aug = len(text_aug)
    texture = []
    for c in tags:    
        for txt in text_aug:         
            caption = txt.format(c)
            texture.append(caption)
    # for c in tags:
    #     captions = []  
    #     for txt in text_aug:         
    #         caption = txt.format(c)
    #         captions.append(caption)
    #     texture.append(captions)

    # text_input = []
    # for item, j in zip(texture, text_id):
    #     a = item[j]
    #     text_input.append(a)
    ###############################################################################
    # clip tokenizer
    # text_input = clip.tokenize(texture)
    # text_input = text_input.view(-1, num_text_aug, text_input.size(-1))
    # b, _, _ = text_input.size()
    # order = torch.arange(b)
    # text_input = torch.stack([text_input[i][j,:] for i,j in zip(order,text_id)])
    ###############################################################################
    # # tokenizer = get_tokenizer("roberta-base")
    if do_prompt:
        try:
            text_input = tokenizer.batch_encode_plus(texture, padding="longest", return_tensors="pt").to(device)
        except:
            text_input = tokenizer(texture, padding=True, return_tensors="pt").to(device)
            
        
        # # # print(type(text_input), text_input)
        for item in text_input.keys():
            text_input[item] = text_input[item].view(-1, num_text_aug, text_input[item].size(-1))
            b, _, _ = text_input[item].size()
            order = torch.arange(b)
            text_input[item] = torch.stack([text_input[item][i][j,:] for i,j in zip(order,text_id)])
    else:
        tags = list(tags)
        try:
            text_input = tokenizer.batch_encode_plus(tags, padding="longest", return_tensors="pt").to(device)
        except:
            text_input = tokenizer(texture, padding=True, return_tensors="pt").to(device)
        # print(text_input)
    ###############################################################################
    # print(text_input)
    # exit(0)
    text_input_ids = text_input['input_ids']
    attention_mask = text_input['attention_mask']
    text_input = (text_input_ids, attention_mask)
    return text_input

# def tokenizer():
#     tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
#     return 

    # visual_fea 2048
    # tag_fea 512
    # CLS(MLP([visual_fea, tag_fea]))  vs. CLS(visual_fea)
    # DET--> box, tags; detected box (ROI/embedding): 54
    #