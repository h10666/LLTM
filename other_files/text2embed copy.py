import torch
import torch.nn as nn
import CLIP
from CLIP import clip
# from clip import model
from archs.clip_prepare_model import clip_prepare_model
from archs.clip_model import build_model
from transformers import RobertaModel, RobertaTokenizerFast
from transformers import AutoTokenizer, AutoModel
from opts import parser

# print(clip.available_models())
# exit(0)

args, _ = parser.parse_known_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B-32", device=device)
# exit(0)

class text2embed(nn.Module):
    def __init__(self, text_encoder_type="roberta-base"):
        super(text2embed, self).__init__()
        self.embed_way = args.embed_way

        if self.embed_way == 'RoBERTa':
            self.tokenizer = RobertaTokenizerFast.from_pretrained(text_encoder_type)
            self.text_encoder = RobertaModel.from_pretrained(text_encoder_type)
        
            for p in self.text_encoder.parameters():
                    p.requires_grad_(False)
        elif self.embed_way == 'AutoBERT':
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.text_encoder = AutoModel.from_pretrained("bert-base-uncased")
        else:
            self.model, self.preprocess = clip.load("ViT-B-32", device=device)
            # self.model = torch.nn.DataParallel(self.model)
        
    def forward(self, x):
        x = list(x)
        # print(x)
        if self.embed_way == 'CLIP':
            tokenized = clip.tokenize(x).to(device)
            # print(tokenized)
            output = self.model.encode_text(tokenized)
            # print(output)
            # print(type(output))
        elif self.embed_way == 'RoBERTa':
            tokenized = self.tokenizer.batch_encode_plus(x, padding="longest", return_tensors="pt")
            outputs = self.text_encoder(**tokenized)
            output = outputs.pooler_output
            # print(output)
            # print(type(output))
            # print(output.size())
        else:
            inputs = self.tokenizer(x, padding="longest", return_tensors="pt").to(device)
            outputs = self.text_encoder(**inputs)
            output = outputs.pooler_output
            # print(output)
            # print(type(output))
            # print(output.size())
        # exit(0)
        
        return output