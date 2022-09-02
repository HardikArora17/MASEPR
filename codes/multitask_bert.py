
name_model_part="bert_base_original"

import torch
import matplotlib.pyplot as plt

#If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda:0")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
  print('No GPU available, using the CPU instead.')
  
torch.cuda.set_device(0)
device = torch.device("cpu")

import json
import numpy as np
import os
import random
import re
import pickle
import torch
from tqdm.autonotebook import tqdm
from transformers import AutoTokenizer,AutoModel

model_name='bert-base-uncased'

tokenizer=AutoTokenizer.from_pretrained(model_name)


from torch.utils.data import Dataset, DataLoader

class Data(Dataset):
    def __init__(self,embed,aspects,sentiments):

        self.labels=aspects
        self.labels_s=sentiments
        self.sentences=embed
        self.max_len=50
        self.size=len(embed)

    @classmethod
    def getReader(cls,low,up,test=None,r=1):
        if(True):
          with open("data/onehot_aspect_multitask.pkl",'rb') as out:
              labels =pickle.load(out)
              labels = labels[low:up]
          
          with open("data/onehot_sentiment_multitask.pkl",'rb') as out:
              labels_s=pickle.load(out)
              labels_s = labels_s[low:up]
          
          with open("data/dataframe_multitask.pkl",'rb') as out:
              data_s=pickle.load(out)
              sents = list(data_s['sentences'])[low:up]
        
        assert len(labels) == len(sents) ==len(labels_s)
        print("Total number of Reviews", len(labels))
        
        return cls(sents, labels,labels_s)

    def __getitem__(self,idx):

        sen=self.sentences[idx]

        inputs = tokenizer(sen,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            return_length = True,
            truncation=True)
        
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']
        
        return {
            'ids_sen': torch.tensor(ids),
            'mask_sen': torch.tensor(mask),
            'token_type_ids_sen': torch.tensor(token_type_ids, dtype=torch.float64),
            'targets':self.labels[idx],
            'targets_senti':self.labels_s[idx]}


    def __len__(self):
        return self.size

def getLoaders (batch_size):

        print('Reading the training Dataset...')
        print()
        train_dataset = Data.getReader(0,100000) #19200 #21216
        
        print()

        print('Reading the validation Dataset...')
        print()
        valid_dataset = Data.getReader(100000, 148000) #23200 #25216

        print('Reading the test Dataset...')
        print()
        test_dataset = Data.getReader(148000, 218000) #23200:25248
        
        trainloader = DataLoader(dataset=train_dataset, batch_size = batch_size, num_workers=8,shuffle=True)
        validloader = DataLoader(dataset=valid_dataset, batch_size = batch_size, num_workers=8,shuffle=True)
        testloader = DataLoader(dataset=test_dataset, batch_size = batch_size, num_workers=8)
        
        return trainloader, validloader, testloader

trainloader, validloader, testloader = getLoaders(batch_size=32)

print("Length of TrainLoader:",len(trainloader))
print("Length of ValidLoader:",len(validloader))
print("Length of TestLoader:",len(testloader))

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BertModel(nn.Module):
    def __init__(self, in_features, out_features):

        super(BertModel, self).__init__()
        self.model=AutoModel.from_pretrained(model_name,  output_hidden_states=True)

        self.in_features = in_features   #768
        self.out_features = out_features    #7

        self.flatten=nn.Flatten()
        self.lstm_1 = nn.LSTM(in_features, 200//2, batch_first=True, bidirectional=True) #bidirectional=True
    
      
        self.linear1=nn.Linear(200*out_features,200*2)
        self.linear2=nn.Linear(200*2,256)
        self.linear3=nn.Linear(256,64)

        self.linear1_sen=nn.Linear(200,64)
        self.linear2_sen=nn.Linear(64,2)
   
        self.last_dense = nn.Linear(64, self.out_features)
        self.dropout1=nn.Dropout(p=0.5)
        self.dropout2=nn.Dropout(p=0.2)

        self.relu = nn.ReLU()
        self.sigmoid=nn.Sigmoid()
        self.tanh=nn.Tanh()

        category = torch.rand(200, out_features,requires_grad=True)  #(512,7)
        nn.init.xavier_normal_(category)
       
        self.category=category.to(device)
       
    def forward(self, t1,strategy:str):
        
        ids, mask, token_type_ids = t1
        encoded_layers = self.model(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)[2]
        scibert_hidden_layer = encoded_layers
        
        if(strategy=='last_4'):
          scibert_hidden_layers=torch.cat((scibert_hidden_layer[-1],
                                        scibert_hidden_layer[-2],
                                        scibert_hidden_layer[-3],
                                        scibert_hidden_layer[-4]),dim=2)
          
        if(strategy=='last'):
          scibert_hidden_layers=encoded_layers[12]


        if(strategy=='mean'):
          scibert_hidden_layers=torch.mean(encoded_layers,dim=2)
      

        s_e=scibert_hidden_layers                  #(32,13,768)

        h0 = torch.zeros(2, s_e.size(0), 200 // 2)
        c0 = torch.zeros(2, s_e.size(0), 200 // 2)
        h0, c0 = h0.to(device), c0.to(device)
        s_e, (hn, cn) = self.lstm_1(s_e, (h0, c0))    #(32,50,200)
      
  
        c=self.category.unsqueeze(0)                       #(1,512,7)
        comp = torch.matmul(s_e,c)                         #(32,13,7)
        comp = comp.permute(0,2,1)                         #(32,7,13)

        comp1=    self.relu(self.linear1_sen(s_e))         #(32,50,256)
        comp1 =   self.linear2_sen(comp1)                  #(32,50,2)
        
        wts = F.softmax(comp, dim=2) #(32,7,50)
        wts1= torch.bmm(wts,comp1)   #(32,7,2)
      
        e=torch.bmm(wts,s_e)         #(32,7,200)

        l = torch.reshape(e, (ids.size(0), 200*7))

        l = self.relu(self.linear1(l))
        l = self.dropout1(l)
        l = self.relu(self.linear2(l))
        l = self.dropout1(l)
        l = self.relu(self.linear3(l))

        model_output = self.sigmoid(self.last_dense(l))
        model_output_sent = self.sigmoid(wts1)
        
        del l,comp,s_e,hn,cn,scibert_hidden_layer,ids,mask,token_type_ids
      
        return model_output, wts,comp1,model_output_sent

text_model = BertModel(768,7)
text_model.to(device)
criterion1 = nn.BCELoss()
criterion2 = nn.BCELoss()
from transformers import AdamW, get_linear_schedule_with_warmup

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in text_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in text_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.01}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5)

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=1000)


text_model.train()
result=[]
EPOCH=8

train_out = []
val_out = []
train_true = []
val_true = []
test_out = []
test_true = []
attn_train = []
attn_val = []
attn_test = []
attn_test_senti=[]
test_out_senti=[]
test_true_senti=[]
loss_log1 = []
loss_log2 = []


for epoch in range(EPOCH):

  final_train_loss=0.0
  final_val_loss=0.0
  l1 = []
  text_model.train()

  for idx,data in tqdm(enumerate(trainloader),desc="Train epoch {}/{}".format(epoch + 1, EPOCH)):

    ids = data['ids_sen'].to(device,dtype = torch.long)
    mask = data['mask_sen'].to(device,dtype = torch.long)
    token_type_ids = data['token_type_ids_sen'].to(device,dtype = torch.long)
    targets = data['targets'].to(device,dtype = torch.float)
    targets_s = data['targets_senti'].to(device,dtype = torch.float)
    
    t1 = (ids,mask,token_type_ids)
    
    optimizer.zero_grad()
    out, attn_t,_,out_sen = text_model(t1,'last')

    if (epoch+1 == EPOCH):
      train_out.append((torch.transpose(out,0,1)).detach().cpu())
      train_true.append((torch.transpose(targets,0,1)).detach().cpu())

    loss = (criterion1(out, targets)+criterion2(out_sen, targets_s))/2
    l1.append(loss.item())
    final_train_loss +=loss.item()
    loss.backward()
    optimizer.step()
    if idx % 100 == 0:
      scheduler.step()

    
  loss_log1.append(np.average(l1))

  text_model.eval()
  l2 = []

  
  for data in tqdm(validloader,desc="Valid epoch {}/{}".format(epoch + 1, EPOCH)):
    ids = data['ids_sen'].to(device,dtype = torch.long)
    mask = data['mask_sen'].to(device,dtype = torch.long)
    token_type_ids = data['token_type_ids_sen'].to(device,dtype = torch.long)
    targets = data['targets'].to(device,dtype = torch.float)
    targets_s = data['targets_senti'].to(device,dtype = torch.float)
    
    t1 = (ids,mask,token_type_ids)
    
    out_val, attn_v ,_,out_val_senti= text_model(t1,'last')

    if (epoch+1 == EPOCH):
      val_out.append((torch.transpose(out_val,0,1)).detach().cpu())
      val_true.append((torch.transpose(targets,0,1)).detach().cpu())

    loss = (criterion1(out_val, targets) + criterion2(out_val_senti, targets_s))/2
    l2.append(loss.item())
    final_val_loss+=loss.item()

  loss_log2.append(np.average(l2))
  curr_lr = optimizer.param_groups[0]['lr']

  print("Epoch {}, loss: {}, val_loss: {}".format(epoch+1, final_train_loss/len(trainloader), final_val_loss/len(validloader)))
  print()
  

with torch.no_grad():
   for data in testloader:
     ids = data['ids_sen'].to(device,dtype = torch.long)
     mask = data['mask_sen'].to(device,dtype = torch.long)
     token_type_ids = data['token_type_ids_sen'].to(device,dtype = torch.long)
     targets = data['targets'].to(device,dtype = torch.float)
     targets_s = data['targets_senti'].to(device,dtype = torch.float)

     t1=(ids,mask,token_type_ids)
  
     out_test, attn_T,attn_T_S,out_test_senti = text_model(t1,'last')

     test_out.append((torch.transpose(out_test,0,1)).detach().cpu())
     test_true.append((torch.transpose(targets,0,1)).detach().cpu())
     test_true_senti.append((torch.transpose(targets_s,0,1)).detach().cpu())
     attn_test.append((torch.tensor(attn_T)).detach().cpu())
     attn_test_senti.append((torch.tensor(attn_T_S)).detach().cpu())
     test_out_senti.append((torch.transpose(out_test_senti,0,1)).detach().cpu())
     

plt.plot(range(len(loss_log1)), loss_log1)
plt.plot(range(len(loss_log2)), loss_log2)
plt.savefig('graphs/'+name_model_part+'loss_multi.png')

torch.save(text_model, "ckpt/"+name_model_part+"multi_full_data_model.pt")
torch.save(text_model.state_dict(), "ckpt/"+name_model_part+"multi_full_data_state_dict.pt")

train_out = torch.cat(train_out, 1)
val_out = torch.cat(val_out, 1)
train_true = torch.cat(train_true, 1)
val_true = torch.cat(val_true, 1)
test_out = torch.cat(test_out, 1)
test_out_senti = torch.cat(test_out_senti, 1)
test_true = torch.cat(test_true, 1)
attn_test = torch.cat(attn_test, 0)
attn_test_senti = torch.cat(attn_test_senti, 0)
test_true_senti=torch.cat(test_true_senti, 1)


attnfile = open('outputs/'+name_model_part+'attn_aspect_multi.pkl', 'wb')
pickle.dump(attn_test, attnfile)

attnfile_ss = open('outputs/'+ name_model_part+'attn_sentiment_multi.pkl', 'wb')
pickle.dump(attn_test_senti, attnfile_ss)

test_out_ = (test_out, test_true)
test_out_senti_=(test_out_senti,test_true_senti)

test_outs = open('outputs/'+name_model_part+'test_multi_aspect.pkl', 'wb')
pickle.dump(test_out_, test_outs)

test_outs_sentiment = open('outputs/'+name_model_part+'test_multi_sentiment.pkl', 'wb')
pickle.dump(test_out_senti_, test_outs_sentiment)


f=open("results/"+name_model_part+"multi"+".txt",'w')
f.close()


def labelwise_metrics(pred, true, split):
  f1_final=0.0
  
  f=open("results/"+name_model_part+"multi"+".txt",'a')
  f.write('-'*25 + split + '-'*25 + '\n\n')
   
  pred = (pred>0.425)

  batch_size = len(pred)

  pred = pred.to(torch.int)
  true = true.to(torch.int)

  from sklearn.metrics import accuracy_score
  from sklearn.metrics import confusion_matrix

  for i in range(batch_size):
    acc=accuracy_score(true[i],pred[i])

    epsilon = 1e-7
    confusion_vector = pred[i]/true[i]

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    precision = true_positives/(true_positives+false_positives+epsilon)
    recall = true_positives/(true_positives+false_negatives+epsilon)
    f1 = 2*precision*recall/(precision+recall+epsilon)
    f1_final+=f1

    print("Label: {}, acc: {:.3f}, f1: {:.3f}".format(i+1, acc, f1))
    f.write("Label: {}, acc: {:.3f}, f1: {:.3f}\n".format(i+1, acc, f1))
    f.write(str(confusion_matrix(true[i], pred[i])))
    f.write('\n')
  
  #print(split+" aspect f1 =",f1_final/7)
  return 0

f1=open("results/"+name_model_part+"multi_senti"+".txt",'w')
f1.close()

def labelwise_metrics_senti(pred, true, split):
  f1_final=0.0
  classes = ['MOT +', 'MOT -', 'CLA +', 'CLA -', 'SOU +', 'SOU -', 'SUB +', 'SUB -', 'MEA +', 'MEA -', 'ORI +', 'ORI -', 'REP +', 'REP -']

  f=open("results/"+name_model_part+"multi_senti"+".txt",'a')
  f.write('-'*25 + split + '-'*25 + '\n\n')
   
  pred = (pred>0.425)

  pred = pred.to(torch.int)
  true = true.to(torch.int)
  pred = pred.reshape(7, 2, -1)
  true = true.reshape(7, 2, -1)
  pred = pred.reshape(14, -1)
  true = true.reshape(14, -1)

  batch_size = len(pred)

  from sklearn.metrics import accuracy_score
  from sklearn.metrics import confusion_matrix

  for i in range(batch_size):
    acc=accuracy_score(true[i],pred[i])

    epsilon = 1e-7
    confusion_vector = pred[i]/true[i]

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    precision = true_positives/(true_positives+false_positives+epsilon)
    recall = true_positives/(true_positives+false_negatives+epsilon)
    f1 = 2*precision*recall/(precision+recall+epsilon)
    f1_final+=f1
    f.write("Label: {}, acc: {:.3f}, f1: {:.3f}\n".format(classes[i], acc, f1))
    f.write(str(confusion_matrix(true[i], pred[i])))
    f.write('\n')
   
  print(split+" sentiment f1 =",f1_final/14)
  return 0


print('Training...')
labelwise_metrics(train_out, train_true, 'TRAINING')
labelwise_metrics_senti(train_out, train_true, 'TRAINING')
print()
print('Validation...')
labelwise_metrics(val_out, val_true, 'VALIDATION')
labelwise_metrics_senti(val_out, val_true, 'VALIDATION')
print()
print('Test...')
labelwise_metrics(test_out, test_true, 'TESTING')
labelwise_metrics_senti(test_out_senti, test_true_senti, 'TESTING')
