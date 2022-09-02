
import os
import jsonlines
import nltk
nltk.download('punkt',quiet=True)
count_no=0
from tqdm import tqdm


class DataLoading():
    def __init__(self):
        super().__init__()
        
    def preprocess_text(self,string:str):
        string=string.lower()
        punctuations = '''!()-[]{};:'"\<>/?@#$^&*_~=+'''
        string=string.replace('�'," ")
        string=string.replace('\n',"")
        for x in string.lower(): 
            if x in punctuations: 
                string = string.replace(x, " ") 

        return string
    
    def general_statistics(self,data:dict):
        
        l=0   
        count_aspects={}
        n_p=0
        
        n_p=len(data.keys())
        
        
        for k in data.keys():
            l+=len(data[k])    
            
            for v1,v2 in data[k].items():
                for v3 in v2:
                    if(v3 not in count_aspects.keys()):
                        count_aspects[v3]=0
                    else:
                        count_aspects[v3]+=1
                        
                        
        
        print(f"Total number of different papers covered: {n_p}")
        print(f"Total number of sentences captured: {l}\n")
        
        print(f"ASPECT NAME  \t COUNT\n")
        
        for k,v in count_aspects.items():
            print(f"{k}  \t {v}")
            
        
        
    
    def load_dataset(self,input_path:str):
        global count_no
        
        data_path=r'C:\Users\Dell\Desktop\ReviewAdvisor\dataset\aspect_data'

        r_data={}
        counter=0
        
        with jsonlines.open(os.path.join(data_path,'review_with_aspect.jsonl')) as f:
            
            pbar=f.iter()
            
            for line in tqdm(pbar,desc='Loading Datasets'):
                id1=line['id']
                s=line['text']
                labels=line['labels']
                count=0

                r_d={}
                s1=[]

                enter=nltk.sent_tokenize(s)
                enter=list(set(enter))
                enter=[self.preprocess_text(s56) for s56 in enter]

                for k in labels:
                    if(k[2].startswith('summary')):
                        continue
                    
                    
                    a=k[2].split('_')
                    if(a[0]=='meaningful'):
                        a=(a[0],a[2])
                    else:
                        a=(a[0],a[1])
                        

                    h=self.preprocess_text(s[k[0]:k[1]]).strip()
                    h=nltk.sent_tokenize(h)

                    for h1 in h:
                        for sen in enter:
                            if(h1 in sen and sen is not None):
                                s1.append(sen)

                                if(sen in r_d.keys()):
                                    r_d[sen].append(a)
                                    r_d[sen]=list(set(r_d[sen]))

                                else:
                                    r_d[sen]=[]
                                    r_d[sen].append(a)
                                    r_d[sen]=list(set(r_d[sen]))


                                break


                s1=list(set(s1))
                assert len(s1)==len(r_d.keys())


                for e in enter:
                    if(e not in s1):
                        if(count_no>100000):
                            break
                        r_d[self.preprocess_text(e)]=[("no_aspect","no_sentimnet")]
                        count_no+=1

                try:
                    r_data[counter]=r_d
                    counter+=1
                except:
                    p=0
        
        self.general_statistics(r_data)
        
        return r_data
                    
                    
            
            
                    
        
        
            
            
            
            
            
        