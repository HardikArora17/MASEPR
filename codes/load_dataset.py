import pandas as pd
from reader import DataLoading 

def read():
    obj=DataLoading()
    input_path="data/dataset/aspect_data/"

    data_cmu=pd.DataFrame()

    data=obj.load_dataset(input_path)
    sentences=[]
    aspect=[]
    ids=[]

    c=0

    for k in data.keys():
        for v1,v2 in data[k].items():
            sentences.append(v1)
            aspect.append(v2)
            ids.append(c)
            c+=1


    data_cmu['ids']=ids
    data_cmu['sentences']=sentences
    data_cmu['aspects']=aspect
    
    data_cmu=data_cmu.sample(frac=1)

    return data_cmu
