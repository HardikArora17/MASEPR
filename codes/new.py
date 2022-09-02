from load_dataset import read
import pickle

t = read()

with open('data/dataframe_multitask.pkl','wb') as out:
  pickle.dump(t,out)



