This is the official repository for the paper titled:

# MASEPR: A Sentiment-leveraged Multi-taskDeep Neural Architecture for Aspect Extraction from Peer Reviews

# DEMO VIDEO OF OUR INTERFACE
Please open the link below to watch the demo:

https://user-images.githubusercontent.com/54908862/143673155-25b50c7e-5aad-4699-ab8c-dc01f1eb098a.mp4


# Experimental Setup

We use Pytorch as the DL framework and pre-trained transformer models from  https://huggingface.co for implementation.

We split our dataset into 80 \% train,5\% validation and 15 \% test sets.
During validation, we experiment with different network configurations and for optimal performance we take 
* Batch size=32, 
* activation function=ReLu
* dropout= 0.5 
* Learning rate= 1e-3 
* Loss function = Binary Crossentropy 
* Epoch= 15.
* We used Adam optimizer with  a weight\_decay=1e-3( for avoiding overfitting) for training. 
We perform all our experiments on a GPU (GeForce RTX 2080) with 16 GB of memory.

If you wish to run our codes
Clone the github repo and follow the steps as mentioned below

    cd MASEPR
    
Install the requirements for our code

    pip install -r requirements.txt

For our experimental needs we reconstucted the ASAP dataset. The reconstucted data files can be obtained by running the code below.

    1) cd data
    2) wget https://drive.google.com/file/d/1nJdljy468roUcKLbVwWUhMs7teirah75/view?usp=sharing
    3) cd ..
    4) python codes/new.py
    5) python codes/create_labels.py
    
 After running this code, a folder named modified_data will be created with three files in it, namely
 
 1) dataframe_multitask.pkl        - which contains the sentence wise annotations
 2) onehot_sentiment_multitask.pkl - which contains the one hot vectors for the sentiments labels
 3) onehot_aspect_multitask.pkl    - which contains the one hot vectors for the aspects labels


You can train our model with any transformer model using the command below

    python codes/multitask_bert.py
    
If you want to change the transformer model used for generating representations , just change the parameter named model_name in the file multitask_bert.py
The default model is bert-based-uncased ,which gives the best results on the test dataset.



For example if you want to want to use scibert you can change the model_name parameter as follows:

    model_name="scibert-scivocab-uncased"
    
    
After running the code , 

* your model  will get saved in the ckpt folder.

* Labelwise results for both aspects and sentiments on training , validation and test set get stored in the results folder.

* Outputs, attentions weights on the test dataset get stored in the outputs folder. 




<!-- 
Sample data:

ids           | sentences                                         | aspects
------------- | -------------                                     | -----------
0             | the issue researched in this work is of signif... | [motivation]
1             | is this comparison fair                           | [meaningful]


![alt text](https://github.com/HardikArora17/MASPR-PAKDD-2021/blob/ce9bc5e21925c8dc45967d00c4ae26b95ab5f035/loss_bert_multi.png)
 -->
