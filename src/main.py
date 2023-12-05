from data_loader import get_data,get_texts_labels,get_data_for_melbert
import pandas as pd
from train import base_bert_model,melbert_model

if __name__=="__main__":
    data=get_data()
    #texts,labels=get_texts_labels(data)

    #base_bert_model(texts,labels)

    #texts,labels,target,target_index=get_data_for_melbert(data)
    #melbert_model(texts,labels,target,target_index)
    
    