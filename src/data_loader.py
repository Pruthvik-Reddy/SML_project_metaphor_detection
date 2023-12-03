import pandas as pd

metaphor_id={
    0: "road",
1: "candle",
2: "light",
3: "spice",
4: "ride",
5: "train",
6: "boat"
}

def add_target_index_and_target_word_to_data(data):
    target=[]
    target_index=[]
    for index,row in data.iterrows():
        met_id=row["metaphorID"]
        text=row["text"]
        text=text.lower()
        target_word=metaphor_id[met_id]
        words=text.split()
        word_index=words.index(target_word)
        target.append(target_word)
        target_index.append(word_index)

    print(len(target))
    print(len(target_index))
    print(data.shape)

def get_data():
    data=pd.read_csv("train.csv")

    add_target_index_and_target_word_to_data(data)
    
