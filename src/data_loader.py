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


plurals={
0: "roads",
1: "candles",
2: "lights",
3: "spices",
4: "rides",
5: "trains",
6: "boats"

}


def add_target_index_and_target_word_to_data(data):
    target=[]
    target_index=[]
    for index,row in data.iterrows():
        met_id=row["metaphorID"]
        text=row["text"]
        text=text.lower()
        target_word=metaphor_id[met_id]
        plural_word=plurals[met_id]
        words=text.split()
        if target_word in words:

            word_index=words.index(target_word)
            target.append(target_word)
            target_index.append(word_index)
        else:
            word_index=words.index(plural_word)
            target.append(plural_word)
            target_index.append(word_index)

    data["target"]=target
    data["target_index"]=target_index
    label_counts = data['label_boolean'].value_counts()

    print(label_counts)
    
    data['label'] = data['label_boolean'].apply(lambda x: 1 if x else 0)
    label_counts = data['label'].value_counts()

    print(label_counts)
    return data

    
def get_data():
    data=pd.read_csv("../data/train.csv")

    data=add_target_index_and_target_word_to_data(data)
    return data
    
