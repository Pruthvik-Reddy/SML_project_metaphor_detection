from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch
from metaphordataset import MetaphorDataset
from models import BaseBERTClassifier
from torch.utils.data import DataLoader
import torch.nn as nn

def base_bert_model(texts,labels):
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=.2)
    tokenizer=BertTokenizer.from_pretrained("bert-base-uncased")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    train_dataset = MetaphorDataset(train_encodings, train_labels)
    val_dataset = MetaphorDataset(val_encodings, val_labels)

    batch_size=12
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader= DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    num_classes = 2  
    model = BaseBERTClassifier(num_classes=num_classes)

    learning_rate=0.01

    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    
    model.to(device)
    loss_function = nn.BCEWithLogitsLoss()

    epochs=10

    training_loss=0
    val_loss=0
    for epoch in range(epochs):
        model.train()
        training_loss=0
        for batch in train_loader:
            optim.zero_grad()
            input_ids_1= batch['input_ids'].to(device)
            attention_mask_1= batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids_1, attention_mask_1=attention_mask_1)
            loss = loss_function(outputs.squeeze(),labels.to(torch.float))
            loss.backward()
            optim.step()
            training_loss+=loss.item()
            
        avg_training_loss=training_loss/len(train_loader)
        print("Training loss for epoch {} is {}".format(epoch+1,avg_training_loss))
        
        
        model.eval()
        dev_loss=0
        with torch.no_grad():
            
            for batch in val_loader:
                input_ids_1= batch['input_ids'].to(device)
                attention_mask_1= batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids_1, attention_mask_1=attention_mask_1)
                loss = loss_function(outputs.squeeze(),labels.to(torch.float))
                dev_loss+=loss.item()
                
                
        print("Dev loss for epoch {} is {}".format(epoch+1,dev_loss/len(val_loader)))
        
    model.eval()


def melbert_model(texts,labels,target,target_index):
    pass
    



    
    
    
