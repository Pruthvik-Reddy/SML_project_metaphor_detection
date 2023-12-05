from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch
from metaphordataset import MetaphorDataset, MelBERTDataset
from models import BaseBERTClassifier, MelBERTCLassifier
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
            outputs = model(input_ids_1, attention_mask_1)
            loss = loss_function(outputs.squeeze(),labels.to(outputs.dtype))
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
                loss = loss_function(outputs.squeeze(),labels.to(outputs.dtype))
                dev_loss+=loss.item()
                
                
        print("Dev loss for epoch {} is {}".format(epoch+1,dev_loss/len(val_loader)))
        
    model.eval()


def melbert_model(texts,labels,target,target_index):


    data = list(zip(texts, labels, target,target_index))

    # Split the data
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    # Unzip the split data
    train_texts, train_labels, train_target, train_target_index = zip(*train_data)
    val_texts, val_labels, val_target, val_target_index = zip(*val_data)

    tokenizer=BertTokenizer.from_pretrained("bert-base-uncased")
    
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    train_target_encodings=tokenizer(train_target, truncation=True, padding=True)
    val_target_encodings=tokenizer(val_target, truncation=True, padding=True)


    train_dataset = MelBERTDataset(train_encodings, train_labels,train_target_encodings, train_target_index)
    val_dataset = MelBERTDataset(val_encodings, val_labels, val_target_encodings,val_target_index)

    batch_size=12
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader= DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    num_classes = 2  
    model = MelBERTCLassifier(num_classes=num_classes)

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
            input_ids_2= batch['input_ids_2'].to(device)
            attention_mask_2= batch['attention_mask_2'].to(device)
            
            labels = batch['labels'].to(device)
            target_index=batch["target_index"].to(device)
            outputs = model(input_ids_1, attention_mask_1, input_ids_2,attention_mask_2,target_index)
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
                input_ids_2= batch['input_ids_2'].to(device)
                attention_mask_2= batch['attention_mask_2'].to(device)
            
                labels = batch['labels'].to(device)
                target_index=batch["target_index"].to(device)
            
                outputs = model(input_ids_1, attention_mask_1,input_ids_2,attention_mask_2,target_index)
                loss = loss_function(outputs.squeeze(),labels.to(torch.float))
                dev_loss+=loss.item()
                
                
        print("Dev loss for epoch {} is {}".format(epoch+1,dev_loss/len(val_loader)))
        
    model.eval()

    



    
    
    
