import custom_model
import transfer_learning_pytorch
import model_train_torch
import preparing_dataset_torch
import torch
import warnings
warnings.filterwarnings('ignore')

num_classes=29

path='../asl_alphabet_train/asl_alphabet_train/'
train_loader, val_loader = preparing_dataset_torch.preparing_dataset(path)
print('Data Loaded')

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device',device)
#curr_model = custom_model.MySeqModel(num_classes)
curr_model=transfer_learning_pytorch.pretrained_model(num_classes)
curr_model.to(device)
print('Model Loaded')

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(curr_model.parameters(), lr=0.003)

print('Training Started')
model_train_torch.model_train(10,train_loader, val_loader,curr_model,
criterion,optimizer,device)


