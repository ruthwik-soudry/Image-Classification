# importing libraries
import torch
from torchvision import models, transforms, datasets
import glob
from torch import nn
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm

# function to prepare dataset
def prepare_dataset(resize_shape,path,samples):

  train_transforms = transforms.Compose([
      transforms.Resize(resize_shape),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])])

  test_transforms = transforms.Compose([
      transforms.Resize(resize_shape),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])])
  train_dataset = datasets.ImageFolder(train_data_path, transform=train_transforms)
  val_dataset = datasets.ImageFolder(train_data_path, transform=train_transforms)
  torch.manual_seed(1)

  num_train_samples = samples

  val_split = 0.2
  split = int(num_train_samples * val_split)
  indices = torch.randperm(num_train_samples)


  train_subset = torch.utils.data.Subset(train_dataset, indices[split:])
  val_subset = torch.utils.data.Subset(val_dataset, indices[:split])

  print('Total images in train dataset',len(train_subset))
  print('Total images in test dataset',len(val_subset))
  return train_subset,val_subset

# function to initialize and load the model
def model_loader(batch_size,train_subset,val_subset,model):
  batch_size = batch_size

  train_dataloader = torch.utils.data.DataLoader(
      dataset=train_subset, 
      batch_size=batch_size,
      shuffle=True)

  val_dataloader = torch.utils.data.DataLoader(
      dataset=val_subset,
      batch_size=batch_size,
      shuffle=False)
  classes = train_dataloader.dataset.dataset.classes
  resnet = model
  resnet = nn.DataParallel(resnet)
  for param in resnet.parameters():
    param.requires_grad = False
  in_features = resnet.module.fc.in_features
  fc = nn.Linear(in_features=in_features, out_features=len(classes))
  resnet.fc = fc
  params_to_update = []
  for name, param in resnet.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)

        
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(params_to_update, lr=0.001)
  return resnet,criterion,optimizer,train_dataloader,val_dataloader

# function to train the model
def train(model,criterion,optimizer,train_dataloader,test_dataloader,print_every,num_epoch,device):
    steps = 0
    train_losses, val_losses = [], []
    train_accuracy,val_accuracy=[], []
    model.to(device)
    for epoch in tqdm(range(num_epoch)):
        running_loss = 0
        correct_train = 0
        total_train = 0
        start_time = time()
        iter_time = time()
        
        model.train()
        for i, (images, labels) in enumerate(train_dataloader):
            steps += 1
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            output = model(images)
            loss = criterion(output, labels)

            correct_train += (torch.max(output, dim=1)[1] == labels).sum()
            total_train += labels.size(0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Logging
            if steps % print_every == 0:
                print(f'Epoch [{epoch + 1}]/[{num_epoch}]. Batch [{i + 1}]/[{len(train_dataloader)}].', end=' ')
                print(f'Train loss {running_loss / steps:.3f}.', end=' ')
                print(f'Train acc {correct_train / total_train * 100:.3f}.', end=' ')
                with torch.no_grad():
                    model.eval()
                    correct_val, total_val = 0, 0
                    val_loss = 0
                    for images, labels in test_dataloader:
                        images = images.to(device)
                        labels = labels.to(device)
                        output = model(images)
                        loss = criterion(output, labels)
                        val_loss += loss.item()
                        correct_val += (torch.max(output, dim=1)[1] == labels).sum()
                        total_val += labels.size(0)

                print(f'Val loss {val_loss / len(test_dataloader):.3f}. Val acc {correct_val / total_val * 100:.3f}.', end=' ')
                print(f'Took {time() - iter_time:.3f} seconds')
                iter_time = time()
              
                train_losses.append(running_loss / total_train)
                val_losses.append(val_loss / total_val)
                train_accuracy.append((correct_train / total_train) * 100)
                val_accuracy.append((correct_val / total_val) * 100)



        #print(f'Epoch took {time() - start_time}') 
        #torch.save(model, f'checkpoint_{correct_val / total_val * 100:.2f}')
        
    return model, train_losses, val_losses,train_accuracy,val_accuracy

# main function
if __name__=='__main__':
  train_data_path='asl_alphabet_train/asl_alphabet_train'
  train_subset,val_subset=prepare_dataset(64,train_data_path,10000)
  model=models.resnet50(pretrained=True)
  resnet,criterion,optimizer,train_dataloader,val_dataloader=model_loader(32,train_subset,val_subset,model)
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  #device='cpu'
  start=time()
  final_model,train_losses,val_losses,train_accuracy,val_accuracy=train(resnet,criterion,optimizer,train_dataloader,val_dataloader,150,5,device)
  end=time()
  print('On Device',device)
  print('Time taken',end-start)