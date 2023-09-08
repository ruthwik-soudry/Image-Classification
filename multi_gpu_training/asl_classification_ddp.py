import torch
import os
import sys
from torchvision import models, transforms, datasets
from torch import nn
from time import time
import torch.distributed as distributed
import torch.multiprocessing as mp
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

#function to transform input images to tensor and normalize them further
def transform_split_dataset(resize_shape,path,samples):

  #transforming the dataset i.e. resizing and recenntering it
  torch.manual_seed(45)
  const_mean=[0.485, 0.456, 0.406]
  const_std=[0.229, 0.224, 0.225]
  transformations = transforms.Compose([transforms.Resize(resize_shape),transforms.ToTensor(),
      transforms.Normalize(mean=const_mean,std=const_std)])

  train_dataset = datasets.ImageFolder(path, transform=transformations)
  test_dataset = datasets.ImageFolder(path, transform=transformations)

  #performing random train test split
  test_split = 0.2
  pivot = int(samples * test_split)
  indices = torch.randperm(samples)

  train_subset = torch.utils.data.Subset(train_dataset, indices[pivot:])
  test_subset = torch.utils.data.Subset(test_dataset, indices[:pivot])

  return train_subset,test_subset

#function to initialized the DDP model and re-define final fc layer 
def model_loader(batch_size,train_data,test_data,model,rank,name):
  #loading the data in batches using dataloader 
  train_dataloader = torch.utils.data.DataLoader(dataset=train_data, 
                    batch_size=batch_size,shuffle=True,num_workers=4)
  test_dataloader = torch.utils.data.DataLoader( dataset=test_data,
                              batch_size=batch_size,shuffle=True,num_workers=4)

  classes = 29
  #defining the models 
  train_model = model.to(rank)
  
  #Using DDP to for parallel processing of models
  parallel_model= DDP(train_model, device_ids=[rank])
    
  for param in parallel_model.parameters():
    param.requires_grad = False
  
  #the final layer of models used is different for some models
  if name=='resnet' or name=='googlenet':
      fc = nn.Linear(in_features=parallel_model.module.fc.in_features, out_features=classes)
      parallel_model.module.fc = fc
  else:
     in_features = parallel_model.module.classifier[-1].in_features
     parallel_model.module.classifier[-1] = nn.Linear(in_features, classes)
  #defining which parameter to update
  updated_parameters = []
  for name, grad in parallel_model.named_parameters():
    if grad.requires_grad == True:
        updated_parameters.append(grad)

  #defining the loss function and optimizer      
  loss_function = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(updated_parameters, lr=0.0003)
  return parallel_model,loss_function,optimizer,train_dataloader,test_dataloader

#calculating accuracy
def evaluation_accuracy(dataloader,model,rank):
    total_datapoints, true_positives = 0, 0
    for data in dataloader:
        inputs, labels = data
        inputs=inputs.to(rank)
        labels=labels.to(rank)
        outputs = model(inputs)
        val, pred = torch.max(outputs.data, 1)
        total_datapoints += labels.size(0)
        true_positives += (pred == labels).sum().item()
    return (true_positives / total_datapoints)*100

#function to train the model
def train_model(model,criterion,optimizer,train_dataloader,test_dataloader,num_epoch,rank):
    train_losses=[]
    train_accuracy_list,val_accuracy_list=[], []
    model.to(rank)
    start=time()
    for epoch in (range(num_epoch)):
        model.train()
        start_epoch=time()
        for i, (images, labels) in enumerate(train_dataloader):
            images = images.to(rank)
            labels = labels.to(rank)

            output = model(images)
            loss = criterion(output, labels)
            loss_value=loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_accuracy= evaluation_accuracy(train_dataloader,model,rank) 
        test_accuracy= evaluation_accuracy(test_dataloader,model,rank)
        train_losses.append(loss_value)
        print('Epoch: {}/{} Loss: {} Test acc: {} Train acc: {} Time : {} seconds'.format(epoch+1, num_epoch, loss_value,round(test_accuracy,4),round(train_accuracy,4), round(time()-start_epoch,4)))
        train_accuracy_list.append(train_accuracy)
        val_accuracy_list.append(test_accuracy)
        
        
    return model, train_losses,train_accuracy_list,val_accuracy_list

#function to call training for multiple models
def all_process(rank, world_size,model_name):
      
    train_data_path='asl_alphabet_train/asl_alphabet_train'
    train_subset,val_subset=transform_split_dataset(32,train_data_path,2000)

    distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    
    model_name=model_name
    if model_name=='resnet':
        model=models.resnet50(pretrained=True)
    elif model_name=='alexnet':
        model=models.alexnet(pretrained=True)
    elif model_name=='googlenet':
        model=models.googlenet(pretrained=True)
    else:
        model=models.vgg19(pretrained=True)
   
        
    resnet,criterion,optimizer,train_dataloader,val_dataloader=model_loader(512,train_subset,val_subset,model,rank,model_name)
    
    
    final_model, train_losses,train_accuracy_list,val_accuracy_list=train_model(resnet,criterion,optimizer,train_dataloader,val_dataloader,5,rank)
    
      

def main(num_gpus,model):
    start=time()
    print('Training with ',model)
    world_size=num_gpus
    print('GPU ',world_size)
    mp.spawn(all_process,
        args=(world_size,model),
        nprocs=world_size,
        join=True)

    
    total_time=time()-start
    print('*'*50)
    print('Total Time taken - '+str(round(total_time,2))+' seconds.')
    print('*'*50)

if __name__ == "__main__":
    
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    model=sys.argv[1]
    num_gpus = int(sys.argv[2])
    
    main(num_gpus,model)
    


