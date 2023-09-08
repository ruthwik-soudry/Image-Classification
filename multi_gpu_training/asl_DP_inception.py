import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import models
import torchvision.utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline

if __name__=="__main__":
    # transforming and normalizing the image data
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(299),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_data = dsets.ImageFolder('asl_alphabet_train/asl_alphabet_train/', train_transform)
    val_dataset = datasets.ImageFolder('asl_alphabet_train/asl_alphabet_train/', transform=train_transform)
    torch.manual_seed(1)

    num_train_samples = 10000

    val_split = 0.2
    split = int(num_train_samples * val_split)
    indices = torch.randperm(num_train_samples)

    train_subset = torch.utils.data.Subset(train_data, indices[split:])
    val_subset = torch.utils.data.Subset(val_dataset, indices[:split])
    print('Total images in train dataset',len(train_subset))
    print('Total images in test dataset',len(val_subset))

    batch_size = 5

    # subsetting data into different batches
    train_loader = DataLoader(train_subset,
                              batch_size=batch_size,
                              shuffle=True)

    test_loader = DataLoader(val_subset, 
                             batch_size=batch_size,
                             shuffle=True)


    # function to display images from tensors
    def imshow(img, title):
        img = torchvision.utils.make_grid(img, normalize=True)
        npimg = img.numpy()
        fig = plt.figure(figsize = (5, 15))
        plt.imshow(np.transpose(npimg,(1,2,0)))
        plt.title(title)
        plt.axis('off')
        plt.show()

    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    imshow(images, [train_data.classes[i] for i in labels])

    # downloading the model
    model = models.inception_v3(pretrained=True)

    model.aux_logits = False

    for parameter in model.parameters():
        parameter.requires_grad = False

    # re-defining the fc layer for the model
    classes = train_dataloader.dataset.dataset.classes
    model.fc = nn.Linear(model.fc.in_features, len(classes))

    # calling DataParallel class on the model
    model = model.cuda()
    model = nn.DataParallel(model)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    num_epochs = 10

    # training the model and capturing start and end time
    start=time()
    for epoch in range(num_epochs):
        total_batch = len(train_data)//batch_size

        for i, (batch_images, batch_labels) in enumerate(train_loader):

            X = batch_images.cuda()
            Y = batch_labels.cuda()

            pre = model(X)
            cost = loss(pre, Y)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            if (i+1) % 500 == 0:
                print('Epoch [%d/%d], lter [%d/%d] Loss: %.4f'
                     %(epoch+1, num_epochs, i+1, total_batch, cost.item()))

    print('Time Taken ',time()-start)

    # evaluating model's performance
    model.eval()

    correct = 0
    total = 0

    for images, labels in test_loader:

        images = images.cuda()
        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum()

    print('Accuracy of test images: %f %%' % (100 * float(correct) / total))