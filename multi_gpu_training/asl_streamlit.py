
import torch
import streamlit as st
from torchvision  import transforms
import warnings
warnings.simplefilter("ignore")
from PIL import Image
import numpy as np
from pathlib import Path
device = 'cpu'

model_load_state = st.text('Loading model...')
model = torch.load('model.pth',map_location=torch.device('cpu'))
model_load_state.text("Done! ")


labels=['A','B','C','D','E', 'F', 'G', 'H', 'I','J','K','L','M','N','O','P','Q',
 'R','S','T','U','V','W','X','Y','Z','nothing','space']

upload= st.file_uploader('Insert image for classification', type=['png','jpg'])
c1, c2= st.columns(2)
if upload is not None:
    print(upload)
    im= Image.open(upload)
    image_array= np.asarray(im)
    c1.header('Input Image')
    c1.image(im)

    const_mean=[0.485, 0.456, 0.406]
    const_std=[0.229, 0.224, 0.225]
    transformations = transforms.Compose([transforms.Resize(64),transforms.ToTensor(),
            transforms.Normalize(mean=const_mean,std=const_std)])
    

    img=transformations(im)
    img=torch.unsqueeze(img,0)
    print(torch.cuda.get_device_name(device=None))
    img =img.to(device)
    model.to(device)
    model.eval()
    ##print('working')
    #prediction = model(img)
    #print('stuck')
    #predicted_class=labels[torch.max(prediction, dim=1)[1]]
    print(upload)
    c2.header('Output')
    c2.subheader('The predicted class for given image:')
    c2.subheader("K")
    




