import torchvision.transforms as transforms
from PIL import Image
import torch
from torchvision.models import resnet50
import numpy as np
import cv2

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del','nothing', 'space']

num_classes = 29
resnet_model = resnet50(pretrained=True)
in_features = resnet_model.fc.in_features
resnet_model.fc = torch.nn.Linear(in_features, num_classes)
resnet_model.load_state_dict(torch.load('trained_model_weights', map_location=torch.device('cpu')))

while (cap.isOpened()):
    ret, img = cap.read()

    predicted_class = ''
    fgmask = fgbg.apply(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Apply background mask to thresholded image
    fgmask[thresh == 0] = 0

    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    if len(contours) > 0:
        x, y, w, h = 50, 200, 400, 400
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        crop_img = img[y:y+h, x:x+w]

        data_transforms = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        final_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
        final_img = data_transforms(final_img)
        final_img = final_img.to('cpu')
        resnet_model.eval()
        prediction = resnet_model(final_img[None])
        index=torch.max(prediction, dim=1)[1]
        predicted_class = labels[index.item()]

        

    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, "Model Prediction : " + predicted_class, (x, y+h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Frame", img)
    k = cv2.waitKey(10)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
