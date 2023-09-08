# Multi-GPU Image Classification: ASL Translation

## Dataset Introduction

Imagine communicating without words. This is the essence of American Sign Language (ASL), a key communication medium for the Deaf community in the USA. While ASL is a rich, visual language, technology offers the opportunity to transform this visual medium into audible speech.

Our project uses a Deep Learning Model to translate ASL signs from images. To achieve optimal performance, we've embraced DataParallelism and DistributedDataParallelism techniques.

![ASL Sample Image](https://user-images.githubusercontent.com/99056351/215807223-73927ea5-bead-4a96-8dbe-ab9d4bed5483.png)

Our training dataset boasts 87,000 images, all 200x200 pixels. The images are categorized into 29 classes: 26 for letters A-Z, and 3 others representing SPACE, DELETE, and NOTHING. These three categories enhance real-time application use and classification. The test dataset, however, contains 29 images, pushing users to utilize real-world images for testing.

![Dataset Image](https://user-images.githubusercontent.com/99056351/215807500-b7d0e6e9-cab2-4ea9-ad8a-b889d1bcab03.png)

## Dataset Pre-processing

Raw image data requires transformation for efficient model consumption. We’ve crafted a `prepare_dataset()` function for this. The images undergo tensor transformation and normalization, ensuring data consistency. Subsequently, for optimal training, we segregate the data into training and validation sets.

![Preprocessing Image](https://user-images.githubusercontent.com/99056351/215807556-ab75d6b5-1848-42e2-8dca-037f2990cd7c.png)

## Model Architecture

Our endeavor uses two parallel techniques: DataParallel and DistributedDataParallel.

1. **DataParallel Method:** Our `model_loader()` function, contingent on batch size and data subsets, facilitates efficient data sampling and loading. After model initialization, we employ PyTorch’s DataParallel class. The model’s output layer is subsequently adjusted to match the number of our categories.

   ![DataParallel Image](https://user-images.githubusercontent.com/99056351/215808127-0958c7ee-0c31-41a1-b3fe-730263f962c6.png)

2. **DistributedDataParallel Method:** Here, we first distribute the data using PyTorch’s DataLoader function. By tweaking the `num_workers` parameter, we control data partitioning across multiple GPUs. Once our model is in place, we adopt the DistributedDataParallel class and modify the model's final layer to align with our requirements.

## Training Process

The heart of our project lies in the training. Each model's training approach is unique.

1. **DataParallel Training (on 10k samples):** We've devised a `train()` helper function. This function takes in hyperparameters and other required arguments from the primary calling function. The training procedure is intuitive, and for analysis, we clock the time taken across different GPU counts.

   ![Training DataParallel Image](https://user-images.githubusercontent.com/99056351/215808323-4b94a79f-c10b-4dda-80f1-6e5a45952fe8.png)

Our analysis of the varying training times across different GPU counts, particularly with the ResNet model, is visualized below:

![Training Analysis Chart](https://user-images.githubusercontent.com/99056351/215808462-c3286d90-54a5-4925-b2c9-becb5d42fd01.png)