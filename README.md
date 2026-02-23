# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
Include the Problem Statement and Dataset.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Import the required libraries (torch, torchvision, torch.nn, torch.optim) and load the image dataset with necessary preprocessing like normalization and transformation.

### STEP 2:

Split the dataset into training and testing sets and create DataLoader objects to feed images in batches to the CNN model.

### STEP 3:

Define the CNN architecture using convolutional layers, ReLU activation, max pooling layers, and fully connected layers as implemented in the CNNClassifier class.

### STEP 4:

Initialize the model, define the loss function (CrossEntropyLoss), and choose the optimizer (Adam) for training the network.

### STEP 5:

Train the model using the training dataset by performing forward pass, computing loss, backpropagation, and updating weights for multiple epochs.

### STEP 6:

Evaluate the trained model on test images and verify the classification accuracy for new unseen images.

## PROGRAM

### Name:

### Register Number:

```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.conv3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1=nn.Linear(128*3*3,128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,10)
    def forward(self, x):
        x=self.pool(torch.relu(self.conv1(x)))
        x=self.pool(torch.relu(self.conv2(x)))
        x=self.pool(torch.relu(self.conv3(x)))
        x=x.view(x.size(0),-1)
        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        x=self.fc3(x)
        return x
# Initialize model, loss function, and optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)
## Step 3: Train the Model
def train_model(model, train_loader, num_epochs=3):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('Name:Dharini.S')
        print('Register Number:212224040072')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

```

### OUTPUT

## Training Loss per Epoch

<img width="391" height="207" alt="image" src="https://github.com/user-attachments/assets/7f899475-9f03-4800-a0f7-b797b6b3530e" />

## Confusion Matrix

<img width="709" height="608" alt="image" src="https://github.com/user-attachments/assets/48655a56-6742-4727-a770-29aa707e605d" />

## Classification Report

<img width="605" height="430" alt="image" src="https://github.com/user-attachments/assets/4e201553-07ed-4a71-b94a-2d8e44e07c3a" />

### New Sample Data Prediction

<img width="576" height="630" alt="image" src="https://github.com/user-attachments/assets/42387fe2-36e8-40e1-bdaa-92267f363bd0" />

<img width="582" height="630" alt="image" src="https://github.com/user-attachments/assets/f928e158-167b-4ffb-b0a3-d82ceb186218" />


## RESULT

The Convolutional Neural Network (CNN) model was successfully trained and achieved good classification performance on the given image dataset. 
