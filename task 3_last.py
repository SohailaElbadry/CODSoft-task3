# Importing libraries
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Loading Data
df = pd.read_csv('./Churn_Modelling.csv')
print(df.head())
print(df.isnull().sum())
print(df.describe())

# Preprocessing Data
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Label Encoding
label_encoder = LabelEncoder()
df['Geography'] = label_encoder.fit_transform(df['Geography'])
df['Gender'] = label_encoder.fit_transform(df['Gender'])

# Feature Scaling
scaler = StandardScaler()
df[['CreditScore', 'Balance', 'EstimatedSalary']] = scaler.fit_transform(df[['CreditScore', 'Balance', 'EstimatedSalary']])

# Splitting into X and y
X = df.drop('Exited', axis=1).values
y = df['Exited'].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Neural Network Model
class NeuralNetwork(nn.Module):
    def __init__(self, len_f):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(len_f, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2)
        self.norm1 = nn.BatchNorm1d(64)
        self.norm2 = nn.BatchNorm1d(128)
        self.norm3 = nn.BatchNorm1d(64)

    def forward(self, X):
        out = F.relu(self.norm1(self.fc1(X)))
        out = F.relu(self.norm2(self.fc2(out)))
        out = F.relu(self.norm3(self.fc3(out)))
        out = F.softmax(self.fc4(out))
        return out

# Creating DataLoader
X_shape = X_train.shape[1]
train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# Training the model
NN_model = NeuralNetwork(X_shape)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(NN_model.parameters(), lr=0.001)

n_epochs = 5
line_dash = '-' * 40

for epoch in range(n_epochs):
    total_accuracy = 0
    total_loss = 0
    print(line_dash)
    print(f'Epoch: {epoch+1}')
    for data, label in train_loader:
        data = data.view(-1, X_shape)
        label = label.long()

        # Forward
        out = NN_model(data)
        loss = criterion(out, label)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculating accuracy
        accuracy = (torch.argmax(out, dim=1) == label).float().mean()

        total_accuracy += accuracy / len(train_loader)
        total_loss += loss / len(train_loader)
    
    print(f'Train Accuracy: {total_accuracy}\t Train Loss: {total_loss}')
    
    test_accuracy = 0
    test_loss = 0

    with torch.no_grad():
        for data2, label2 in test_loader:
            data2 = data2.view(-1, X_shape)
            label2 = label2.long()
            out2 = NN_model(data2)
            loss = criterion(out2, label2)

            accuracy2 = (torch.argmax(out2, dim=1) == label2).float().mean()
            test_accuracy += accuracy2 / len(test_loader)
            test_loss += loss / len(test_loader)
    
    print(f'Test Accuracy: {test_accuracy}\t Test Loss: {test_loss}')
    print(line_dash)
    print('\n')
