import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from Preprocessing import *
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix



def Accuracy(Predicted, Labels):
    return (Predicted == Labels).sum().item() / len(Labels)

def augment_batch(images, transforms):
    augmented_images = []

    for image in images:
        augmented_image = transforms(image)
        augmented_images.append(augmented_image)

    # to tensor
    augmented_images = torch.stack(augmented_images)

    return augmented_images

def train_model(model, dataloader_train,dataloader_test,num_epochs,criterion, optimizer,device,augment=False,transforms=None,train_on="labels"):

    model = model.to(device)
    Train_Losses_Per_Batch = []
    Test_Losses_Per_Batch = []
    Train_Accuracy_Per_Batch = []
    Test_Accuracy_Per_Batch = []

    Train_Losses_Per_Epoch = []
    Test_Losses_Per_Epoch = []
    Train_Accuracy_Per_Epoch = []
    Test_Accuracy_Per_Epoch = []


    ProgressBar_Nb_Epochs = tqdm(range(num_epochs), desc="Epochs")

    for epoch in ProgressBar_Nb_Epochs:
        Train_Loss_Epoch = 0
        Test_Loss_Epoch = 0
        Train_Accuracy_Epoch = 0
        Test_Accuracy_Epoch = 0

        ProgressBar_Nb_Batches = tqdm(dataloader_train, desc="Batches Train", leave=False)

        for nb_batch, (input_images, types, sublabels, labels) in enumerate(ProgressBar_Nb_Batches):
            torch.cuda.empty_cache()
            if train_on == "types":
                labels = types
            elif train_on == "sublabels":
                labels = sublabels
            elif train_on == "labels":
                labels = labels
            

            if augment and transforms is not None:
                input_images = augment_batch(input_images, transforms)
            
            input_images = input_images.to(device)
            labels = labels.to(device)
            labels = labels.long()


            optimizer.zero_grad()

            outputs = model(input_images)
            outputs = outputs.to(device)
            outputs = outputs.float()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


            Running_Loss= loss.item()
            Train_Loss_Epoch += Running_Loss * input_images.size(0)
            Train_Losses_Per_Batch.append(Running_Loss)

            _, predicted = torch.max(outputs, 1)

            Running_Accuracy = Accuracy(predicted, labels)
            Train_Accuracy_Epoch += Running_Accuracy * input_images.size(0)
            Train_Accuracy_Per_Batch.append(Running_Accuracy)

            ProgressBar_Nb_Batches.set_description(f"Epoch {epoch+1}/{num_epochs}, Batch {nb_batch+1}/{len(dataloader_train)}, Running Loss: {Running_Loss :.4f}, Running Accuracy: {Running_Accuracy :.4f}")
        Train_Loss_Epoch = Train_Loss_Epoch / len(dataloader_train.dataset)
        Train_Accuracy_Epoch = Train_Accuracy_Epoch / len(dataloader_train.dataset)

        ProgressBar_Nb_Batches_Test = tqdm(dataloader_test, desc="Batches Test", leave=False)

        with torch.no_grad():
            for nb_batch, (input_images, types, sublabels, labels) in enumerate(ProgressBar_Nb_Batches_Test):
                
                if train_on == "types":
                    labels = types
                elif train_on == "sublabels":
                    labels = sublabels
                elif train_on == "labels":
                    labels = labels



                input_images = input_images.to(device)
                labels = labels.to(device)
                labels = labels.long()


                outputs = model(input_images)
                outputs = outputs.to(device)
                outputs = outputs.float()


                loss = criterion(outputs, labels)

                Running_Loss= loss.item()
                Test_Loss_Epoch += Running_Loss * input_images.size(0)
                Test_Losses_Per_Batch.append(Running_Loss)

                _, predicted = torch.max(outputs, 1)

                Running_Accuracy = Accuracy(predicted, labels)
                Test_Accuracy_Epoch += Running_Accuracy * input_images.size(0)
                Test_Accuracy_Per_Batch.append(Running_Accuracy)

                ProgressBar_Nb_Batches_Test.set_description(f"Epoch {epoch+1}/{num_epochs}, Batch {nb_batch+1}/{len(dataloader_test)}, Running Loss: {Running_Loss :.4f}, Running Accuracy: {Running_Accuracy :.4f}")

        Test_Loss_Epoch = Test_Loss_Epoch / len(dataloader_test.dataset)
        Test_Accuracy_Epoch = Test_Accuracy_Epoch / len(dataloader_test.dataset)

        Train_Losses_Per_Epoch.append(Train_Loss_Epoch)
        Test_Losses_Per_Epoch.append(Test_Loss_Epoch)

        Train_Accuracy_Per_Epoch.append(Train_Accuracy_Epoch)
        Test_Accuracy_Per_Epoch.append(Test_Accuracy_Epoch)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {Train_Loss_Epoch :.4f}, Test Loss: {Test_Loss_Epoch :.4f}, Train Accuracy: {Train_Accuracy_Epoch :.4f}, Test Accuracy: {Test_Accuracy_Epoch :.4f}")
        # torch.cuda.empty_cache()

    return Train_Losses_Per_Batch, Test_Losses_Per_Batch, Train_Accuracy_Per_Batch, Test_Accuracy_Per_Batch, Train_Losses_Per_Epoch, Test_Losses_Per_Epoch, Train_Accuracy_Per_Epoch, Test_Accuracy_Per_Epoch




def test_model(model, dataloader_test,device,train_on="labels"):
    model = model.to(device)
    model.eval()

    Predicted = []
    Labels = []

    ProgressBar_Nb_Batches_Test = tqdm(dataloader_test, desc="Batches Test", leave=False)

    with torch.no_grad():
        for nb_batch, (input_images, types, sublabels, labels) in enumerate(ProgressBar_Nb_Batches_Test):
            
            if train_on == "types":
                labels = types
            elif train_on == "sublabels":
                labels = sublabels
            elif train_on == "labels":
                labels = labels

            input_images = input_images.to(device)
            labels = labels.to(device)
            labels = labels.long()

            outputs = model(input_images)
            outputs = outputs.to(device)
            outputs = outputs.float()

            _, predicted = torch.max(outputs, 1)

            Predicted.extend(predicted.cpu().numpy())
            Labels.extend(labels.cpu().numpy())

            ProgressBar_Nb_Batches_Test.set_description(f"Batch {nb_batch+1}/{len(dataloader_test)}")

    return Predicted, Labels

def save_model(model,path,save_architechture=True):
    if save_architechture:
        torch.save(model, path)
    else:
        torch.save(model.state_dict(), path)

def load_model(path,load_architechture=True,model_class=None):
    if load_architechture:
        return torch.load(path)
    elif model_class is not None:
        model = model_class()
        model.load_state_dict(torch.load(path))
        return model
    else:
        print("Please provide a model class to load the model, otherwise set load_architechture to True")







def plot_losses_per_epoch(Train_Losses_Per_Epoch, Test_Losses_Per_Epoch,figsize=(12,6),savefig=False,path=None):
    fig,axes = plt.subplots(1,1,figsize=figsize)

    axes.plot(Train_Losses_Per_Epoch, label="Train Loss")


    Last_Train_Loss = Train_Losses_Per_Epoch[-1]
    Last_Test_Loss = Test_Losses_Per_Epoch[-1]


    axes.set_title(f"Evolution of Losses per Epoch - Last Train Loss: {Last_Train_Loss:.3f} - Last Test Loss: {Last_Test_Loss:.3f}")

    axes.set_xlabel("Epochs")
    axes.set_ylabel("Loss")

    axes.plot(Test_Losses_Per_Epoch, label="Test Loss")
    axes.legend()

    if savefig:
        plt.savefig(path)

    plt.show()



def plot_accuracy_per_epoch(Train_Accuracy_Per_Epoch, Test_Accuracy_Per_Epoch,figsize=(12,6),savefig=False,path=None):
    fig,axes = plt.subplots(1,1,figsize=figsize)

    axes.plot(Train_Accuracy_Per_Epoch, label="Train Accuracy")


    Last_Train_Accuracy = Train_Accuracy_Per_Epoch[-1]
    Last_Test_Accuracy = Test_Accuracy_Per_Epoch[-1]

    axes.set_title(f"Evolution of Accuracy per Epoch,  Last Train Accuracy: {Last_Train_Accuracy:.3f} - Last Test Accuracy: {Last_Test_Accuracy:.3f}")

    axes.set_xlabel("Epochs")
    axes.set_ylabel("Accuracy")

    axes.plot(Test_Accuracy_Per_Epoch, label="Test Accuracy")
    axes.legend()

    if savefig:
        plt.savefig(path)

    plt.show()


def plot_losses_per_batch(Train_Losses_Per_Batch, Test_Losses_Per_Batch,figsize=(12,6),savefig=False,path=None):
    fig,axes = plt.subplots(1,1,figsize=figsize)

    axes.plot(Train_Losses_Per_Batch, label="Train Loss")

    axes.set_title("Evolution of Losses per Batch")

    axes.set_xlabel("Batches")
    axes.set_ylabel("Loss")

    axes.plot(Test_Losses_Per_Batch, label="Test Loss")
    axes.legend()

    if savefig:
        plt.savefig(path)

    plt.show()


def plot_accuracy_per_batch(Train_Accuracy_Per_Batch, Test_Accuracy_Per_Batch,figsize=(12,6),savefig=False,path=None):
    fig,axes = plt.subplots(1,1,figsize=figsize)

    axes.plot(Train_Accuracy_Per_Batch, label="Train Accuracy")

    axes.set_title("Evolution of Accuracy per Batch")

    axes.set_xlabel("Batches")
    axes.set_ylabel("Accuracy")

    axes.plot(Test_Accuracy_Per_Batch, label="Test Accuracy")
    axes.legend()

    if savefig:
        plt.savefig(path)

    plt.show()



def compute_confusion_matrix(Labels,Predicted):
    # print(f"Nb True classes: {len(np.unique(Labels))}, Nb Predicted classes: {len(np.unique(Predicted))}")
    Confusing_Matrix = confusion_matrix(Labels, Predicted)
    return Confusing_Matrix

def plot_confusion_matrix(Confusing_Matrix,int_to_label,normalize=False,figsize=(12,6),adjust_for_labels=False,savefig=False,path=None):
    nb_unique_labels = Confusing_Matrix.shape[0]
    labels_names = [int_to_label[i] for i in range(nb_unique_labels)]

    fig = plt.figure(figsize=figsize)


    if not normalize:
        sns.heatmap(Confusing_Matrix, annot=True, fmt="d", cmap="YlGnBu",cbar=False)
    else:
        Confusing_Matrix = Confusing_Matrix.astype('float') / Confusing_Matrix.sum(axis=1)[:, np.newaxis]
        sns.heatmap(Confusing_Matrix, annot=True, fmt=".2f", cmap="YlGnBu",cbar=False)

    plt.title("Confusion Matrix")
    plt.ylabel('True', fontsize=15)
    plt.xlabel('Predicted', fontsize=15)
    # Utiliser des indices numériques pour les ticks
    tick_positions = np.arange(nb_unique_labels) + 0.5

    plt.xticks(tick_positions, labels_names, rotation=90, fontsize=15)  # Ajout des labels en rotation avec une taille de police plus grande
    plt.yticks(tick_positions, labels_names, rotation=0, fontsize=15)  # Ajout des labels pour l'axe Y avec une taille de police plus grande



    plt.tight_layout() 
    if adjust_for_labels:
        fig.subplots_adjust(bottom=0.5, left=0.25, right=0.95, top=0.95)
    if savefig:
        plt.savefig(path)

    plt.show()


