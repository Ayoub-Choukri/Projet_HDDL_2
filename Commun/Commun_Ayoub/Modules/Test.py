import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image
import random
import string
import re
import json
# Import pytorch 

def load_encoding_dictionaries(path,title_label_to_int ,title_int_to_label):
    
    def check_path_exists(path,message):
        if not os.path.exists(path):
            raise ValueError(message)
        else:
            pass

    check_path_exists(path,"The specified path for the encoding dictionaries does not exist. You probably need to modify it.")

    # Load JSON into dictionaries
    with open(path+title_label_to_int+".json", "r") as read_file:
        label_to_int = json.load(read_file)

    with open(path+title_int_to_label+".json", "r") as read_file:
        int_to_label = json.load(read_file)

    return label_to_int,int_to_label


def import_images_nparrays_test(folder_images_path, resize=False, size=(224,224), normalize=False):

    def check_path_exists(path,message):
        if not os.path.exists(path):
            raise ValueError(message)
        else:
            pass


    # Check if the folder exists

    check_path_exists(folder_images_path,"The specified path for the images does not exist. You probably need to modify it.")

    # Parcourir le dossier et importer les images. On considère que le dossier contient des images seulement
    images = []
    images_resized = []
    for image_name in os.listdir(folder_images_path):
        image = Image.open(folder_images_path + "/" + image_name).convert("RGB")
        if resize : 
            image_array = image.resize(size)


        image_array = np.array(image_array)
        # print("Image imported : ", image_array)

        if normalize:
            image_array = image_array/255

        # print("Image resized shape : ", image_array.shape)
        # image_array.transpose((2,1,0))
        images_resized.append(image_array)
        images.append(image)

    return images,images_resized


def test_image(image,folder_models_path,dict_paths):

    def load_model(path):

        model = torch.load(path)

        return model
    

    def predict_image(model,image):
        
        model.eval()
        with torch.no_grad():
            image = image.unsqueeze(0)
            print(image.shape)
            image = image.permute(0,3,1,2)
            output = model(image)
            _, predicted = torch.max(output, 1)
            predicted = predicted.item()
            
        return predicted
    

    def load_pannel_dict(path,title_label_to_int,title_int_to_label):
        
        def check_path_exists(path,message):
            if not os.path.exists(path):
                raise ValueError(message)
            else:
                pass

        check_path_exists(path,"The specified path for the encoding dictionaries does not exist. You probably need to modify it : "+path)

        # Load JSON into dictionaries
        with open(path+title_label_to_int+".json", "r") as read_file:
            label_to_int = json.load(read_file)

        with open(path+title_int_to_label+".json", "r") as read_file:
            int_to_label = json.load(read_file)

        return label_to_int,int_to_label



    type_to_int, int_to_type = load_pannel_dict(dict_paths+"Types/","types_to_int","int_to_types")
    # Importing the model that will predict the type pannel
    type_pannel_model_path = folder_models_path['Types']
    # Importing the image
    ## Load the image


    type_pannel_model = load_model(type_pannel_model_path)

    



    # Predicting the type pannel

    type_pannel_encoded_predict = predict_image(type_pannel_model,image)

    type_pannel = int_to_type[str(type_pannel_encoded_predict)]

    # Load Corresponding type dict
    label_to_int,int_to_label = load_pannel_dict(dict_paths+type_pannel+"/","labels_to_int","int_to_labels")





    # Getting the model that will predict the label
    model_label_path = folder_models_path[type_pannel]

    # Importing the model that will predict the label
    label_model = load_model(model_label_path)

    # Predicting the label
    label_encoded = predict_image(label_model,image)


    label = int_to_label[str(label_encoded)]

    print(f"Type pannel: {type_pannel} - Label: {label}")
    return type_pannel,label




def test_images(nbimages,images,folder_models_path,dict_paths):

    for i in range(nbimages):

        image = random.choice(images)


        type_pannel,label = test_image(image,folder_models_path,dict_paths)

        # plot the image
        plt.figure()
        image = image.detach().cpu().numpy()
        # Permute the dimensions
        image = image * 255
        # To float 
        image = image.astype(int)
        # image = np.transpose(image,(2,0, 1))
        plt.imshow(image)
        plt.title(f'Type pannel: {type_pannel} - Label: {label}')
        plt.show()

    return None




