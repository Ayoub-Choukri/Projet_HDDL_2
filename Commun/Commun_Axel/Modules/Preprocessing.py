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
# import Dataloader

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Import transform
from torchvision.transforms.v2 import *

# Pre-Preprocessing

# Le but de cette partie est de créer au final un DataFrame qui contient les informations suivantes:
# | Image_i | Path_i | Label_i |
def rename_images_in_folder_aux(folder_path, rename_random=False, verbose=False):
    """
    Renomme toutes les images d'un dossier sous le format img_i.png.

    :param folder_path: Chemin du dossier contenant les images.
    """

    # --- Définition des sous-fonctions ---
    def check_if_folder_exist(folder_path):
        """
        Vérifie si le chemin fourni est un dossier valide.
        """
        if not os.path.isdir(folder_path):
            print(f"Erreur : Le dossier '{folder_path}' n'existe pas.")
            return False
        return True

    def get_image_files(folder_path):
        """
        Récupère la liste des fichiers image dans le dossier.
        """
        # Récupérer le contenu du dossier
        content  = os.listdir(folder_path)
        # Filtrer les fichiers image
        files= [f for f in content if os.path.isfile(os.path.join(folder_path, f))]

         # sort the files
        files.sort()
        return files
    def rename_image_file(old_path, new_path, verbose=False):
        """
        Renomme un fichier image.
        """
        os.rename(old_path, new_path)
        if verbose:
            print(f"Renommé : {os.path.basename(old_path)} -> {os.path.basename(new_path)}")

    def convert_image_to_png(image_path):
        """
        Convertit une image en format PNG.
        """

        # Si le nom de l'image ne se termine pas par .png, save sous format png
        if not image_path.endswith(".png"):
            print(f"Conversion de l'image {image_path} en format PNG.")
            img = Image.open(image_path)
            img.save(image_path, 'PNG')

    # Cette fonction permet de trier les fichiers en fonction de leur nom (proprement)
    def sort_key(file_name):
        # Diviser le nom de fichier en parties numériques et non numériques
        parts = re.split(r'(\d+)', file_name)
        # Convertir les parties numériques en entiers pour un tri correct
        return [int(part) if part.isdigit() else part for part in parts]
    
    def generate_random_name(length):
        """
        Génère un nom aléatoire de longueur donnée, composé de lettres et de chiffres.
        
        Parameters:
            length (int): La longueur du nom à générer.
            
        Returns:
            str: Un nom aléatoire.
        """
        if length <= 0:
            raise ValueError("La longueur doit être un entier positif.")
        
        # Utiliser des lettres majuscules, minuscules et chiffres
        characters = string.ascii_letters + string.digits
        return ''.join(random.choices(characters, k=length))
    
    # Vérifier si le dossier existe
    if not check_if_folder_exist(folder_path):
        print(" Dossier introuvable.")
        return

    # Récupérer la liste des fichiers dans le dossier
    image_files = get_image_files(folder_path)

    # sort 
    image_files = sorted(image_files, key=sort_key)
    # Si aucune image n'est trouvée, afficher un message et arrêter
    if not image_files:
        print(f"Aucune image trouvée dans le dossier '{folder_path}'.")
        return

    # Parcourir et renommer les images
    for i, filename in enumerate(image_files, start=1):
        # Chemin complet de l'ancien fichier
        old_path = os.path.join(folder_path, filename)

        # # Convertir l'image en format PNG
        # convert_image_to_png(old_path)

        if rename_random:
            # Générer le nouveau nom du fichier
            new_filename = generate_random_name(np.random.randint(10, 20)) + ".png"
        else:
            # Générer le nouveau nom du fichier
            new_filename = f"img_{i}.png"

        # Chemin complet du nouveau fichier
        new_path = os.path.join(folder_path, new_filename)

        # Renommer le fichier
        rename_image_file(old_path, new_path, verbose)

    if verbose:
        # Message final de succès
        print("Toutes les images ont été renommées avec succès.")

def rename_images_in_folder(folder_path):
    rename_images_in_folder_aux(folder_path, rename_random=True, verbose=False)
    rename_images_in_folder_aux(folder_path, rename_random=False, verbose=True)

def rename_images(folder_path):
    """
    Renomme toutes les images de notre dossier donné sous le format img_i.png.

    Le dossier donné doit contenir des sous dossiers (Obligation-Interdiction-Danger) qui contiennent eux-mêmes des dossiers d'images.

    :param folder_path: Chemin du dossier de données
    """

    List_Subfolders = os.listdir(folder_path)

    for subfolder in List_Subfolders:
        List_Subfolders_Of_Subfolder = os.listdir(folder_path + '/' + subfolder)

        for subsubfolder in List_Subfolders_Of_Subfolder:
            rename_images_in_folder(folder_path + '/' + subfolder + '/' + subsubfolder)


def create_dataframe_dataset(folder_path):
    """
    Cette fonction est censée parcourir le dossier des données, creer un DataFrame du type suivant:
    | Image_i | Path_i | Label_i |

    Le Label sera exaxctement la concatenation des noms des dossiers parents de l'image.

    :param folder_path: Chemin du dossier de données

    """

    # --- Définition des sous-fonctions ---
    def check_if_folder_exist(folder_path):
        """
        Vérifie si le chemin fourni est un dossier valide.
        """
        if not os.path.isdir(folder_path):
            print(f"Erreur : Le dossier '{folder_path}' n'existe pas.")
            return False
        return True

    def get_image_files(folder_path):
        """
        Récupère la liste des fichiers image dans le dossier.
        """
        # Récupérer le contenu du dossier
        content  = os.listdir(folder_path)
        # Filtrer les fichiers image
        files= [f for f in content if os.path.isfile(os.path.join(folder_path, f))]

         # sort the files
        files.sort()
        return files

    def get_subfolders(folder_path):
        """
        Récupère la liste des sous-dossiers dans le dossier. 
        """
        # Récupérer le contenu du dossier
        content  = os.listdir(folder_path)
        # Filtrer les sous-dossiers
        subfolders = [f for f in content if os.path.isdir(os.path.join(folder_path, f))]
        return subfolders


    def get_label_from_path(path): 
        """
        Récupère le label à partir du chemin de l'image.
        """
        return path.split('/')[-3] , path.split('/')[-2], path.split('/')[-3] + '-' + path.split('/')[-2]
    

    def gets_image_paths_and_labels(folder_path):

        # Parcourir les images
        image_relative_paths=[]
        image_paths = []
        image_type = []
        image_sublabel = []
        image_labels = []
        # Récupérer la liste des fichiers dans le dossier
        image_files = get_image_files(folder_path)  

        actual_path = os.path.abspath(folder_path)
        actual_relative_path = os.path.relpath(folder_path)
        for image in image_files:
            path_image = actual_path + '/' + image
            relative_path_image = actual_relative_path + '/' + image
            image_paths.append(path_image)
            type,sublabel, label = get_label_from_path(path_image)
            image_type.append(type)
            image_relative_paths.append(relative_path_image)
            image_sublabel.append(sublabel)
            image_labels.append(label)


        return image_paths, image_relative_paths, image_type, image_sublabel, image_labels
    

    # Vérifier si le dossier existe
    if not check_if_folder_exist(folder_path):
        print(" Dossier introuvable.")
        return
    
    # Récupérer la liste des sous-dossiers
    subfolders = get_subfolders(folder_path)

    image_paths = []
    image_relative_paths = []
    image_labels = []
    image_types = []
    image_sublabels = []
    # Parcourir les sous-dossiers
    for subfolder in subfolders: # (Obligation-Interdiction-Danger)
        subfolder_of_subfolder = get_subfolders(folder_path + '/' + subfolder)

        for subsubfolder in subfolder_of_subfolder: # (Type 1 - Type 2 - Type 3)

            image_paths_to_add,image_relative_paths_to_add, image_types_to_add,image_sublabels_to_add, image_labels_to_add,  = gets_image_paths_and_labels(folder_path + '/' + subfolder + '/' + subsubfolder)

            image_paths += image_paths_to_add
            image_relative_paths += image_relative_paths_to_add
            image_labels += image_labels_to_add
            image_types += image_types_to_add
            image_sublabels += image_sublabels_to_add


            # Créer le DataFrame
            df = pd.DataFrame({ 'Path': image_paths, 'Relative_Path': image_relative_paths, 'Type': image_types, 'Sublabel': image_sublabels,  'Label': image_labels })

            
    return df


def Separate_Train_Validation_Test(Images_Dataframe_Infos, test_size=0.2,Save=False, Path = None,Name_Train='Train.csv',Name_Test='Test.csv'):
    if test_size == 0:
        train_data = Images_Dataframe_Infos
        test_data = Images_Dataframe_Infos.copy()
    else:
        # use train_test_split to split the data into train and test sets
        train_data, test_data = train_test_split(Images_Dataframe_Infos, test_size=test_size,stratify=Images_Dataframe_Infos['Label'])

    if Save and Path is not None  : 
        train_data.to_csv(Path + Name_Train )
        test_data.to_csv(Path + Name_Test )

    return train_data, test_data
    
# Preprocessing

## Le but dans cette partie est de créer un pipeline de preprocessing qui va nous permettre de:
# - Charger les images
# - Les redimensionner
# - Les normaliser ou pas
# - Les augmenter ou pas




def load_image(image_path,resize=False, normalize=False, size=(512,512)):
    """
    Charge une image à partir de son chemin.

    :param image_path: Chemin de l'image
    """

    # --- Définition des sous-fonctions ---
    def check_if_image_exist(image_path):
        """
        Vérifie si le chemin fourni est un fichier image valide.
        """
        if not os.path.isfile(image_path):
            print(f"Erreur : Le fichier '{image_path}' n'existe pas.")
            return False
        return True

    def load_image_aux(image_path,format='RGB'):
        """
        Charge l'image à partir du chemin.
        """
        try:
            image = Image.open(image_path).convert(format)
            return image
        except :
            print(f"\033[91mErreur lors du chargement de l'image : {image_path} en format {format}\033[0m")
            return None
        
    def resize_image(image, size):
        """
        Redimensionne l'image.
        """
        image = image.resize(size)
        return image
    
    def normalize_image(image):
        """
        Normalise l'image.
        """
        image = np.array(image)
        image = image / 255.0
        return image
    


    # Vérifier si l'image existe
    if not check_if_image_exist(image_path):
        print(f" Image introuvable : {image_path}")
        return

    # Charger l'image
    image = load_image_aux(image_path)

    if resize and image is not None:
        image = resize_image(image, size)

    if normalize and image is not None:
        image = normalize_image(image)

    return image



def load_images(image_paths,resize=False, normalize=False, size=(512,512)):
    """
    Charge une liste d'images à partir de leurs chemins.

    :param image_paths: Liste des chemins des images
    """




    ProgressBar = tqdm(range(len(image_paths)))

    images = []

    index_images_to_remove = []

    for Path in ProgressBar:
        ProgressBar.set_description('Processing image %s' % image_paths[Path])
        image = load_image(image_paths[Path],resize, normalize, size)

        if image is not None:
            image_array = np.array(image)
            if image_array.shape == (size[0], size[1], 3):
                images.append(image_array)
            else:
                index_images_to_remove.append(Path)
                print(f"\033[91mErreur lors du resize de l'image : {image_paths[Path]}\033[0m")
                print(f"Shape : {image_array.shape}")
        else:
            index_images_to_remove.append(Path)
            print(f"\033[91mErreur lors du chargement de l'image : {image_paths[Path]}\033[0m")

    if not all(img.shape == (size[0], size[1], 3) for img in images):
        print("Error: Not all images have the same shape")


    if resize : 
        images_array = np.array(images)
        images_array = np.moveaxis(images_array, 3, 1)
    else:
        images_array = images
    return images , index_images_to_remove


def Create_Encoding_Label_Dictionary(labels,save=False, path=None,title_label_to_int='label_to_int',title_int_to_label='int_to_label'):
    """
    Encode les labels en entiers.
    """
    unique_labels = np.unique(labels)
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    int_to_label = {i: label for i, label in enumerate(unique_labels)}

    # Save in JSON
    if save and path is not None:
        with open(path+f'{title_label_to_int}.json', 'w') as fp:
            json.dump(label_to_int, fp)

        with open(path+f'{title_int_to_label}.json', 'w') as fp:
            json.dump(int_to_label, fp)




    return label_to_int, int_to_label

def encode_labels(labels, label_to_int):
    """
    Encode les labels en entiers.
    """
    return [label_to_int[label] for label in labels]

def decode_labels(labels, int_to_label):
    """
    Decode les labels en entiers.
    """
    return [int_to_label[label] for label in labels]




def import_images(dataframe_paths,path_column, type_column , sublabel_column, label_column,resize=False, normalize=False, size=(512,512)):
    """
    Importe les images à partir des chemins dans le DataFrame.

    :param dataframe_paths: DataFrame contenant les chemins des images
    """

    paths = dataframe_paths[path_column].values

    types = dataframe_paths[type_column].values

    sublabels = dataframe_paths[sublabel_column].values


    labels = dataframe_paths[label_column].values


    images , index_images_to_remove = load_images(paths,resize, normalize, size)

    types = np.delete(types, index_images_to_remove)
    sublabels = np.delete(sublabels, index_images_to_remove)
    labels = np.delete(labels, index_images_to_remove)

    

    return images, types, sublabels, labels



def import_images_numpy_array(dataframe_paths,path_column, type_column , sublabel_column, label_column,resize=False, normalize=False, size=(512,512)):
    """
    Importe les images à partir des chemins dans le DataFrame.

    :param dataframe_paths: DataFrame contenant les chemins des images
    """
    print("Importing images")
    images, types, sublabels, labels = import_images(dataframe_paths,path_column, type_column , sublabel_column, label_column,resize, normalize, size)


    if resize:
        print("Converting Images to numpy array")
        images = np.array(images)
        images = np.moveaxis(images, 3, 1)

    return images, types, sublabels, labels




def import_images_tensor(dataframe_paths,path_column, type_column , sublabel_column, label_column,resize=False, normalize=False, size=(512,512)):
    """
    Importe les images à partir des chemins dans le DataFrame.

    :param dataframe_paths: DataFrame contenant les chemins des images
    """
    print("Importing images")
    images, types, sublabels, labels = import_images(dataframe_paths,path_column, type_column , sublabel_column, label_column,resize, normalize, size)

    if resize:
        print("Converting Imported images to tensor")
        images = np.array(images)
        images = torch.tensor(images).float()

    return images, types, sublabels, labels






def create_dataloader(images, types, sublabels, labels, batch_size=32, shuffle=True, test_size=0.2):

    """
    Crée un DataLoader à partir des images.

    :param images: Liste des images
    :param batch_size: Taille du batch
    :param shuffle: Mélanger les données
    """
    
    class CustomDataset(Dataset):
        def __init__(self, images, types, sublabels, labels):
            self.images = images
            self.types = types
            self.sublabels = sublabels
            self.labels = labels

            # Channel second
            print(self.images.shape)
            self.images = self.images.permute(0,3,1,2)

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):

            return self.images[idx], self.types[idx], self.sublabels[idx], self.labels[idx]


    if test_size == 0 : 
        dataset = CustomDataset(images, types, sublabels, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader, dataloader
    else:
        images_train, images_test, types_train, types_test, sublabels_train, sublabels_test, labels_train, labels_test = train_test_split(images, types, sublabels, labels, test_size=test_size, random_state=42)
        images_train = torch.tensor(images_train).float()
        images_test = torch.tensor(images_test).float()
        types_train = torch.tensor(types_train).float()
        types_test = torch.tensor(types_test).float()
        sublabels_train = torch.tensor(sublabels_train).float()
        sublabels_test = torch.tensor(sublabels_test).float()
        labels_train = torch.tensor(labels_train).float()
        labels_test = torch.tensor(labels_test).float()

        
        dataset_train = CustomDataset(images_train, types_train, sublabels_train, labels_train)
        dataset_test = CustomDataset(images_test, types_test, sublabels_test, labels_test)
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle)
        dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=shuffle)
        return dataloader_train, dataloader_test
    

def create_transofrm_augmentation_types(p=0.6):
    # p est la probabilité d'appliquer la transformation
    List_Transforms = [
        RandomHorizontalFlip(0.8),
        RandomVerticalFlip(0.7),
        RandomRotation(90),
        RandomPerspective(),
        ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        RandomAffine(degrees=30),
        GaussianBlur(5),
        transforms.Lambda(lambda img: img + torch.randn_like(img)*0.05),
    ] 
    
    Transforms = Compose([
        RandomApply(List_Transforms, p=p),
        ToTensor()
    ])

    return Transforms

def save_transfroms(transforms, path):

    class Trransform_class:
        def __init__(self, transforms):
            self.transforms = transforms

        def __call__(self, img):
            return self.transforms(img)
        

    torch.save(Trransform_class(transforms), path)

def load_transforms(path):
    transform_class = torch.load(path)

    return transform_class.transforms




def create_transofrm_augmentation_labels(p=0.6):
    # p est la probabilité d'appliquer la transformation
    List_Transforms = [
        RandomVerticalFlip(0.6),
        RandomRotation(45),
        RandomPerspective(),
        ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        RandomAffine(degrees=30),
        GaussianBlur(5),
        transforms.Lambda(lambda img: img + torch.randn_like(img)*0.05),
    ] 
    
    Transforms = Compose([
        RandomApply(List_Transforms, p=p),
        ToTensor()
    ])

    return Transforms



def show_images(images, types, sublabels, labels, n=9,figsize=(20,20), l= 3,c=3,is_normalised = False):
    """
    Affiche les images.

    :param images: Liste des images
    :param n: Nombre d'images à afficher
    """

    def channel_last(image):
        """
        Met le channel en dernier.
        """
        print(image.shape)
        return np.moveaxis(image, 0, 2)
    
    def image_as_int(image):
        """
        Convertit l'image en entier.
        """
        if is_normalised:
            return image
        else:
            return image.astype(int)
    
    def select_random_images(images, types, sublabels, labels, n):
        """
        Sélectionne un nombre aléatoire d'images à afficher.
        """
        indices_images_to_show = np.random.choice(range(len(images)), n, replace=False)
        images = [image_as_int(images[i]) for i in indices_images_to_show]
        types = [types[i] for i in indices_images_to_show]
        sublabels = [sublabels[i] for i in indices_images_to_show]
        labels = [labels[i] for i in indices_images_to_show]
        return images, types, sublabels, labels




    images = np.array(images)
    fig,axes = plt.subplots(l,c,figsize=figsize)

    images, types, sublabels, labels = select_random_images(images, types, sublabels, labels, n)

    for i in range(l):
        for j in range(c):
            axes[i,j].imshow(images[i*c+j])
            axes[i,j].set_title(f"Type: {types[i*c+j]} - Sublabel: {sublabels[i*c+j]} - Label: {labels[i*c+j]}")
            axes[i,j].axis('off')

    plt.show()


