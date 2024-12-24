import sys
import os


def check_path_exists(Path,Message):
    if not os.path.exists(Path):
        print(Message)
        sys.exit(1)
# Cette partie permet d'ajouter le dossier parent au path, pour pouvoir importer les modules que nous avons créés. ça peut être soit un chemin
# absolu, soit un chemin relatif. Ici, c'est un chemin absolu.
Path_Modules = "./Modules"
Path_Models = "./Models"
check_path_exists(Path_Modules,"Le chemin spécifié pour importer les modules n'existe pas. Il faut surement le modifier.")


sys.path.append(Path_Modules)
sys.path.append(Path_Models)

# Importer les module
from Preprocessing import *
from Resnet import *
from Test import *
from EfficientNet import *






resize=True
size=(224,224)
normalize=True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

folder_test_images_path = "./../Donnees_Test"


# Checker si le dossier existe
check_path_exists(folder_test_images_path,"Le chemin spécifié pour les images n'existe pas. Il faut surement le modifier.")

# Renommer les images
rename_images_in_folder(folder_test_images_path)

# Créer le DataFrame
Images_Original,Images = import_images_nparrays_test(folder_test_images_path, resize=True, size=(224,224), normalize=True)


import matplotlib.pyplot as plt

# # Plot the first image
# Image_To_Plot = Images[0]

# Image_To_Plot= np.transpose(Image_To_Plot, (0, 1, 2))
# plt.imshow(Image_To_Plot)
# plt.title("First Image")
# plt.show()
Images = torch.tensor(Images, dtype=torch.float32)

Images = Images.to(device)

Frac_Images_To_Import = 1

model_type_path = "./Saved_Models/Model_Types.pth"

model_Dangers_path = "./Saved_Models/Model_Dangers.pth"

model_Fin_Interdictions_path = "./Saved_Models/Model_Fin_Interdictions.pth"

model_Obligations_path = "./Saved_Models/Model_Obligations.pth"

model_Indications_path = "./Saved_Models/Model_Indications.pth"

model_Interdictions_path = "./Saved_Models/Model_Interdictions.pth"


# Gather models in Dict

models_paths = {
    "Types": model_type_path,
    "Dangers": model_Dangers_path,
    "Fin_Interdictions": model_Fin_Interdictions_path,
    "Obligations": model_Obligations_path,
    "Indications": model_Indications_path,
    "Interdictions": model_Interdictions_path
}



dict_paths = "./Encoding_Dictionaries/"

test_images(5,Images,models_paths,dict_paths)




