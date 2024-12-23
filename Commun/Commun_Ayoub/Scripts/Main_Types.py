import sys
import os


def check_path_exists(Path,Message):
    if not os.path.exists(Path):
        print(Message)
        sys.exit(1)
# Cette partie permet d'ajouter le dossier parent au path, pour pouvoir importer les modules que nous avons créés. ça peut être soit un chemin
# absolu, soit un chemin relatif. Ici, c'est un chemin absolu.
Path_Modules = "/home/ayoubchoukri/Etudes/5A/Stats_Grande_Dimension/Projets/Projet/Projet_HDDL_2/Commun/Commun_Ayoub/Modules"
Path_Models = "/home/ayoubchoukri/Etudes/5A/Stats_Grande_Dimension/Projets/Projet/Projet_HDDL_2/Commun/Commun_Ayoub/Models"
check_path_exists(Path_Modules,"Le chemin spécifié pour importer les modules n'existe pas. Il faut surement le modifier.")


sys.path.append(Path_Modules)
sys.path.append(Path_Models)

# Importer les module
from Preprocessing import *
from Resnet import *
from Train import *




# On définit une variable booleeene, pour savoir si on renomme les images, parcourt tous les dossiers et sous dossiers d'images
# pour creer un dataframe contenant les informations des images (Path, Type, Sublabel, Label). Si, elle à False, le DataFrame en question 
# doit déjà exister, et va donc être utilisé pour la suite.

Pre_Preprocessing = True
Image_Information_Path = "/home/ayoubchoukri/Etudes/5A/Stats_Grande_Dimension/Projets/Projet/Projet_HDDL_2/Commun/Anotations/"
Name_Image_Information = "Image_Information.csv"
Name_Image_Information_Train = "Image_Information_Train.csv"
Name_Image_Information_Test = "Image_Information_Test.csv"

if Pre_Preprocessing:
    # Chemin du dossier contenant les images
    folder_images_path = "/home/ayoubchoukri/Etudes/5A/Stats_Grande_Dimension/Projets/Projet/Projet_HDDL_2/Commun/Donnees"

    # Checker si le dossier existe
    check_path_exists(folder_images_path,"Le chemin spécifié pour les images n'existe pas. Il faut surement le modifier.")

    # Renommer les images
    rename_images(folder_images_path)

    # Créer le DataFrame
    Image_Information_Dataframe = create_dataframe_dataset(folder_images_path)

    # Enregistrer le DataFrame
    Image_Information_Dataframe.to_csv(Image_Information_Path+Name_Image_Information)


    Image_Information_Dataframe_Train, Image_Information_Dataframe_Test = Separate_Train_Validation_Test(Image_Information_Dataframe,0.2, Save=True, Path=Image_Information_Path , Name_Train=Name_Image_Information_Train, Name_Test=Name_Image_Information_Test)
else:
    # Checker si le fichier existe
    check_path_exists(Image_Information_Path,"Le chemin spécifié pour le DataFrame n'existe pas. Il faut surement le modifier.")

    # Charger le DataFrame
    Image_Information_Dataframe = pd.read_csv(Image_Information_Path+Name_Image_Information)

    Image_Information_Dataframe_Train = pd.read_csv(Image_Information_Path+Name_Image_Information_Train)
    Image_Information_Dataframe_Test = pd.read_csv(Image_Information_Path+Name_Image_Information_Test)







# Preprocessing des images






resize=True
size=(224,224)
normalize=True
# Nb_Images_To_Import = 5
Frac_Images_To_Import = 1

images_train, types_train, sublabels_train, labels_train = import_images_tensor(Image_Information_Dataframe_Train.sample(frac=Frac_Images_To_Import),"Relative_Path", "Type", "Sublabel", "Label",resize=resize, size=size,normalize=normalize)
images_test , types_test, sublabels_test, labels_test = import_images_tensor(Image_Information_Dataframe_Test.sample(frac=Frac_Images_To_Import),"Relative_Path", "Type", "Sublabel", "Label",resize=resize, size=size,normalize=normalize)



# Création des dictionnaires d'encodage
save_dicts = True
Path_Save_Dicts = "/home/ayoubchoukri/Etudes/5A/Stats_Grande_Dimension/Projets/Projet/Projet_HDDL_2/Commun/Commun_Ayoub/Encoding_Dictionaries/Types/"
types_to_int , int_to_types = Create_Encoding_Label_Dictionary(types_train,save=save_dicts, path=Path_Save_Dicts,title_label_to_int='types_to_int',title_int_to_label='int_to_types')
sublabels_to_int , int_to_sublabels = Create_Encoding_Label_Dictionary(sublabels_train)
labels_to_int , int_to_labels = Create_Encoding_Label_Dictionary(labels_train)

# Encodage des labels
types_encoded_train,types_encoded_test = encode_labels(types_train, types_to_int) , encode_labels(types_test, types_to_int)
sublabels_encoded_train,sublabels_encoded_test = encode_labels(sublabels_train, sublabels_to_int) , encode_labels(sublabels_test, sublabels_to_int)
labels_encoded_train,labels_encoded_test = encode_labels(labels_train, labels_to_int) , encode_labels(labels_test, labels_to_int)


# Creating Dataloaders : Train and Test
dataloader_train, _ = create_dataloader(images_train, types_encoded_train, sublabels_encoded_train, labels_encoded_train, batch_size=32, test_size=0, shuffle=True)

dataloader_test ,_= create_dataloader(images_test, types_encoded_test, sublabels_encoded_test, labels_encoded_test, batch_size=32, test_size=0, shuffle=True)






# Training


What_To_Train = "Type"
Train = True
Save = True and Train
Save_Architecture = True 
Model_Path = "/home/ayoubchoukri/Etudes/5A/Stats_Grande_Dimension/Projets/Projet/Projet_HDDL_2/Commun/Commun_Ayoub/Saved_Models/Model_Types.pth"

Model = resnet18(num_classes=len(types_to_int))



# Train the model
Nb_Epochs = 70
optimizer = torch.optim.Adam(Model.parameters(), lr=0.001, weight_decay=0.0001)
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")
augment = True
transform_type =create_transofrm_augmentation_types(p=0.4)

Confusion_Matrix_Saving_Path = "/home/ayoubchoukri/Etudes/5A/Stats_Grande_Dimension/Projets/Projet/Projet_HDDL_2/Commun/Commun_Ayoub/Metrics/Metrics_Types/Confusion_Matrix/"
Losses_Saving_Path = "/home/ayoubchoukri/Etudes/5A/Stats_Grande_Dimension/Projets/Projet/Projet_HDDL_2/Commun/Commun_Ayoub/Metrics/Metrics_Types/Losses/"
if Train:
    Train_Losses_Per_Batch, Test_Losses_Per_Batch, Train_Accuracy_Per_Batch, Test_Accuracy_Per_Batch, Train_Losses_Per_Epoch, Test_Losses_Per_Epoch, Train_Accuracy_Per_Epoch, Test_Accuracy_Per_Epoch = train_model(Model, dataloader_train,dataloader_test,Nb_Epochs,criterion, optimizer,device,train_on="types",augment=augment,transforms= transform_type)

    # Save the model
    save_model(Model,Model_Path,Save_Architecture)


    # Losses and Accuracy graphs

    plot_losses_per_epoch(Train_Losses_Per_Epoch, Test_Losses_Per_Epoch,figsize=(12,6),savefig=True,path=Losses_Saving_Path+"Losses_Per_Epoch.png")
    plot_accuracy_per_epoch(Train_Accuracy_Per_Epoch, Test_Accuracy_Per_Epoch,figsize=(12,6),savefig=True,path=Losses_Saving_Path+"Accuracy_Per_Epoch.png")
    plot_losses_per_batch(Train_Losses_Per_Batch, Test_Losses_Per_Batch,figsize=(12,6),savefig=True,path=Losses_Saving_Path+"Losses_Per_Batch.png")
    plot_accuracy_per_batch(Train_Accuracy_Per_Batch, Test_Accuracy_Per_Batch,figsize=(12,6),savefig=True,path=Losses_Saving_Path+"Accuracy_Per_Batch.png")


else:
    # Charger le modèle
    Model = load_model(Model_Path,load_architechture=Save_Architecture, model_class=resnet18)
    Model = Model.to(device)





# Testing
Predicted_Train , Labels_Train = test_model(Model,dataloader_train,device,train_on="types")
confusion_Matrix_Train = compute_confusion_matrix(Labels_Train,Predicted_Train)
# confusion_Matrix_Train_normalized = confusion_Matrix_Train.astype('float') / confusion_Matrix_Train.sum(axis=1)[:, np.newaxis]


Predicted_Test , Labels_Test = test_model(Model,dataloader_test,device,train_on="types")
confusion_Matrix_Test = compute_confusion_matrix(Labels_Test,Predicted_Test)
# confusion_Matrix_Test_normalized = confusion_Matrix_Test.astype('float') / confusion_Matrix_Test.sum(axis=1)[:, np.newaxis]

# plot confusion matrix
plot_confusion_matrix(confusion_Matrix_Train,int_to_label=int_to_types,figsize=(12,6),savefig=True,path= Confusion_Matrix_Saving_Path+"Confusion_Matrix_Train.png")
plot_confusion_matrix(confusion_Matrix_Test,int_to_label=int_to_types,figsize=(12,6),savefig=True,path= Confusion_Matrix_Saving_Path+"Confusion_Matrix_Test.png")
plot_confusion_matrix(confusion_Matrix_Train,normalize=True,int_to_label=int_to_types,figsize=(12,6),savefig=True,path= Confusion_Matrix_Saving_Path+"Confusion_Matrix_Train_Normalized.png")
plot_confusion_matrix(confusion_Matrix_Test,normalize=True,int_to_label=int_to_types,figsize=(12,6),savefig=True,path= Confusion_Matrix_Saving_Path+"Confusion_Matrix_Test_Normalized.png")



