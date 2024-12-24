# Projet HDDL





## Description
Ce dépôt contient le travail collaboratif réalisé par l'équipe dans le cadre du projet **HDDL**.


## Lien dépot : 

Lien du dépot : [https://github.com/Ayoub-Choukri/Projet_HDDL_2.git](https://github.com/Ayoub-Choukri/Projet_HDDL_2.git)

## Sujet du projet
Détéction des panneaux routiers.
## Structure du dépôt

Le dépôt **Projet_HDDL** est organisé comme suit :  

- **Donnees/** : Contient les données d'entraînement utilisées pour entraîner les différents modèles. Ce répertoire inclut toutes les images de panneaux classées par type et catégorie.  

- **Donnees_Test/** : Regroupe les données de test, séparées des données d'entraînement, utilisées pour évaluer les performances des modèles sur des données non vues. (Images de Google Street)

- **Resnet/** : Contient tous les éléments relatifs au modèle ResNet :  
  - **Metrics/** : Inclut les résultats et les métriques calculées pendant l'entraînement et les tests.
    - **Metrics_Types/** : Métriques liés à la prédictions de types de panneaux (Interdictions, Dangers, ....)
    - **Metrics_Dangers/** : Métriques spécifiques pour les panneaux de type "Dangers".  
    - **Metrics_Interdictions/** : Métriques spécifiques pour les panneaux de type "Interdictions".  
    - *(etc.)*  
  - **Scripts/** : Contient les scripts utilisés pour entraîner, évaluer ou analyser les performances du modèle ResNet.  
  - **Autres Dossiers/** : Divers fichiers ou outils supplémentaires spécifiques au modèle ResNet.  

- **EfficientNet/** : Contient tous les éléments relatifs au modèle EfficientNet :  
  - **Metrics/** : Résultats et métriques calculées pour ce modèle.
    - **Metrics_Types/** : Métriques liés à la prédictions de types de panneaux (Interdictions, Dangers, ....)
    - **Metrics_Dangers/** : Métriques spécifiques pour les panneaux de type "Dangers".  
    - **Metrics_Interdictions/** : Métriques spécifiques pour les panneaux de type "Interdictions".  
    - *(etc.)*  
  - **Scripts/** : Scripts utilisés pour l'entraînement et l'évaluation.  
  - **Autres Dossiers/** : Fichiers ou outils spécifiques à EfficientNet.  

- **MobileNet/** : Contient tous les éléments relatifs au modèle MobileNet, avec la même structure que pour ResNet et EfficientNet.  

- **Vit/** : Contient tous les éléments relatifs au modèle ViT (Vision Transformer), avec la même structure que pour ResNet et EfficientNet.  

- **Modules/** : Répertoire contenant des modules Python personnalisés ou génériques utilisés pour le projet, comme des fonctions de traitement d'images ou des pipelines d'entraînement.  

- **Annotations/** : Contient les fichiers d'annotations des données, qui définissent les classes des panneaux dans les images.  



## Collaboration avec Git


Lien du dépot : [https://github.com/Ayoub-Choukri/Projet_HDDL_2.git](https://github.com/Ayoub-Choukri/Projet_HDDL_2.git)

1. Clonez le dépôt avec HTTPS :
   ```bash
   git clone https://github.com/Ayoub-Choukri/Projet_HDDL_2.git

2. Clonez le dépôt avec SSH :
   ```bash
   git clone git@github.com:Ayoub-Choukri/Projet_HDDL_2.git
