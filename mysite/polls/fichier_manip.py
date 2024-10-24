import os
import numpy as np
import json
from PIL import Image
from .couleurs_convert import * 
from django.conf import settings


def get_images_from_directories():
    base_dir = os.path.join(settings.MEDIA_ROOT, 'BD_images')
    
    image_files = []
    
    valid_extensions = ('.png', '.jpg', '.jpeg', '.gif')
    for root, dirs, files in os.walk(base_dir):
        
        for file in files:
            if file == '.DS_Store' or file.startswith('.'):
                print(f"Fichier ignoré : {file}")
                continue

            if file.lower().endswith(valid_extensions):
                image_path = os.path.join(root, file)
                try:
                    with open(image_path, 'rb') as img_file:
                        print(f"Image valide trouvée : {image_path}")
                        if image_path not in image_files:
                            image_files.append(image_path)
                except (IOError, OSError) as e:
                    print(f"Erreur lors de l'ouverture de {image_path} : {e}")
            else:
                print(f"Fichier ignoré (pas une image) : {file}")
    
    return image_files



def convert_To_RGB_DICO_NPARRAY(liste):
    result = {}
    for x in liste:
        y = Image.open(x)
        yarray = np.array(y)
        if yarray.ndim !=3  :
            y=y.convert('RGB')
            yarray = np.array(y)
            if yarray.shape[2] != 3:
                y=y.convert('RGB')
                yarray = np.array(y)
        result[x]= yarray
    return result

def convert_To_RGB(liste):
    result = []
    for x in liste:
        y = Image.open(x)
        yarray = np.array(y)
        if yarray.ndim !=3  :
            y=y.convert('RGB')
            yarray = np.array(y)
            if yarray.shape[2] != 3:
                y=y.convert('RGB')
        result.append(y)
    return result

def convert_To_RGB_image(img):
    y = Image.open(img)
    yarray = np.array(y)
    if yarray.ndim !=3  :
            y=y.convert('RGB')
            yarray = np.array(y)
            if yarray.shape[2] != 3:
                y=y.convert('RGB')
    return y

def genereJSON(liste_images,name = 'images_info.json'):
    data = {} 
    
    # Pour chaque image dans le dictionnaire (clé = nom de l'image, valeur = tableau numpy)
    for image_name in liste_images:
        data[image_name] = {}
    
    # Créer un fichier JSON avec ces informations
    with open(name, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    
    print("Le fichier JSON  "+ name+" a été créé avec succès.")
    return name

def lire_fichier_json(nom_fichier):
    with open(nom_fichier, 'r') as json_file:
        data = json.load(json_file)
    return data

# Fonction pour écrire dans le fichier JSON
def ecrire_fichier_json(nom_fichier, data):
    with open(nom_fichier, 'w') as json_file:
        json.dump(data, json_file, indent=4)

# Fonction principale pour remplir le dictionnaire de chaque image
def ajouter_conversions_images(nom_fichier_json):
    # Charger le fichier JSON existant
    with open(nom_fichier_json, 'r') as json_file:
        data = json.load(json_file)

    # Pour chaque image dans le dictionnaire (les clés sont les paths)
    for image_path, image_info in data.items():
        try:
            # Charger l'image depuis son chemin
            img = Image.open(image_path)
            arr0 = np.array(img)
            arr = convert_image(arr0)
            # Appliquer les différentes conversions
            conversions = {
                "rgb_2_grayUNI": rgb_2_grayUNI(arr),
                "rgb_2_gray601": rgb_2_gray601(arr),
                "rgb_2_gray709": rgb_2_gray709(arr),
                "rgb_2_yiq"    : rgb_2_yiq(arr),
                "rgb_2_yuv"    : rgb_2_yuv(arr),
                "rgb_to_I1I2I3": rgb_to_I1I2I3(arr),
                "RGB_to_NormalizedRGB": RGB_to_NormalizedRGB(arr),
                "RGB_to_HSV" : RGB_to_HSV(arr),
                "RGB_to_HSL" : RGB_to_HSL(arr),
                "RGB_to_LAB" : RGB_to_LAB(arr),
                "RGB_to_LUV" : RGB_to_LUV(arr),
                "RGB_to_CMYK": RGB_to_CMYK(arr)
            }

            # Mettre à jour le dictionnaire de l'image avec les conversions
            data[image_path].update(conversions)
        
        except Exception as e:
            print(f"Erreur lors du traitement de l'image {image_path} : {e}")
    
    # Sauvegarder les modifications dans le fichier JSON
    with open(nom_fichier_json, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    
    print("Les données ont été ajoutées/modifiées dans le fichier JSON avec succès.")


def image_genrator(liste_image):
    for i in liste_image:
        yield i