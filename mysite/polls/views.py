from django.shortcuts import render
from django.http import JsonResponse
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
from django.utils.datastructures import MultiValueDictKeyError
from .distances import *
from .descripteurs import *
from .couleurs_convert import *
from .fichier_manip import *
import numpy as np
import os
from PIL import Image 




def interface(request):
    uploaded_file_url = None
    results_count = 0
    images_triees = []
    param = 'start'
    erreur_image = ''
    espaces_methodes  ={
        "RGB"    : RGB,
        "grisUNI": rgb_2_grayUNI,
        "gris601": rgb_2_gray601,
        "gris709": rgb_2_gray709,
        "YIQ"    : rgb_2_yiq,
        "YUV"    : rgb_2_yuv,
        "I1I2I3" : rgb_to_I1I2I3,
        "RGBnorm": RGB_to_NormalizedRGB,
        "HSV"    : RGB_to_HSV,
        "HSL"    : RGB_to_HSL,
        "LAB"    : RGB_to_LAB,
        "LUV"    : RGB_to_LUV,
        "CMYK"   : RGB_to_CMYK,
    }
    descripteur_methodes ={
        'Histogram' : histogram ,
        'Stats-IMG' : stats
    }
    distance_methodes = {
        'manhattan'     : manhattan,
        'euclidienne'   : euclidienne,
        'tchebychev'    : tchebychev,
        'intersectionX' : intersectionX,
        'intersectionY' : intersectionY,
        'khi2'          : khi2,
        'minkowski'     : minkowski
    }
    base_images = get_images_from_directories()
    if  base_images:
        print('yes')
    if request.method == 'POST':
        try :
            uploaded_image = request.FILES['image']
            fs = FileSystemStorage()
            filename = fs.save(uploaded_image.name, uploaded_image)
            uploaded_file_url = fs.url(filename)

            espace      = request.POST.get('espace-couleur')
            descripteur = request.POST.get('descripteur-couleur')
            distance    = request.POST.get('distance')
            results_count = int(request.POST.get('num-results', 1))
            normalise = request.POST.get('normalisé', 'off') == 'on'
            print(espace,descripteur,distance,bool(normalise))
            
            if espace in espaces_methodes:
                numpimg = convert_image(Image.open(uploaded_image))
                realIMG = espaces_methodes[espace](numpimg)
                images  = list(map(Image.open,base_images))
                espaces = list(map(convert_image , images))
                realIMGS= list(map(espaces_methodes[espace] , espaces))
                if descripteur in descripteur_methodes :
                    if descripteur == 'Histogram':
                        vecteur = histogram(realIMG,density=bool(normalise))
                        if distance in distance_methodes:
                            vecteurs = list(map(lambda x : histogram(x,density=bool(normalise)), realIMGS))
                            distances = [(base_images[i],vecteurs[i],distance_methodes[distance](vecteurs[i],vecteur)) for i in range(len(base_images))]
                            distances_triees = sorted(distances, key=lambda x: x[2])
                            if results_count > 0 and distances_triees:
                                images_triees = [img for img, vec, dist in distances_triees[:results_count]]
                    else :
                        vecteur = descripteur_methodes[descripteur](realIMG)
                        if distance in distance_methodes:
                            vecteurs = list(map(lambda x : descripteur_methodes[descripteur](x), realIMGS))
                            distances = [(base_images[i],vecteurs[i],distance_methodes[distance](vecteurs[i],vecteur)) for i in range(len(base_images))]
                            distances_triees = sorted(distances, key=lambda x: x[2])
                            if results_count > 0 and distances_triees:
                                images_triees = [img for img, vec, dist in distances_triees[:results_count]]
    
        except MultiValueDictKeyError :
            erreur_image = "Veuillez importer une image avant de soumettre le formulaire."
            


                    
                    



        print(erreur_image)

        return JsonResponse({
                    'images': [img for img in images_triees]
                })
    return render(request, 'interface.html')




