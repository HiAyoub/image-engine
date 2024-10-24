Ce projet vise a comparer des images en utilisant plusieurs methodes : partie 1 descripteurs et partie 2 cnn
partie 1 :
    l'utilsateur doit choisir une image depuis la base de données , apres en utilisant une liste dans le 
    formulaire , premierement il doit choisir l'espace de couleur(RGB ,HSV,HSL ...) danslequel il faut faire la comparaison.
    Ensuite , il va choisir un descripteur avec lequel on va faire la comparaison , en ce moment on a fait juste l'histogramme 
    des couleurs + les descripteur stistiques :  moyenne , ecertype , Q1 , medianne , Q3
    ensuite il va choisir la distance pour mesurer la difference entre les images  , ensuite il choisit 
    le nombre de resultats a afficher
     
    AVANT DE CALCULER LES DISTANCES IL FAUT ABSULOMENT CHOISIR UN DESCRIPTEUR CAR SINON LES IMAGES N'ONT PAS LA MEME TAILLE ET DONC ON PEUT PAS FAIRE LA DIFFERENCE


    Chaque images est caracterisé par un vecteur qui est le descripteur , dans le cas on a le vecteur histogramme 
    le vecteur [moyenne , ecertype , Q1 , medianne , Q3] , et donc apres peut import la taille des images on aura des vecteurs à comparer de meme taille

    descripteur hue : on fait l'histogramme sur la valeur de hue lorsque on est dans l'espace hsv ou hsl 
    descripteur hue ponderé : on fait l'histogramme mais au lieu de incrementer l'effectif de chaque bin par 1 on l'incremente par la valeur de S
    descripteut de histogramme de blop : on applique un filtre sur l'image (par exemple 4x4) , on commence par la premiere colonne
    donc on va prendre 4 lignes et 4 colonne (le filtre est 4x4) , on regarge les valeurs existants et on compte leur occurances
    ensuite  
---------------------------------------------------------------------------------------------------------------

IMAGES INDEXES :
    le but c de faire une projection d'une image RGB dans un espace unidimansionnel .
    donc pour faire cela il faut qu'on aille des quantificateur de chaque couleur Qrouge , Qvert , Qbleu
    qui vas nous permettre de diviser l'intervalle qui est [0 255] en un intervalle plus simple pour ne pas avoir 256 valeurs
    par exemple , si on a Qrouge = 2 , Qvert = 4 , Qbleu = 8 , donc on aura un intervalle pour le rouge qui est [0,1]
    et donc avec un pas de 128 , on calcule le pas avec la formule :256/Qrouge
    l'intervalle de vert est [0,3] avec un pas de 64
    l'intervalle de bleu est [0,7] avec un pas de 32( voir photo 15/10/2024 15h01 pour voir le repere)

    alors les formules pour la projection dependent de la numerotation des blocs(voir photo) : 
    NouveuINDEX = IndexBLEU * (QRouge * QVert) + indexVERT(QROuge)+ indexROUGE
    NouveuINDEX = IndexBLEU * (QRouge * QVert) + indexROUGE(QVERT)+ indexVERT
    ...


    -------------------------------------------------------------------------------
    descripteur gradient : appliquer un filtre(cv2.schar ,cv2.sobel,cv2.filter2D regarde image 22/10) pour avoir la derivé par rapport à x et par rapport à y , apres on calcule 
    la norme (racine(derivéeX²+derivéY²) à comparer avec cv2.magnitude) apres il faut calculer l'orientation (arctangente(derivéX/derivéY) en utilisant opencv cv2.phase)
    et apres on cree un histograme d'occurence ou bien un histogramme pondéré par la norme (au lieu d'ajouter 1 à chaque occurence 
    on ajoute la norme comme histo de hue)

    ----------------------------------------------------------------------------
    normalisation : frequence
                    norme : valeur/racine(somme(vecteur²))
                    minmax : le min en le transforme en 0 et le max en 1 et le reste des valeurs doit etre entre 0 et 1 en appliquant la methode des 3
                    par standarisation -u/ecartype
                    par rang : on a deux vecteurs [200,0.1,3] et [0,6,500] ,on transforme les deux vecteurs en vecteur des index en ordre et donc on aura:
                    [2,0,1] et [0,1,2] et on fait que la difference des rang(avec les fonctions de distances)

    -------------------------------------------------------------------------
    histo blob : 
                prend une image indexé , la taille du filtre, et un nombre sur lequel on va diviser notre hitogramme 
                l'hitogramme resultat c un tableau 2d en ligne il y a le nombre de couleur de l'image indexé et en colonnes les plages(intervalles)
                par exemple on a 8 couleurs et o divise notre intervalle en 4 (0-25% , 25%-50%,50%-75%,75%-100%).
                on applique notre filre disant 3x3 (taille 9) , donc pour la premiere fentre de notre image, pour chaque pixel on calcul sa frequence
                (dans notre cas ni/somme(n)=ni/9 )

                on fait le histo de blob pour image indexé - image en niveau de gris - image en niveau de hue(on prend la premiere composante de hsv ou hsl)

    --------------------------------------------------------------------------
    hitogramme de'orientation
    hitogramme de'orientation pondéré par la norme
    histogramme de blob des orientations
    histogramme de blob des quantification : on divise les orientation en des intervalles et on aura au lieu d'un intervalle de [0-259] on aura un 
    nouveau intervalle [0-8] par exxemple en fonction d'un parametre qu'on choisi(comme trux d'image indexée)
    --------------------------------------------------------------------------
    descripteur de texture :
    histograme de stats
    LBP : Local binary Pattern
        on reduit l'image avec un filtre (exemple on a une image de taille LxH et un filtre de taille f, on aura une nouvelle image de taille : (L-f+1)x(H-f+1) )
        on applique le filtre avec un stride de 1 , et on va ce focuser sur la valeur centrale, apres on vas faire voisin-centre(pour chaque valeur dans la fenetre(f*f))
        aprés pour chaque fentre on applique un filtre : si la valeur est negative-->0 si positive -->1 et donc on aura une nouvelle image binaire(on considere 0 comme valeur positive,, il devient 1)
        apres on choisi une numerotation pour les cases [0,1,2] aprés on va utiliser cette numerotation pour transformer les fentres binaires qu'on a calculer en des nombres bianires
                                                        [3, ,4]
                                                        [5,6,7] 
        dans ce cas on aura dans l'image resultante des nombres par exemple : 2^7 2^6 2^5 2^4 2^3 2^2 2^1 2^0 et par la suite on applique un hitogramme de blop
                                                                               1   0   0   0   1   1   0   0

        on l'applique sur les images 2D uniquement
partie 2:
    architecture VGG(8/16/19 ...)
    il faut recuperer le vecteur juste apres le flatening et avant le fully connected qui va nous servir du descripteur cnn
