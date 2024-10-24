import numpy as np

def histogram(img , density = False ):
    if img.ndim == 2 :
        return np.histogram(img ,bins = 256 , range =(0,255),density=density)[0]
    elif img.ndim == 3 and img.shape[0] == 3 : 
        hist_r, _ = np.histogram(img[0], bins=256, range=(0, 255), density=density)
        hist_g, _ = np.histogram(img[1], bins=256, range=(0, 255), density=density)
        hist_b, _ = np.histogram(img[2], bins=256, range=(0, 255), density=density)
    
        return np.concatenate((hist_r, hist_g, hist_b),axis = None)
    else :
        print('pas la bonne shape de l\'image')
        return None

    
def stats(img):
    return np.array([np.mean(img),np.std(img),np.percentile(img,25),np.median(img),np.percentile(img,75)])


#arctangente : atan2 de opencv

#def gradient(img ,filtreX,filtreY):
