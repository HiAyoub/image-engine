import numpy as np


def convert_image(imgreel):
    tab = np.array(imgreel)
    if tab.ndim !=3  :
            imgreel=imgreel.convert('RGB')
            tab = np.array(imgreel)
            if tab.shape[2] != 3:
                imgreel=imgreel.convert('RGB')
    transpose = tab.transpose(2,0,1)
    return transpose

def convert_image_back(imgreel):
    tab=np.array(imgreel)
    return tab.transpose(1,2,0)

def RGB(img):
   return np.array(img)

def rgb_2_grayUNI(img):
  return np.mean(img,axis = 0)

def rgb_2_gray709(img):
  if img.shape[0] >3:
    img=img[:3]
  return np.sum(img*np.array([0.2125,0.7154,0.0721]).reshape(img.shape[0],1,1) , axis = 0)

def rgb_2_gray601(img):
  if img.shape[0] >3:
    img=img[:3]
  return np.sum(img*np.array([0.299,0.587,0.114]).reshape(img.shape[0],1,1) , axis = 0)


def PIXELrgb_2_yiq(pixel):
    mat = np.array([[0.299,0.587,0.114],[0.596,-0.274,-0.322],[0.211,-0.523,0.312]])
    return np.round(np.dot(mat,RGB_to_NormalizedRGB(pixel)),3)

def rgb_2_yiq(img):
  print('yiq')
  return np.apply_along_axis(PIXELrgb_2_yiq, 0, img)

def PIXELrgb_2_yuv(pixel):
    mat = np.array([[0.299,0.587,0.114],[-0.147,-0.289,-0.437],[0.615,-0.515,0.100]])
    return np.round(np.dot(mat,RGB_to_NormalizedRGB(pixel)),3) 

def rgb_2_yuv(img):
    print('yuv')
    return np.apply_along_axis(PIXELrgb_2_yuv, 0, img)

def PIXELrgb_to_I1I2I3(rgb):
  return np.array([[np.sum(rgb)/3],[(rgb[0]-rgb[1])/2],[(2*rgb[2]-rgb[0]-rgb[1])/4]])

def rgb_to_I1I2I3(img):
  print('I2i1')

  return np.apply_along_axis(PIXELrgb_to_I1I2I3, 0, img)

def RGB_to_NormalizedRGB(img):
  return img /255.0

def PIXELRGB_to_HSV(img):
  norm = RGB_to_NormalizedRGB(img)
  cmax = np.argmax(norm)
  cmin = np.argmin(norm)
  delta = norm[cmax] - norm[cmin]
  H,S,V =0,0,norm[cmax]
  if delta != 0:
    if cmax == 0:
      H = 60*(((norm[1]-norm[2])/delta)%6)
    elif cmax == 1:
      H = 60*(((norm[2]-norm[0])/delta)+2)
    elif cmax == 2:
      H = 60*(((norm[0]-norm[1])/delta)+4)

  if norm[cmax] != 0:
    S = delta/norm[cmax]

  return np.round(np.array([H,S,V]),3)

def RGB_to_HSV(img):
  return np.apply_along_axis(PIXELRGB_to_HSV, 0, img)

def PIXELRGB_to_HSL(img):
    norm = RGB_to_NormalizedRGB(img)
    cmax = np.argmax(norm)
    cmin = np.argmin(norm)
    delta = norm[cmax] - norm[cmin]

    L = (norm[cmax] + norm[cmin]) / 2
    H, S = 0, 0

    if delta != 0:
        if cmax == 0:  
            H = 60 * (((norm[1] - norm[2]) / delta) % 6)
        elif cmax == 1:  
            H = 60 * (((norm[2] - norm[0]) / delta) + 2)
        elif cmax == 2:  
            H = 60 * (((norm[0] - norm[1]) / delta) + 4)

    # Calcul de la saturation (S)
    if L == 0 or L == 1:
        S = 0
    else:
        S = delta / (1 - abs(2 * L - 1))

    return np.round(np.array([H, S, L]), 3)

def RGB_to_HSL(img):
  return np.apply_along_axis(PIXELRGB_to_HSL, 0, img)

def RGB_to_XYZ(pixel):
    # Normalisation des valeurs RGB
    norm_pixel = RGB_to_NormalizedRGB(pixel)

    # Correction gamma (sRGB)
    def gamma_correction(value):
        return ((value + 0.055) / 1.055) ** 2.4 if value > 0.04045 else value / 12.92

    norm_pixel = np.array([gamma_correction(c) for c in norm_pixel])

    # Matrice de transformation de sRGB à XYZ (D65 standard illuminant)
    mat = np.array([[0.4124564, 0.3575761, 0.1804375],
                    [0.2126729, 0.7151522, 0.0721750],
                    [0.0193339, 0.1191920, 0.9503041]])

    # Calcul du produit matriciel pour obtenir XYZ
    XYZ = np.dot(mat, norm_pixel)

    return XYZ

def PIXELRGB_to_LAB(pixel):
    norm_pixel = RGB_to_NormalizedRGB(pixel)
    XYZ = RGB_to_XYZ(norm_pixel)

    def f(t):
        delta = 6/29
        return t ** (1/3) if t > delta ** 3 else (t / (3 * delta ** 2)) + (4 / 29)

    X, Y, Z = XYZ / np.array([0.95047, 1.00000, 1.08883])  
    L = 116 * f(Y) - 16
    a = 500 * (f(X) - f(Y))
    b = 200 * (f(Y) - f(Z))

    return np.round(np.array([L, a, b]), 3)

def RGB_to_LAB(img):
  return np.apply_along_axis(PIXELRGB_to_LAB, 0, img)


def PIXELRGB_to_LUV(pixel):
    norm_pixel = RGB_to_NormalizedRGB(pixel)
    XYZ = RGB_to_XYZ(norm_pixel)

    X, Y, Z = XYZ
    Y_ref = 1.00000

    if Y / Y_ref > (6/29) ** 3:
        L = 116 * (Y / Y_ref) ** (1/3) - 16
    else:
        L = (Y / Y_ref) * (29/3) ** 3

    u_prime = 4 * X / (X + 15 * Y + 3 * Z)
    v_prime = 9 * Y / (X + 15 * Y + 3 * Z)

    u_ref = 4 * 0.95047 / (0.95047 + 15 * 1.00000 + 3 * 1.08883)
    v_ref = 9 * 1.00000 / (0.95047 + 15 * 1.00000 + 3 * 1.08883)

    u = 13 * L * (u_prime - u_ref)
    v = 13 * L * (v_prime - v_ref)

    return np.round(np.array([L, u, v]), 3)

def RGB_to_LUV(img):
    return np.apply_along_axis(PIXELRGB_to_LUV, 0, img)


def PIXELRGB_to_CMYK(pixel):
    norm_pixel = RGB_to_NormalizedRGB(pixel)
    C = 1 - norm_pixel[0]
    M = 1 - norm_pixel[1]
    Y = 1 - norm_pixel[2]
    K = min(C, M, Y)

    if K == 1:
        C, M, Y = 0, 0, 0
    else:
        C = (C - K) / (1 - K)
        M = (M - K) / (1 - K)
        Y = (Y - K) / (1 - K)

    return np.round(np.array([C, M, Y, K]), 3)

def RGB_to_CMYK(img):
    return np.apply_along_axis(PIXELRGB_to_CMYK, 0, img)


def PIXELYUV_to_RGB(yuv_pixel):
    Y, U, V = yuv_pixel
    R = Y + 1.13983 * V
    G = Y - 0.39465 * U - 0.58060 * V
    B = Y + 2.03211 * U

    return np.round(np.clip(np.array([R, G, B]) * 255, 0, 255))

def Yuv_to_RGB(img):
    return np.apply_along_axis(PIXELYUV_to_RGB, 0, img)


def PIXELI1I2I3_to_RGB(i1i2i3_pixel):
    I1, I2, I3 = i1i2i3_pixel
    R = I1 + I2 + I3
    G = I1 - I2 - I3
    B = I1 - I3 + 2 * I2

    return np.round(np.clip(np.array([R, G, B]), 0, 255))

def I1I2I3_to_RGB(img):
    return np.apply_along_axis(PIXELI1I2I3_to_RGB, 0, img)

def PIXELYiq_to_RGB(pixel):
    mat = np.array([[1.0, 0.956, 0.621],
                    [1.0, -0.272, -0.647],
                    [1.0, -1.106, 1.703]])
    
    rgb = np.dot(mat, pixel)
    
    rgb = np.clip(rgb * 255, 0, 255)
    
    return np.round(rgb)

def Yiq_to_RGB(img):
    return np.apply_along_axis(PIXELYiq_to_RGB, 0, img)
def NormalizedRGB_to_RGB(img):
  return np.round(np.clip(img * 255, 0, 255))

def PIXELHSV_to_RGB(hsv_pixel):
    H, S, V = hsv_pixel
    C = V * S
    X = C * (1 - abs((H / 60) % 2 - 1))
    m = V - C

    if 0 <= H < 60:
        R, G, B = C, X, 0
    elif 60 <= H < 120:
        R, G, B = X, C, 0
    elif 120 <= H < 180:
        R, G, B = 0, C, X
    elif 180 <= H < 240:
        R, G, B = 0, X, C
    elif 240 <= H < 300:
        R, G, B = X, 0, C
    else:
        R, G, B = C, 0, X

    R, G, B = R + m, G + m, B + m
    return np.round(np.clip(np.array([R, G, B]) * 255, 0, 255))

def HSV_to_RGB(img):
    return np.apply_along_axis(PIXELHSV_to_RGB, 0, img)


def PIXELHSL_to_RGB(hsl_pixel):
    H, S, L = hsl_pixel
    C = (1 - abs(2 * L - 1)) * S
    X = C * (1 - abs((H / 60) % 2 - 1))
    m = L - C / 2

    if 0 <= H < 60:
        R, G, B = C, X, 0
    elif 60 <= H < 120:
        R, G, B = X, C, 0
    elif 120 <= H < 180:
        R, G, B = 0, C, X
    elif 180 <= H < 240:
        R, G, B = 0, X, C
    elif 240 <= H < 300:
        R, G, B = X, 0, C
    else:
        R, G, B = C, 0, X

    R, G, B = R + m, G + m, B + m
    return np.round(np.clip(np.array([R, G, B]) * 255, 0, 255))

def HSL_to_RGB(img):
    return np.apply_along_axis(PIXELHSL_to_RGB, 0, img)


def PIXELLAB_to_RGB(lab_pixel):
    L, a, b = lab_pixel
    Y = (L + 16) / 116
    X = a / 500 + Y
    Z = Y - b / 200

    X = 0.95047 * ((X ** 3) if X > 0.206897 else (X - 16 / 116) / 7.787)
    Y = 1.00000 * ((Y ** 3) if Y > 0.206897 else (Y - 16 / 116) / 7.787)
    Z = 1.08883 * ((Z ** 3) if Z > 0.206897 else (Z - 16 / 116) / 7.787)

    RGB = np.dot(np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]]), [X, Y, Z])
    return np.round(np.clip(RGB * 255, 0, 255))

def LAB_to_RGB(img):
    return np.apply_along_axis(PIXELLAB_to_RGB, 0, img)


def PIXELLUV_to_RGB(luv_pixel):
    L, u, v = luv_pixel
    Y = (L + 16) / 116 if L > 7.9996 else L / (29 / 3) ** 3
    u_ref = 4 * 0.95047 / (0.95047 + 15 * 1.00000 + 3 * 1.08883)
    v_ref = 9 * 1.00000 / (0.95047 + 15 * 1.00000 + 3 * 1.08883)

    u_prime = u / (13 * L) + u_ref
    v_prime = v / (13 * L) + v_ref

    X = Y * 9 * u_prime / (4 * v_prime)
    Z = Y * (12 - 3 * u_prime - 20 * v_prime) / (4 * v_prime)

    RGB = np.dot(np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]]), [X, Y, Z])
    return np.round(np.clip(RGB * 255, 0, 255))

def LUV_to_RGB(img):
    return np.apply_along_axis(PIXELLUV_to_RGB, 0, img)


def PIXELCMYK_to_RGB(cmyk_pixel):
    C, M, Y, K = cmyk_pixel
    R = 255 * (1 - C) * (1 - K)
    G = 255 * (1 - M) * (1 - K)
    B = 255 * (1 - Y) * (1 - K)
    
    return np.round(np.clip(np.array([R, G, B]), 0, 255))

def CMYK_to_RGB(img):
    return np.apply_along_axis(PIXELCMYK_to_RGB, 0, img)

def index(img,QR,QV,QB):
    #calculer les pas
    pr ,pv,pb = 256/QR,256/QV,256/QB
    #les indexes de chaque pixel le pixel [0] à pour index (1,0,0) voir photo
    indexesR = np.array(img[0,:,:])//pr
    indexesV = np.array(img[1,:,:])//pv
    indexesB = np.array(img[2,:,:])//pb

    return (indexesB*QR*QV) + (indexesV*QR)+ indexesR