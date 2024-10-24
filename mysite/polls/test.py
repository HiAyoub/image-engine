import numpy as np
def rgb_2_gray709(img):
  print('709')
  return np.sum(img*np.array([0.2125,0.7154,0.0721]).reshape(img.shape[0],1,1) , axis = 0)

tab = np.array([[[1,1,1,1,1,1],[2,2,2,2,2,2],[3,3,3,3,3,3]],[[1,1,1,1,1,1],[2,2,2,2,2,2],[3,3,3,3,3,3]],[[1,1,1,1,1,1],[2,2,2,2,2,2],[3,3,3,3,3,3]]])
print(tab[0].shape)
print(rgb_2_gray709(tab))