import numpy as np

discount_factor = 0.99 # discount factor pentru recompensă

def functia_sigmoidala(x):
  return 1.0 / (1.0 + np.exp(-x))

def procesare_imagine(img):
  """ preprocesam 210x160x3 uint8 frame într-un vector unidimensional de dimensiune: 6400 (80x80) """
  img = img[35:195] # tăiem 35px de la început și 15px de la sfârșitul imaginii (părțile unde mingea trece de paletă)
  img = img[::2,::2,0] # micșoră imaginea.
  img[img == 144] = 0 # ștergem background-ul
  img[img == 109] = 0 # la fel
  img[img != 0] = 1 # restul (paletele, mingea) le setăm 1 (de culoare albă).
  return img.astype(np.float).ravel() # ravel transformă array-ul nostru într-o matrice coloană.

def modelare_recompense(recompense):
  indice_recompensa = 0
  recompense_modificate = np.zeros_like(recompense)
  for r in reversed(range(0, recompense.size)):
    if recompense[r] != 0: indice_recompensa = 0 # resetăm suma, deoarece s-a terminat un joc (cineva a primit un punct).
    indice_recompensa = discount_factor * indice_recompensa + recompense[r]
    recompense_modificate[r] = indice_recompensa
  return recompense_modificate

def transfer_cache(cache1, cache2, cache3, cache4):
      return np.vstack(cache1), np.vstack(cache2), np.vstack(cache3), np.vstack(cache4)
