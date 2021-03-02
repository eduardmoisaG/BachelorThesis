import numpy as np
import pickle
import gym
import csv
import datetime
import functii as f
from gym import wrappers

nr_neuroni_strat_ascuns = 200 
dimensiune_pachet_episoade = 10 # folosit la actualizarea parametrilor cu RMSprop la fiecare 10 episoade.
rata_invatare = 1e-3 # rata de învățare folosită la RMS prop
decay_rate = 0.99 # factor descompunere pentru RMSProp
SUS = 2
JOS = 3
'''
with open('score.csv', 'a') as csvFile:
      writer = csv.writer(csvFile)
      writer.writerow(['Numar episod', 'scor', 'Data'])
      csvFile.close()'''

def activare_neuron(imagine_procesata):
  """Implementare pentru forward propagation"""
  produs_imagine_strat_ascuns = np.dot(multime_date_antrenare['W1'], imagine_procesata) # (200 x 6400) . (6400 x 1) = (200 x 1)
  produs_imagine_strat_ascuns[produs_imagine_strat_ascuns<0] = 0 # aplicăm ReLU
  produs_strat_ascuns_iesire = np.dot(multime_date_antrenare['W2'], produs_imagine_strat_ascuns) # (1 x 200) . (200 x 1) = 1 (scalar)
  sus_prob = f.functia_sigmoidala(produs_strat_ascuns_iesire)  # folosim funcția sigmoid definită mai sus pentru a transforma probabilitatea într-o valoare cuprinsa între 0 și 1
  return sus_prob, produs_imagine_strat_ascuns # returnăm probabilitatea de a muta paleta în sus și starea din stratul ascuns.

def modificare_ponderi_dupa_gradient(sus_prob, strat_ascuns_episod, imagine_episod):

  derivate_ponderi2 = np.dot(strat_ascuns_episod.T, sus_prob).ravel()
  dh = np.outer(sus_prob, multime_date_antrenare['W2'])
  dh[strat_ascuns_episod <= 0] = 0 # backprop relu
  derivate_ponderi1 = np.dot(dh.T, imagine_episod)
  return {'W1':derivate_ponderi1, 'W2':derivate_ponderi2}

continuare_antrenament = True # reia antrenamentul de la un anumit punct precedent (din fisierul save.p )?
afiseaza_env = True # afisare joc?

# initializarea modelului
dimensiune_model = 80 * 80 # dimensiunea: 80x80
if continuare_antrenament == False:
  multime_date_antrenare = {}
  multime_date_antrenare['W1'] = np.random.randn(nr_neuroni_strat_ascuns, dimensiune_model) / np.sqrt(dimensiune_model) # - Forma va fi H x D
  multime_date_antrenare['W2'] = np.random.randn(nr_neuroni_strat_ascuns) / np.sqrt(nr_neuroni_strat_ascuns) #Forma va fi H
else:
  multime_date_antrenare = pickle.load(open('save.p', 'rb'))
  

buffer_gradient = { k : np.zeros_like(v) for k,v in multime_date_antrenare.items() } 
cache_rmsprop = { k : np.zeros_like(v) for k,v in multime_date_antrenare.items() } 

env = gym.make("Pong-v0")
env = wrappers.Monitor(env, 'tmp/pong-base', force=True)
observatie = env.reset()
frame_anterior = None # folosit să calculăm diferența dintre frame-uri
cache_imagini,cache_straturi_ascunse,cache_probabilitati,cache_recompense = [],[],[],[]
suma_recompense = 0
numar_episod = 0
while True:
  if afiseaza_env: env.render()

  frame_curent = f.procesare_imagine(observatie)
  # luam diferenta dintre pixeli, deoarece este mai probabil ca aceasta să reprezinte informații care ne pot ajuta.
  if frame_anterior is not None:
    diferenta_frame = frame_curent - frame_anterior 
  else:
    diferenta_frame = np.zeros(dimensiune_model)
  frame_anterior = frame_curent

  # calculăm probabilitatea de a muta paleta în sus
  sus_prob, neroni_strat_ascuns = activare_neuron(diferenta_frame)
  # În continuare alegem un număr aleator între 0 și 1
  # Dacă numărul este mai mare decât probabilitatea de a muta paleta în sus pentru imaginea furnizată retelei neuronale, 
  # atunci o mutăm în jos. (Exploatare și Explorare).
  if np.random.uniform() < sus_prob:
    actiune_aleasa = SUS
  else:
    actiune_aleasa = JOS

  cache_imagini.append(diferenta_frame) 
  cache_straturi_ascunse.append(neroni_strat_ascuns) 
  if actiune_aleasa == 2:
    y = 1
  else:
    y = 0

  cache_probabilitati.append(y - sus_prob) # gradient care incurajează acțiunea ce a fost aleasa, sa fie aleasa și în viitor.

  # efectuăm acțiunea aleasă si salvăm noi date
  observatie, recompensa, done, info = env.step(actiune_aleasa)
  suma_recompense += recompensa
  cache_recompense.append(recompensa) # salvăm recompensa (facem asta după ce am ales acțiunea să știm ce recompensă am obținut)

  if done: # dacă un episod s-a terminat
    numar_episod += 1
    # salvăm împreuna toate datele de intrare, starile din stratul ascuns, acțiunile și recompensele pentru respectivul episod
    Ep_cache_imagini, Ep_cache_straturi_ascunse, Ep_cache_probabilitati, Ep_cache_recompense = f.transfer_cache(cache_imagini, cache_straturi_ascunse, cache_probabilitati, cache_recompense)

    cache_imagini,cache_straturi_ascunse,cache_probabilitati,cache_recompense = [],[],[],[] # resetăm memoria

    # calculăm recompensa înapoi in timp (de la momentul când am câștigat/pierdut inapoi) 
    # pentru ca cele mai apropiate acțiuni de când s-a terminat un joc sa aibă o influență mai mare.
    recompense_modelate = f.modelare_recompense(Ep_cache_recompense)
    recompense_modelate -= np.mean(recompense_modelate)
    recompense_modelate /= np.std(recompense_modelate)

    Ep_cache_probabilitati *= recompense_modelate #
    grad = modificare_ponderi_dupa_gradient(Ep_cache_probabilitati, Ep_cache_straturi_ascunse, Ep_cache_imagini)
    for k in multime_date_antrenare: buffer_gradient[k] += grad[k] 
    if numar_episod % dimensiune_pachet_episoade == 0:
      for k,v in multime_date_antrenare.items():
        g = buffer_gradient[k] # gradient
        cache_rmsprop[k] = decay_rate * cache_rmsprop[k] + (1 - decay_rate) * g**2
        multime_date_antrenare[k] += rata_invatare * g / (np.sqrt(cache_rmsprop[k]) + 1e-3)
        buffer_gradient[k] = np.zeros_like(v) 


    print ('Scor = %f' % (suma_recompense))
    '''
    row = [numar_episod, suma_recompense, datetime.datetime.now()]

    with open('score.csv', 'a') as csvFile:
      writer = csv.writer(csvFile)
      writer.writerow(row)
    csvFile.close()'''
    if numar_episod % 100 == 0: pickle.dump(multime_date_antrenare, open('save.p', 'wb'))
    suma_recompense = 0
    observatie = env.reset() # reset env
    frame_anterior = None

  if recompensa != 0: 
    print ('Numar episod %d: joc terminat, recompensa: %f' % (numar_episod, recompensa) + (' LOSE' if recompensa == -1 else ' WIN'))