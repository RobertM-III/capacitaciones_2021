#!/usr/bin/env python

"""
Este programa implementa un freno de emergencia para evitar accidentes en Duckietown.
"""

import sys
import argparse
import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
import numpy as np
import cv2

def mov_duckiebot(key):
    # La acción de Duckiebot consiste en dos valores:
    # velocidad lineal y velocidad de giro
    actions = {ord('w'): np.array([1.0, 0.0]),
               ord('s'): np.array([-1.0, 0.0]),
               ord('a'): np.array([0.0, 1.0]),
               ord('d'): np.array([0.0, -1.0]),
               ord('q'): np.array([0.3, 1.0]),
               ord('e'): np.array([0.3, -1.0])
               }

    action = actions.get(key, np.array([0.0, 0.0]))
    return action

def det_duckie(imagen):
    ### DETECTOR HECHO EN LA MISIÓN ANTERIOR
    limite_inferior = np.array([25,216,170]) 
    limite_superior = np.array([31,255,255])
    min_area = 2500
    transformacion = cv2.cvtColor(imagen, cv2.COLOR_RGB2HSV)
    mascara = cv2.inRange(transformacion, limite_inferior, limite_superior)
    enmascarada = cv2.bitwise_and(transformacion, transformacion, mask = mascara)
    contorno, hierarchy = cv2.findContours(mascara, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    detecciones = list()

    for cnt in contorno:
        x1, y1, ancho, altura = cv2.boundingRect(cnt)
        AREA = 5*ancho*altura
        if AREA > min_area:
            detecciones.append((x1,y1,ancho,altura))

    return detecciones

def draw_dets(imagen, detecciones):
    for d in detecciones:
        x1, y1 = d[0], d[1]
        x2 = x1 + d[2]
        y2 = y1 + d[3]
        cv2.rectangle(imagen, (int(x1), int(y1)), (int(x2),int(y2)), (0,255,0), 3)

    return imagen

def red_alert(imagen):
    red_img = np.zeros((480, 640, 3), dtype = np.uint8)
    red_img[:,:,0] = 255
    blend = cv2.addWeighted(imagen, 0.5, red_img, 0.5, 0)

    return blend

if __name__ == '__main__':

    # Se leen los argumentos de entrada
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default="Duckietown-udem1-v1")
    # Los datos que se printean son solo validos para el mapa "free"
    parser.add_argument('--map-name', default='udem1')
    args = parser.parse_args()

    # Definición del environment
    if args.env_name and args.env_name.find('Duckietown') != -1:
        env = DuckietownEnv(
            map_name = args.map_name,
            domain_rand = False,
        )
    else:
        env = gym.make(args.env_name)

    # Se reinicia el environment
    env.reset()

    # Inicialmente no hay alerta
    alert = False

    # Posición del pato en el mapa (fija)
    duck_pos = np.array([2,0,2])

    # Constante que se debe calcular
    C = 0.08*800 # f * Hr (f es constante, Hr es conocido)

    while True:

        # Captura la tecla que está siendo apretada y almacena su valor en key
        key = cv2.waitKey(0)
        # Si la tecla es Esc, se sale del loop y termina el programa
        if key == 27:
            break

        # Se define la acción dada la tecla presionada
        action = mov_duckiebot(key)

        # Si hay alerta evitar que el Duckiebot avance
        if alert:
            action = np.array([-1.0, 0.0])

        # Se ejecuta la acción definida anteriormente y se retorna la observación (obs),
        # la evaluación (reward), etc
        obs, reward, done, info = env.step(action)

        # Detección de patos, retorna lista de detecciones
        lista_detecciones = det_duckie(obs)

        # Dibuja las detecciones
        dibujo = draw_dets(obs,lista_detecciones)

        # Obtener posición del duckiebot
        dbot_pos = env.cur_pos
        # Calcular distancia real entre posición del duckiebot y pato
        # esta distancia se utiliza para calcular la constante
        dist = ((duck_pos[0]-dbot_pos[0])**2+(duck_pos[1]-dbot_pos[1])**2+(duck_pos[2]-dbot_pos[2])**2)**(1/2)

        # Use det_pato.py ( el modulo del desafio 3) para obtener la distancia focal aproximada, que si bien cambiaba
        # dependiendo de la distancia real, decidi usar que f = 800 ( en pixeles ) ya que me parecio que para distancias
        # cortas seria una relativamente buena aproximacion y que seria util para efectos de este desafio.

        # La alerta se desactiva (opción por defecto)
        alert = False
        
        for d in lista_detecciones:
            # Alto de la detección en pixeles
            p = d[3]
            # La aproximación se calcula según la fórmula mostrada en la capacitación
            d_aprox = C/p

            if p > 69: # Este es un filtro para que solo me muestre el rectangulo grande ( el pato entero )
                # y no los sub-rectangulitos que tiene el pato ( que se identifican como algo aparte por el tema del
                # rango de colores) que al parecer hacen que me arrojen distancias aproximadas erroneas y que de
                # esa forma no me funcione bien el programa, entonces con esto lo arreglo, aunque la contraparte es
                # que cuando se esta muy lejos no muestra los datos relevante en pantalla, aun así, es suficiente para
                # cumplir con el objetivo principal de este desafio, que es frenar cuando se esta muy cerca de un Duckie.

                # Muestra información relevante
                print('p:', p)
                print('Da:', d_aprox)
                print('Dr:', dist)

                # Si la distancia es muy pequeña activa alerta
                if d_aprox < 0.3:
                    
                    # Activar alarma
                    alert = True
                    
                    # Muestra ventana en rojo
                    dibujo = red_alert(dibujo)

        # Se muestra en una ventana llamada "patos" la observación del simulador
        cv2.imshow('patos', cv2.cvtColor(dibujo, cv2.COLOR_RGB2BGR))

    # Se cierra el environment y termina el programa
    env.close()
