import numpy as np
from PIL import Image, ImageDraw
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


""" Parámetros del algoritmo genético """

TAM_POBLACION = 100         # Número de individuos por generación
NUM_POLIGONO = 90           # Cantidad de polígonos por individuo, ADN
VERT_POLIGONO = 4           # Cantidad vértices en los polígonos
TASA_MUTACION = 0.04        # Probabilidad de mutación sobre los poligonos en cada individuo 
GENERACIONES = 3000         # Cantidad de generaciones 
TAM_IMAGEN = (256, 256)     # Tamaño de la imagen de referencia
TAM_INICIAL_POLIGONO = 30   # Tamaño inicial para los polígonos
TAM_FINAL_POLIGONO = 10     # Tamaño final de los polígonos

""" Cargar imagen de referencia en escala de grises """
ima_referencia = Image.open("imagen.png").convert("L")
ima_referencia = ima_referencia.resize(TAM_IMAGEN)
array_referencia = np.array(ima_referencia)

# Función para calcular el tamaño de los polígonos en función de la generación
def calcular_tam_poligono(generacion):
    return max(TAM_INICIAL_POLIGONO - (generacion * (TAM_INICIAL_POLIGONO - TAM_FINAL_POLIGONO) / GENERACIONES), TAM_FINAL_POLIGONO)

# Función para generar un polígono aleatorio con tamaño adaptable
"""  Genera un polígono aleatorio dentro de los
límites de la imagen, con un tamaño adaptable. 
Tomando en cuenta la cantidad de vertices dados. """

def random_poligono(tamano):
    return [(random.randint(0, TAM_IMAGEN[0]), random.randint(0, TAM_IMAGEN[1])) for _ in range(VERT_POLIGONO)]

# Función para generar un individuo
""" Genera un individuo, que es una lista de polígonos y 
colores aleatorios. Cada individuo tiene NUM_POLIGONO 
polígonos y un color entre 0 y 255 (escala de grises). """

def crear_individuo(tamano):
    return [(random_poligono(tamano), random.randint(0, 255)) for _ in range(NUM_POLIGONO)]

# Función para dibujar un individuo en una imagen
def dibujar_individuo(individuo):
    img = Image.new('L', TAM_IMAGEN, 255)
    dibujar = ImageDraw.Draw(img)
    for polygon, color in individuo:
        dibujar.polygon(polygon, fill=color)
    return np.array(img)

# Función de evaluación (fitness)
def evaluar_individuo(individuo):
    dibujo_img = dibujar_individuo(individuo)
    return np.sum(np.abs(array_referencia - dibujo_img))

# Selección de padres basada en torneo
def seleccion_padres(poblacion, k=3):
    return min(random.sample(poblacion, k), key=lambda ind: ind[1])

# Cruza de dos individuos (crossover)
def cruce(padre1, padre2):
    point = random.randint(1, len(padre1) - 1)
    hijo = padre1[:point] + padre2[point:]
    return hijo

# Mutación de un individuo con variación de tamaños de polígonos
def mutacion(individuo, tamano, generacion):
    for i in range(len(individuo)):
        if random.random() < TASA_MUTACION:
            # Variar el tamaño del polígono antes de mutar
            new_size = calcular_tam_poligono(generacion) + random.randint(-5, 5)  # Tamaño variable con mutación
            if random.random() < 0.5:  # Modificar el polígono actual
                individuo[i] = (random_poligono(new_size), random.randint(0, 255))
            else:  # Duplicar y modificar ligeramente un polígono
                duplicated_polygon = individuo[i][0][:]
                individuo.append((duplicated_polygon, random.randint(0, 255)))
                if len(individuo) > NUM_POLIGONO:  # Limitar el número de polígonos
                    individuo.pop(random.randint(0, len(individuo) - 1))
    return individuo

# Crear la población inicial
poblacion = [(crear_individuo(TAM_INICIAL_POLIGONO), 0) for _ in range(TAM_POBLACION)]

# Evaluar la población inicial
poblacion = [(individuo, evaluar_individuo(individuo)) for individuo, _ in poblacion]

# Preparar la animación
fig, ax = plt.subplots()
ax.axis('off')
img_display = ax.imshow(dibujar_individuo(poblacion[0][0]), cmap='gray', vmin=0, vmax=255)

def update(frame):
    global poblacion

    # Calcular el tamaño de los polígonos para esta generación
    tam_poligono_actual = calcular_tam_poligono(frame)

    # Selección y cruce
    nueva_poblacion = []
    for _ in range(TAM_POBLACION):
        padre1 = seleccion_padres(poblacion)
        padre2 = seleccion_padres(poblacion)
        hijo = cruce(padre1[0], padre2[0])
        nueva_poblacion.append((hijo, 0))
    
    # Mutación y evaluación
    nueva_poblacion = [(mutacion(individuo, tam_poligono_actual, frame), 0) for individuo, _ in nueva_poblacion]
    nueva_poblacion = [(individuo, evaluar_individuo(individuo)) for individuo, _ in nueva_poblacion]
    
    # Reemplazar la población antigua
    poblacion = sorted(nueva_poblacion + poblacion, key=lambda ind: ind[1])[:TAM_POBLACION]
    
    # Mostrar la mejor aproximación de la generación actual
    mejor_individuo, _ = poblacion[0]
    img_display.set_data(dibujar_individuo(mejor_individuo))
    ax.set_title(f'Generación {frame + 1}')

# Crear la animación
animacion = FuncAnimation(fig, update, frames=GENERACIONES, repeat=False)
plt.show()
