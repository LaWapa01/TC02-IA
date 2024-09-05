import numpy as np
from PIL import Image, ImageDraw
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parámetros del algoritmo genético
POPULATION_SIZE = 100
NUM_POLYGONS = 75
VERTICES_PER_POLYGON = 3
MUTATION_RATE = 0.1
GENERATIONS = 5000
IMAGE_SIZE = (256, 256)
INITIAL_POLYGON_SIZE = 30
FINAL_POLYGON_SIZE = 10

# Cargar imagen de referencia en escala de grises
reference_image = Image.open("imagen.png").convert("L")
reference_image = reference_image.resize(IMAGE_SIZE)
reference_array = np.array(reference_image, dtype=np.uint8)

# Función para calcular el tamaño de los polígonos en función de la generación
def calculate_polygon_size(generation):
    return max(INITIAL_POLYGON_SIZE - (generation * (INITIAL_POLYGON_SIZE - FINAL_POLYGON_SIZE) / GENERATIONS), FINAL_POLYGON_SIZE)

# Función para generar un polígono aleatorio con tamaño adaptable
def random_polygon(size):
    return [(random.randint(0, IMAGE_SIZE[0]), random.randint(0, IMAGE_SIZE[1])) for _ in range(VERTICES_PER_POLYGON)]

# Función para generar un individuo (un polígono)
def create_individual(size):
    polygon = random_polygon(size)
    color = random.randint(0, 255)  # Color aleatorio en escala de grises
    return (polygon, color)

# Función para dibujar una lista de individuos (polígonos) en una imagen
def draw_population(population):
    img = Image.new('L', IMAGE_SIZE, 255)
    draw = ImageDraw.Draw(img)
    for polygon, color in population:
        draw.polygon(polygon, fill=color)
    return np.array(img, dtype=np.uint8)

# Función de evaluación (fitness) de un polígono superpuesto
def evaluate_individual(individual, current_image):
    temp_image = Image.fromarray(current_image.astype(np.uint8))  # Asegurarse de que es uint8
    draw = ImageDraw.Draw(temp_image)
    polygon, color = individual
    draw.polygon(polygon, fill=color)
    new_image = np.array(temp_image, dtype=np.uint8)
    return np.sum(np.abs(reference_array - new_image))  # Fitness basado en la diferencia absoluta

# Selección de padres basada en torneo
def tournament_selection(population, fitness, k=3):
    participants = random.sample(list(zip(population, fitness)), k)
    return min(participants, key=lambda x: x[1])[0]

# Cruza de dos individuos (crossover)
def crossover(parent1, parent2):
    point = random.randint(1, len(parent1[0]) - 1)
    child_polygon = parent1[0][:point] + parent2[0][point:]
    child_color = (parent1[1] + parent2[1]) // 2
    return (child_polygon, child_color)

# Mutación de un individuo con variación de tamaños de polígonos
def mutate(individual, size, generation):
    polygon, color = individual
    if random.random() < MUTATION_RATE:
        # Variar el tamaño o forma del polígono
        polygon = random_polygon(size)
        color = random.randint(0, 255)  # Mutar el color
    return (polygon, color)

# Crear la población inicial (con polígonos individuales)
population = [create_individual(INITIAL_POLYGON_SIZE) for _ in range(POPULATION_SIZE)]
fitness = [evaluate_individual(ind, np.full(IMAGE_SIZE, 255, dtype=np.uint8)) for ind in population]

# Preparar la animación
fig, ax = plt.subplots()
ax.axis('off')
current_image = np.full(IMAGE_SIZE, 255, dtype=np.uint8)  # Imagen inicial (blanco)
img_display = ax.imshow(current_image, cmap='gray', vmin=0, vmax=255)

def update(frame):
    global population, fitness, current_image

    # Calcular el tamaño de los polígonos para esta generación
    current_polygon_size = calculate_polygon_size(frame)

    # Crear una nueva generación
    new_population = []
    new_fitness = []

    for _ in range(POPULATION_SIZE):
        # Selección de padres y cruce
        parent1 = tournament_selection(population, fitness)
        parent2 = tournament_selection(population, fitness)
        child = crossover(parent1, parent2)
        
        # Mutación
        child = mutate(child, current_polygon_size, frame)
        
        # Evaluar el nuevo individuo (polígono) sobre la imagen actual
        child_fitness = evaluate_individual(child, current_image)

        new_population.append(child)
        new_fitness.append(child_fitness)

    # Actualizar la población y la imagen actual con los mejores polígonos
    population = new_population
    fitness = new_fitness

    # Seleccionar los mejores polígonos para la imagen final
    best_individuals = sorted(zip(population, fitness), key=lambda x: x[1])[:NUM_POLYGONS]

    # Dibujar la imagen con los mejores polígonos
    current_image = draw_population([ind for ind, fit in best_individuals])
    img_display.set_data(current_image)
    ax.set_title(f'Generación {frame + 1}')

# Crear la animación
ani = FuncAnimation(fig, update, frames=GENERATIONS, repeat=False)
plt.show()
