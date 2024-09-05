import numpy as np
from PIL import Image, ImageDraw
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parámetros del algoritmo genético
POPULATION_SIZE = 100
NUM_POLYGONS = 75
VERTICES_PER_POLYGON = 6
MUTATION_RATE = 0.04
GENERATIONS = 5000
IMAGE_SIZE = (256, 256)
INITIAL_POLYGON_SIZE = 30
FINAL_POLYGON_SIZE = 10

# Cargar imagen de referencia en escala de grises
reference_image = Image.open("imagen.png").convert("L")
reference_image = reference_image.resize(IMAGE_SIZE)
reference_array = np.array(reference_image)

# Función para calcular el tamaño de los polígonos en función de la generación
def calculate_polygon_size(generation):
    return max(INITIAL_POLYGON_SIZE - (generation * (INITIAL_POLYGON_SIZE - FINAL_POLYGON_SIZE) / GENERATIONS), FINAL_POLYGON_SIZE)

# Función para generar un polígono aleatorio con tamaño adaptable
def random_polygon(size):
    return [(random.randint(0, IMAGE_SIZE[0]), random.randint(0, IMAGE_SIZE[1])) for _ in range(VERTICES_PER_POLYGON)]

# Función para generar un individuo
def create_individual(size):
    return [(random_polygon(size), random.randint(0, 255)) for _ in range(NUM_POLYGONS)]

# Función para dibujar un individuo en una imagen
def draw_individual(individual):
    img = Image.new('L', IMAGE_SIZE, 255)
    draw = ImageDraw.Draw(img)
    for polygon, color in individual:
        draw.polygon(polygon, fill=color)
    return np.array(img)

# Función de evaluación (fitness)
def evaluate_individual(individual):
    drawn_image = draw_individual(individual)
    return np.sum(np.abs(reference_array - drawn_image))

# Selección de padres basada en torneo
def tournament_selection(population, k=3):
    return min(random.sample(population, k), key=lambda ind: ind[1])

# Cruza de dos individuos (crossover)
def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child = parent1[:point] + parent2[point:]
    return child

# Mutación de un individuo con variación de tamaños de polígonos
def mutate(individual, size, generation):
    for i in range(len(individual)):
        if random.random() < MUTATION_RATE:
            # Variar el tamaño del polígono antes de mutar
            new_size = calculate_polygon_size(generation) + random.randint(-5, 5)  # Tamaño variable con mutación
            if random.random() < 0.5:  # Modificar el polígono actual
                individual[i] = (random_polygon(new_size), random.randint(0, 255))
            else:  # Duplicar y modificar ligeramente un polígono
                duplicated_polygon = individual[i][0][:]
                individual.append((duplicated_polygon, random.randint(0, 255)))
                if len(individual) > NUM_POLYGONS:  # Limitar el número de polígonos
                    individual.pop(random.randint(0, len(individual) - 1))
    return individual

# Crear la población inicial
population = [(create_individual(INITIAL_POLYGON_SIZE), 0) for _ in range(POPULATION_SIZE)]

# Evaluar la población inicial
population = [(individual, evaluate_individual(individual)) for individual, _ in population]

# Preparar la animación
fig, ax = plt.subplots()
ax.axis('off')
img_display = ax.imshow(draw_individual(population[0][0]), cmap='gray', vmin=0, vmax=255)

def update(frame):
    global population

    # Calcular el tamaño de los polígonos para esta generación
    current_polygon_size = calculate_polygon_size(frame)

    # Selección y cruce
    new_population = []
    for _ in range(POPULATION_SIZE):
        parent1 = tournament_selection(population)
        parent2 = tournament_selection(population)
        child = crossover(parent1[0], parent2[0])
        new_population.append((child, 0))
    
    # Mutación y evaluación
    new_population = [(mutate(individual, current_polygon_size, frame), 0) for individual, _ in new_population]
    new_population = [(individual, evaluate_individual(individual)) for individual, _ in new_population]
    
    # Reemplazar la población antigua
    population = sorted(new_population + population, key=lambda ind: ind[1])[:POPULATION_SIZE]
    
    # Mostrar la mejor aproximación de la generación actual
    best_individual, _ = population[0]
    img_display.set_data(draw_individual(best_individual))
    ax.set_title(f'Generación {frame + 1}')

# Crear la animación
ani = FuncAnimation(fig, update, frames=GENERATIONS, repeat=False)
plt.show()
