import numpy as np
from PIL import Image, ImageDraw
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Configuraciones
num_polygons = 100
num_vertices = 3  # Triángulos
mutation_rate = 0.01
population_size = 60
generations = 10000
image_size = (128, 128)

# Cargar la imagen de referencia
reference_image = Image.open('reference_4.png').convert('L')
reference_image = reference_image.resize(image_size)

def generate_random_polygon():
    size_factor = random.random()
    polygon = [(int(random.randint(0, image_size[0]) * size_factor), 
                int(random.randint(0, image_size[1]) * size_factor)) for _ in range(num_vertices)]
    
    # Obtener el color promedio del área cubierta por el polígono
    mask = Image.new('L', image_size, 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(polygon, fill=255)
    mask = np.array(mask)
    color = int(np.mean(np.array(reference_image)[mask == 255]))  # Promedio de color en el área cubierta
    return polygon, color

def mutate_polygon(polygon_data):
    polygon, color = polygon_data
    if random.random() < mutation_rate:
        index = random.randint(0, len(polygon) - 1)
        # Mutación hacia los bordes del polígono para ajustarse a la imagen
        polygon[index] = (random.randint(0, image_size[0]), random.randint(0, image_size[1]))
    if random.random() < mutation_rate:
        # Recalcular el color basándose en la nueva posición del polígono
        mask = Image.new('L', image_size, 0)
        draw = ImageDraw.Draw(mask)
        draw.polygon(polygon, fill=255)
        mask = np.array(mask)
        color = int(np.mean(np.array(reference_image)[mask == 255]))
    return polygon, color

def create_individual():
    return [generate_random_polygon() for _ in range(num_polygons)]

def draw_individual(individual):
    img = Image.new('L', image_size, 255)  # Fondo blanco
    draw = ImageDraw.Draw(img)
    for polygon, color in individual:
        draw.polygon(polygon, fill=color)  # Polígonos con color
    return img

def fitness(individual):
    generated_image = draw_individual(individual)
    return -np.sum(np.abs(np.array(generated_image) - np.array(reference_image)))

def crossover(parent1, parent2):
    point = random.randint(0, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def mutate_individual(individual):
    return [mutate_polygon(polygon_data) for polygon_data in individual]

# Generar la población inicial
population = [create_individual() for _ in range(population_size)]

# Algoritmo Genético con visualización en tiempo real
best_individual = None
best_fitness = float('-inf')

fig, ax = plt.subplots()

def update(frame):
    global population, best_individual, best_fitness

    # Evaluar fitness de la población
    fitness_scores = [(individual, fitness(individual)) for individual in population]
    fitness_scores.sort(key=lambda x: x[1], reverse=True)

    # Guardar el mejor individuo
    if fitness_scores[0][1] > best_fitness:
        best_individual, best_fitness = fitness_scores[0]

    # Selección: seleccionamos los mejores individuos
    selected_individuals = [individual for individual, score in fitness_scores[:population_size // 2]]

    # Crossover y Mutación
    new_population = []
    while len(new_population) < population_size:
        parent1, parent2 = random.sample(selected_individuals, 2)
        child1, child2 = crossover(parent1, parent2)
        new_population.append(mutate_individual(child1))
        new_population.append(mutate_individual(child2))

    population = new_population

    # Actualizar imagen en la animación
    img = draw_individual(best_individual)
    ax.clear()
    ax.imshow(img, cmap='gray')
    ax.set_title(f'Generación {frame} - Mejor Fitness: {best_fitness}')

# Crear animación
ani = animation.FuncAnimation(fig, update, frames=range(generations), repeat=False)
plt.show()
