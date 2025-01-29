import random
import matplotlib.pyplot as plt
import numpy as np


# Генерация случайных координат городов с учетом минимального расстояния
def generate_random_coordinates(num_cities, min_distance):

    coordinates_ = []

    while len(coordinates_) < num_cities:
        x, y = random.uniform(0, 100), random.uniform(0, 100)
        # Проверка, что новое место подходит
        if all(np.sqrt((x - cx) ** 2 + (y - cy) ** 2) >= min_distance for cx, cy in coordinates_):
            coordinates_.append((x, y))

    return coordinates_


# Расчет евклидова расстояния между двумя точками
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


# Создание матрицы расстояний между всеми городами
def create_distance_matrix(coordinates__):
    return [[calculate_distance(p1, p2) for p2 in coordinates__] for p1 in coordinates__]


# Инициализация координат городов и матрицы расстояний
coordinates = generate_random_coordinates(6, 10)
distance_matrix = create_distance_matrix(coordinates)

# Переменная для хранения лучшего найденного пути
best_solution = None


# Алгоритм решения задачи коммивояжера с использованием генетического алгоритма
def solve_tsp_with_genetic_algorithm(population_size, number_of_parents, generations):
    global best_solution

    # Генерация случайного пути (перестановка городов)
    def generate_random_path():
        path = list(range(1, len(coordinates)))
        random.shuffle(path)
        return path

    # Расчет длины пути на основе матрицы расстояний
    def calculate_path_length(path):
        distances = [distance_matrix[path[i - 1]][path[i]] for i in range(1, len(path))]
        # Возвращение в начальный город
        distances.append(distance_matrix[path[-1]][path[0]])
        return sum(distances)

    # Инициализация начального поколения
    population = [[0] + generate_random_path() for _ in range(population_size)]

    for _ in range(generations):
        # Селекция: отбор лучших родителей
        parents = sorted(population, key=calculate_path_length)[:number_of_parents]
        new_population = []

        # Скрещивание: создание потомков из родителей
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                for __ in range((population_size // number_of_parents - 1) * 2):
                    child = []
                    for j in range(len(parents[i])):
                        # Если нет гена у родителя 1 и 2, то выбираем случайное
                        if parents[i][j] not in child and parents[i + 1][j] not in child:
                            child.append(random.choice([parents[i][j], parents[i + 1][j]]))
                        # Если ген имеется только у одного родителя, то выбираем его
                        elif parents[i][j] not in child or parents[i + 1][j] not in child:
                            child.append(parents[i][j] if parents[i][j] not in child else parents[i + 1][j])
                        # Иначе берём свободный
                        else:
                            child.append(next(g for g in range(1, len(coordinates)) if g not in child))

                    # Мутация: случайная перестановка городов
                    if random.random() < 0.1:
                        idx1, idx2 = random.sample(range(1, len(child)), 2)
                        child[idx1], child[idx2] = child[idx2], child[idx1]
                    new_population.append(child)

        # Замена старого поколения на новое
        population = parents + new_population

    # Поиск лучшего найденного решения
    best_path = sorted(population, key=calculate_path_length)[0]
    best_solution = (float(calculate_path_length(best_path)), best_path)


# Визуализация маршрута решения
def visualize_solution(coordinates, best_path, best_path_length):
    plt.figure(figsize=(8, 6))

    # Отрисовка всех возможных путей (серый)
    for i, start in enumerate(coordinates):
        for j, end in enumerate(coordinates):
            if i != j:
                plt.plot([start[0], end[0]], [start[1], end[1]], 'gray', alpha=0.3)

    # Отрисовка оптимального пути (синий)
    for i in range(len(best_path)):
        start = coordinates[best_path[i]]
        end = coordinates[best_path[(i + 1) % len(best_path)]]  # Замкнутый путь
        plt.plot([start[0], end[0]], [start[1], end[1]], 'b', linewidth=2)

    # Отрисовка городов
    for i, (x, y) in enumerate(coordinates):
        plt.scatter(x, y, color='red')
        plt.text(x, y, str(i), fontsize=12, ha='right')

    # Добавление текста с маршрутом и расстоянием
    route_str = ' -> '.join(map(str, best_path)) + f" -> {best_path[0]}"
    plt.text(5, 110, f"Расстояние = {best_path_length:.2f}\nМаршрут: ({route_str})", fontsize=12, color='black', ha='left')

    plt.title("Оптимальный путь (замкнутый)")
    plt.xlim(0, 100)
    plt.ylim(0, 120)
    plt.grid(True)
    plt.show()


# Запуск генетического алгоритма
solve_tsp_with_genetic_algorithm(1000, 100, 10)

# Вывод и визуализация оптимального маршрута
if best_solution:
    print(f"Оптимальный путь: расстояние = {best_solution[0]:.2f}, маршрут = ({' -> '.join(map(str, best_solution[1]))} -> {best_solution[1][0]})")
    visualize_solution(coordinates, best_solution[1], best_solution[0])
