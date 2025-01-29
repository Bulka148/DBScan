import pygame
import sys
import numpy as np
import cv2
from sklearn.cluster import KMeans
from tensorflow import keras

# Загрузка предварительно обученной модели
model = keras.models.load_model('path_to_your_pretrained_mnist_model.h5')

# Инициализация Pygame
pygame.init()

# Настройки экрана
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Рисование и распознавание чисел")

# Цвета
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Переменные
drawing = False
points = []

# Основной цикл программы
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                drawing = True
                points.append(event.pos)
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                drawing = False
        elif event.type == pygame.MOUSEMOTION:
            if drawing:
                points.append(event.pos)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_DELETE:
                points = []
            elif event.key == pygame.K_RETURN:
                # Обработка Enter: сегментация и распознавание
                if points:
                    image = prepare_image(points, width, height)
                    digits = segment_image(image)
                    recognized_numbers = [recognize_digit(d) for d in digits]
                    print("Распознанные числа:", recognized_numbers)
                points = []

    screen.fill(BLACK)
    if len(points) > 1:
        pygame.draw.lines(screen, WHITE, False, points, 2)
    pygame.display.flip()

pygame.quit()
sys.exit()

def prepare_image(points, width, height):
    """Создание изображения из собранных точек."""
    image = np.zeros((height, width), dtype=np.uint8)
    if points:
        for i in range(len(points) - 1):
            pygame.draw.line(image, WHITE, points[i], points[i+1], 2)
    return image

def segment_image(image):
    """Сегментация изображения на основе кластеризации."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Используем алгоритм K-Means для сегментации
    X = np.column_stack(np.where(threshold > 0))
    kmeans = KMeans(n_clusters=min(10, len(X)//10))  # число кластеров можно подобрать
    kmeans.fit(X)
    labels = kmeans.labels_

    # Разделение на компоненты
    components = [X[labels == i] for i in range(kmeans.n_clusters)]
    digits = [get_bounding_box(c) for c in components if len(c) > 10]  # создаём изображения по компонентам
    return digits

def get_bounding_box(component):
    """Создание изображения цифры по компоненту."""
    x, y = component[:, 1], component[:, 0]
    digit_image = np.zeros_like(image)
    digit_image[y, x] = 255
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    return digit_image[y_min:y_max+1, x_min:x_max+1]

def recognize_digit(digit_image):
    """Распознавание отдельной цифры с использованием обученной модели."""
    digit_image = cv2.resize(digit_image, (28, 28), interpolation=cv2.INTER_AREA)
    digit_image = digit_image.astype('float32') / 255.
    digit_image = np.expand_dims(digit_image, axis=(0, -1))
    prediction = model.predict(digit_image)
    return np.argmax(prediction)





