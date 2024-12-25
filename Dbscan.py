import pygame
import numpy as np
from sklearn.neighbors import NearestNeighbors

EPSILON = 50
MIN_POINTS = 3

pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("DBSCAN Visualizer")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)
    (255, 255, 0)]

points = []
flags = []

def draw_points():
    screen.fill(WHITE)
    for point, flag in zip(points, flags):
        pygame.draw.circle(screen, flag, point, 3)

def draw_reset_button():
    font = pygame.font.Font(None, 36)
    text = font.render("RESET", True, BLACK)
    reset_button_rect = pygame.Rect(WIDTH - 100, HEIGHT - 50, 90, 40)
    pygame.draw.rect(screen, COLORS[3], reset_button_rect)
    screen.blit(text, (WIDTH - 95, HEIGHT - 45))
    return reset_button_rect

def dbscan(points):
    clusters = [-1] * len(points)
    cluster_id = 0
    nn = NearestNeighbors(radius=EPSILON).fit(points)
    distances, indices = nn.radius_neighbors(points)

    for i, neighbors in enumerate(indices):
        if clusters[i] != -1 or len(neighbors) < MIN_POINTS:
            continue
        clusters[i] = cluster_id
        flags[i] = COLORS[cluster_id % len(COLORS)]
        draw_points()
        pygame.display.flip()
        pygame.time.delay(50)

        queue = list(neighbors)
        while queue:
            j = queue.pop(0)
            if clusters[j] == -1:
                clusters[j] = cluster_id
                if len(indices[j]) >= MIN_POINTS:
                    queue.extend(indices[j])
                flags[j] = COLORS[cluster_id % len(COLORS)]
                draw_points()
                pygame.display.flip()
                pygame.time.delay(50)
        cluster_id += 1

    for idx, cl in enumerate(clusters):
        if cl == -1:
            flags[idx] = GRAY
            draw_points()
            pygame.display.flip()
            pygame.time.delay(5)

running = True
reset_button_rect = draw_reset_button()

while running:
    draw_points()
    reset_button_rect = draw_reset_button()
    pygame.display.flip()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if reset_button_rect.collidepoint(event.pos):
                points.clear()
                flags.clear()
                draw_points()
                reset_button_rect = draw_reset_button()
                pygame.display.flip()
            else:
                pos = pygame.mouse.get_pos()
                points.append(pos)
                flags.append(BLACK)
                draw_points()
                reset_button_rect = draw_reset_button()
                pygame.display.flip()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                if points:
                    flags = [BLACK] * len(points)
                    dbscan(np.array(points))

pygame.quit()



