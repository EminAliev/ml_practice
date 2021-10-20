import numpy as np
import pygame

class DBScan:
    def algo(self, points, eps):
        labels = [0] * len(points)
        cluster_idx = 0
        minimum_points = 1
        for i in range(0, len(points)):
            if not (labels[i] == 0):
                continue
            near_points = self.near_by_points(points, i, eps)
            if len(near_points) < minimum_points:
                labels[i] = -1
            else:
                cluster_idx += 1
                labels[i] = cluster_idx
                i = 0
                while i < len(near_points):
                    point = near_points[i]
                    if labels[point] == -1:
                        labels[point] = cluster_idx

                    elif labels[point] == 0:
                        labels[point] = cluster_idx
                        point_near = self.near_by_points(points, point, eps)
                        if len(point_near) >= minimum_points:
                            near_points = near_points + point_near
                    i += 1

        return labels

    def near_by_points(self, points, idx, eps):
        near = []
        for point_idx in range(0, len(points)):
            if np.linalg.norm(points[idx] - points[point_idx]) < eps:
                near.append(point_idx)
        return near

    def colors(self, list_col):
        if list_col == 1:
            return (255, 0, 0)
        if list_col == 2:
            return (0, 255, 0)
        if list_col == 3:
            return (0, 0, 255)
        if list_col == 4:
            return (0, 0, 0)
        return (125, 125, 125)

    def draw(self, list_col, clusters):
        for point, cluster in zip(list_col, clusters):
            color = self.colors(cluster)
            radius = 10
            pygame.draw.circle(screen, color, point, radius)



if __name__ == '__main__':
    pygame.init()
    dbcan_class = DBScan()
    screen = pygame.display.set_mode((1280, 720))
    clock = pygame.time.Clock()
    points = []
    screen.fill((255, 255, 255))

    eps = 40

    done = False
    while not done:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.MOUSEBUTTONDOWN:
                points.append(pygame.mouse.get_pos())
                screen.fill((255, 255, 255))
                prediction = dbcan_class.algo(np.array(points), eps)
                dbcan_class.draw(points, prediction)

        pygame.display.update()
