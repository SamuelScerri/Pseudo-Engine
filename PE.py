import pygame
import math
import numba

SIZE = (256, 256)

screen_surface = pygame.display.set_mode(SIZE, pygame.SCALED | pygame.FULLSCREEN, vsync=True)
image = pygame.image.load("texture.png").convert()
image_2 = pygame.image.load("texture2.png").convert()
image_3 = pygame.image.load("texture3.png").convert()

def check_intersection(wall_1, wall_2):
	x1, y1 = wall_1.start_position
	x2, y2 = wall_1.end_position
	x3, y3 = wall_2.start_position
	x4, y4 = wall_2.end_position

	denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)

	if denom == 0:
		return None
	ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
	if ua < 0 or ua > 1: # out of range
		return None
	ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom
	if ub < 0 or ub > 1: # out of range
		return None
	x = x1 + ua * (x2-x1)
	y = y1 + ua * (y2-y1)
	return (x,y)

def lerp(a, b, t):
	return a + (b - a) * t

class Player:
	def __init__(self, position, fov, distance):
		self.position = position
		self.angle = 0
		self.fov = fov
		self.distance = distance

		#Here We Get The Interval To Ensure That Every X Coordinate In The Engine Will Be Accounted For
		self.interval_angle = fov / SIZE[0]

	def update(self):
		keys = pygame.key.get_pressed()

		#self.angle += (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT])
		radian = math.radians(self.angle)

		self.position = (
			self.position[0] + (math.cos(radian) * (keys[pygame.K_w] - keys[pygame.K_s]) * .1),
			self.position[1] + (math.sin(radian) * (keys[pygame.K_w] - keys[pygame.K_s]) * .1)
		)

		#self.position = (
			#self.position[0] + (math.cos(radian + math.radians(90)) * (keys[pygame.K_d] - keys[pygame.K_a]) * .01),
			#self.position[1] + (math.sin(radian + math.radians(90)) * (keys[pygame.K_d] - keys[pygame.K_a]) * .01)
		#)

	def render(self, surface, walls):


		for x in range(SIZE[0]):
			radian = math.radians((self.angle - self.fov / 2.0) + self.interval_angle * x)

			translated_point = (
				self.position[0] + self.distance * math.cos(radian),
				self.position[1] + self.distance * math.sin(radian)
			)

			
			wall_detected = False

			half_height = SIZE[1] / 2

			def get_closest_wall(all_walls):
				closest_wall = None
				smallest_distance = 0
				texture_distance = 0

				for wall in all_walls:
					position = check_intersection(Wall(self.position, translated_point, None), wall)

					if position != None:
						if closest_wall is None:
							closest_wall = wall
							smallest_distance = math.sqrt(
								math.pow(self.position[0] - position[0], 2) +
								math.pow(self.position[1] - position[1], 2))

							texture_distance = math.sqrt(
								math.pow(wall.start_position[0] - position[0], 2) +
								math.pow(wall.start_position[1] - position[1], 2))
							#print(texture_distance)
						else:
							distance = math.sqrt(
								math.pow(self.position[0] - position[0], 2) +
								math.pow(self.position[1] - position[1], 2))

							if distance < smallest_distance:
								smallest_distance = distance
								closest_wall = wall

								texture_distance = math.sqrt(
									math.pow(wall.start_position[0] - position[0], 2) +
									math.pow(wall.start_position[1] - position[1], 2))

								

				return closest_wall, smallest_distance, texture_distance

			closest_wall, smallest_distance, texture_distance = get_closest_wall(walls)
			#print(texture_distance)

			smallest_distance *= math.cos(radian - math.radians(self.angle))

			if smallest_distance != 0:
				wall_height = math.floor(half_height / smallest_distance)
				color_darkness = lerp(0, 1, 1 / smallest_distance) * 255 * 2
				if color_darkness > 255:
					color_darkness = 255

				new_surface = pygame.transform.scale(closest_wall.surface, (closest_wall.surface.get_width(), wall_height * 2))

				if texture_distance > 1:
					texture_distance = texture_distance % 1
				surface.blit(new_surface, (x, half_height - wall_height, 1, 0), (texture_distance * 64, 0, 1, wall_height * 2))

			#pygame.draw.line(surface, (255, 255, 255), self.position, translated_point)
pygame.init()



running = True

clock = pygame.time.Clock()
font = pygame.font.SysFont("Monospace" , 24 , bold = False)

player = Player((68, 66), 60, 256)

class Wall:
	def __init__(self, start_position, end_position, surface):
		self.start_position = start_position
		self.end_position = end_position
		self.surface = surface

walls = [
	Wall((64, 64), (70, 64), image),
	Wall((70, 64), (70, 70), image_2),
	Wall((64, 64), (70, 70), image_3)
]

pygame.mouse.set_visible(False)
pygame.event.set_grab(True)

while running:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False

		#if event.type == pygame.MOUSEBUTTONDOWN:
			#if pygame.mouse.get_pressed()[0]:
				#last_wall_position = walls[len(walls) - 1].end_position

				#walls.append(Wall(last_wall_position, pygame.mouse.get_pos()))

		if event.type == pygame.MOUSEMOTION:
			dx = event.rel[0]
			player.angle += dx * .1


	for wall in walls:
		pygame.draw.line(screen_surface, (255, 255, 255), wall.start_position, wall.end_position)

	player.update()
	player.render(screen_surface, walls)

	screen_surface.blit(font.render("FPS: " + str(clock.get_fps()), False, (255, 255, 255)), (0, 0))

	pygame.display.flip()
	screen_surface.fill((0, 0, 0))
	clock.tick()

pygame.quit()