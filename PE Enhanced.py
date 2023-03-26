import pygame
import numpy
import numba

class Player:
	def __init__(self, position, fov, view_distance):
		self.position = position
		self.angle = 0
		self.fov = fov
		self.view_distance = view_distance


class Wall:
	def __init__(self, start_position, end_position, surface):
		self.start_position = start_position
		self.end_position = end_position
		self.surface = surface

@numba.jit(nopython=True, nogil=True)
def check_intersection(wall_1, wall_2):
	x1, y1 = wall_1[0]
	x2, y2 = wall_1[1]
	x3, y3 = wall_2[0]
	x4, y4 = wall_2[1]

	denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)

	if denom == 0:
		return (0, 0)
	ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
	if ua < 0 or ua > 1: # out of range
		return (0, 0)
	ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom
	if ub < 0 or ub > 1: # out of range
		return (0, 0)
	x = x1 + ua * (x2-x1)
	y = y1 + ua * (y2-y1)
	return (x,y)	

@numba.jit(nopython=True, nogil=True)
def clamp(value, minimum, maximum):
	return max(minimum, min(value, maximum))

@numba.jit(nopython=True, nogil=True)
def scan_line(position, angle, fov, view_distance, level, floor, buffer):
	#Get The Interval Angle To Loop Through Every X-Coordinate Correctly
	interval_angle = fov / buffer.shape[0]
	half_height = int(buffer.shape[1] / 2)

	for x in range(buffer.shape[0]):
		translated_angle = numpy.radians((angle - fov / 2) + interval_angle * x)

		translated_point = (
			(position[0], position[1]),
			(position[0] + view_distance * numpy.cos(translated_angle), position[1] + view_distance * numpy.sin(translated_angle)))

		for wall in level:
			intersection_position = check_intersection(translated_point, wall)

			if intersection_position != (0, 0):
				#print(intersection_position)s
				distance = numpy.sqrt(
					numpy.power(position[0] - intersection_position[0], 2) +
					numpy.power(position[1] - intersection_position[1], 2)) * numpy.cos(translated_angle - numpy.radians(angle))

				if distance != 0:
					wall_height = numpy.floor(half_height / distance) * (buffer.shape[0] / buffer.shape[1])
					
					#We Do % 1 To Repeat The Texture
					texture_distance = numpy.sqrt(
						numpy.power(wall[0][0] - intersection_position[0], 2) +
						numpy.power(wall[0][1] - intersection_position[1], 2)) % 1

					for y in range(clamp(half_height - wall_height, 0, buffer.shape[1]), clamp(half_height + wall_height, 0, buffer.shape[1])):
						buffer[x, y] = wall[2][int(texture_distance * 64), int((y - (half_height - wall_height)) / wall_height * 32)]

					for y in range(clamp(half_height + wall_height, 0, buffer.shape[1]), buffer.shape[1]):
						floor_distance = buffer.shape[1] / (2 * y - buffer.shape[1])
						floor_distance /= numpy.cos(translated_angle - numpy.radians(angle))

						translated_floor_point = (
							position[0] + floor_distance * numpy.cos(translated_angle) * (buffer.shape[0] / buffer.shape[1]),
							position[1] + floor_distance * numpy.sin(translated_angle) * (buffer.shape[0] / buffer.shape[1]))

						final_point = (
							int((translated_floor_point[0] * 32) % 64),
							int((translated_floor_point[1] * 32) % 64))

						buffer[x, y] = level[1][2][final_point[0], final_point[1]]
						buffer[x, buffer.shape[1] - y] = level[1][2][final_point[0], final_point[1]]

pygame.init()

screen_surface = pygame.display.set_mode((512, 512), pygame.SCALED | pygame.FULLSCREEN, vsync=True)
pygame.mouse.set_visible(False)
pygame.event.set_grab(True)

running = True

#Create Buffer
buffer = numpy.zeros((screen_surface.get_width(), screen_surface.get_height()), dtype=numpy.int32)

#Create Textures
basic_wall_1 = pygame.surfarray.array2d(pygame.image.load("texture.png").convert())
basic_wall_2 = pygame.surfarray.array2d(pygame.image.load("texture2.png").convert())

#Create Player
player = Player((68, 66), 60, 128)

#Create Frame Counter
clock = pygame.time.Clock()
font = pygame.font.SysFont("Monospace" , 24 , bold = False)

level = (
	((64, 64), (70, 64), basic_wall_1),
	((70, 64), (70, 70), basic_wall_2),
	((64, 64), (70, 70), basic_wall_1)
)

floor = (64, 64, 32, 32)

while running:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False

		if event.type == pygame.MOUSEMOTION:
			player.angle += event.rel[0] * .05

	keys = pygame.key.get_pressed()

	#Player Movement
	player.position = (
		player.position[0] + (numpy.cos(numpy.radians(player.angle)) * (keys[pygame.K_w] - keys[pygame.K_s]) * .1),
		player.position[1] + (numpy.sin(numpy.radians(player.angle)) * (keys[pygame.K_w] - keys[pygame.K_s]) * .1)
	)

	scan_line(player.position, player.angle, player.fov, player.view_distance, level, floor, buffer)
	pygame.surfarray.blit_array(screen_surface, buffer)
	screen_surface.blit(font.render("FPS: " + str(clock.get_fps()), False, (255, 255, 255)), (0, 0))

	#This Is Unecessary In Closed Areas
	#buffer.fill(0)
	pygame.display.flip()

	clock.tick()