import pygame
import numpy
import numba

#--------------------------------
#Enumerations
#--------------------------------


PLAYER_POSITION, PLAYER_ANGLE, PLAYER_VISION, PLAYER_DISTANCE, PLAYER_OFFSET = 0, 1, 2, 3, 4

WALL_POINT_A, WALL_POINT_B, WALL_FLOOR_HEIGHT, WALL_CEILING_HEIGHT, WALL_SEGMENT, WALL_TEXTURE = 0, 1, 2, 3, 4, 5

INTERSECTED_DISTANCE, INTERSECTED_POSITION, INTERSECTED_WALL = 0, 1, 2


#--------------------------------
#Functions
#--------------------------------


#Checks The Intersection Between Two Line Segments
@numba.jit(nopython=True, nogil=True, cache=True, fastmath=True)
def check_intersection(wall_1, wall_2):
	x1, y1 = wall_1[0]
	x2, y2 = wall_1[1]
	x3, y3 = wall_2[0]
	x4, y4 = wall_2[1]

	denominator = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)

	if denominator == 0:
		return (0, 0)
	
	ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denominator
	if ua < 0 or ua > 1:
		return (0, 0)

	ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denominator
	if ub < 0 or ub > 1:
		return (0, 0)

	return (x1 + ua * (x2 - x1), y1 + ua * (y2 - y1))


#This Is Very Useful For Ceiling Casts & Making Functions More Generalized
@numba.jit(nopython=True, nogil=True, cache=True, fastmath=True)
def clamp_in_order(value, minimum, maximum):
	if minimum < maximum:
		return max(minimum, min(value, maximum))
	else:
		return max(maximum, min(value, minimum))


#Gets The Closest Walls That Have Been Intersected With
@numba.jit(nopython=True, nogil=True, cache=True, fastmath=True)
def get_closest_wall(position, translated_point, level):
	all_walls_intersected = []

	#Add All The Walls Found In The Level
	for wall in level:
		checked_intersection = check_intersection(translated_point, wall)

		if checked_intersection != (0, 0):
			distance = numpy.sqrt(
				numpy.power(position[0] - checked_intersection[0], 2) +
				numpy.power(position[1] - checked_intersection[1], 2))

			all_walls_intersected.append((distance, checked_intersection, wall))

	#Here The Walls Are Ordered By The Distance
	for i in range(len(all_walls_intersected)):
		for j in range(0, len(all_walls_intersected) - i - 1):
			if all_walls_intersected[j][0] > all_walls_intersected[j + 1][0]:
				all_walls_intersected[j], all_walls_intersected[j + 1] = all_walls_intersected[j + 1], all_walls_intersected[j]

	return all_walls_intersected


#Scan The Entire Screen From Left To Right & Render The Walls & Flors
@numba.jit(nopython=True, nogil=True, cache=True, fastmath=True)
def scan_line(player, level, buffer):
	#Obtain The Interval Angle To Rotate Correctly
	interval_angle = player[2] / buffer.shape[0]
	half_height = buffer.shape[1] / 2

	for x in range(buffer.shape[0]):
		#Translate All Points According To Angle
		translated_angle = numpy.radians((player[PLAYER_ANGLE] - player[PLAYER_VISION] / 2) + interval_angle * x)

		translated_point = (
			(player[PLAYER_POSITION][0], player[PLAYER_POSITION][1]),
			(player[PLAYER_POSITION][0] + player[PLAYER_DISTANCE] * numpy.cos(translated_angle), player[PLAYER_POSITION][1] + player[PLAYER_DISTANCE] * numpy.sin(translated_angle)))

		#We Get All The Intersected Walls From Closest To Furthest
		intersected_walls = get_closest_wall(player[PLAYER_POSITION], translated_point, level)

		#Previous Wall Information For Comparision
		previous_floor_height = (0, 0)
		previous_ceiling_height = (0, 0)

		for wall in range(len(intersected_walls)):
			if intersected_walls[wall][INTERSECTED_DISTANCE] != 0:
				wall_reference = intersected_walls[wall]

				#Fix The Distance To Remove The Fish-Eye Distortion
				fixed_distance = wall_reference[INTERSECTED_DISTANCE] * numpy.cos(translated_angle - numpy.radians(player[PLAYER_ANGLE]))
				segment = wall_reference[INTERSECTED_WALL][WALL_SEGMENT]

				wall_height = (half_height / fixed_distance)

				#Get The Repeated Texture Coordinate
				texture_distance = numpy.sqrt(
					numpy.power(wall_reference[INTERSECTED_WALL][WALL_POINT_A][0] - wall_reference[INTERSECTED_POSITION][0], 2) +
					numpy.power(wall_reference[INTERSECTED_WALL][WALL_POINT_A][1] - wall_reference[INTERSECTED_POSITION][1], 2)) % 1

				floor_height = (half_height + wall_height, (half_height + wall_height) - 2 * wall_height * (wall_reference[INTERSECTED_WALL][WALL_FLOOR_HEIGHT]))
				ceiling_height = (half_height - wall_height, (half_height - wall_height) + 2 * wall_height * (wall_reference[INTERSECTED_WALL][WALL_CEILING_HEIGHT]))

				

				#These Are Stored For Later Comparisions
				previous_floor_height = floor_height
				previous_ceiling_height = ceiling_height


#--------------------------------
#Main Loop
#--------------------------------


pygame.init()

screen_surface = pygame.display.set_mode((256, 256), pygame.SCALED, vsync=False)
pygame.mouse.set_visible(False)
pygame.event.set_grab(True)

running = True

#Create Buffer
buffer = numpy.zeros((screen_surface.get_width(), screen_surface.get_height()), dtype=numpy.int32)

#Create Textures
basic_wall_1 = pygame.surfarray.array2d(pygame.image.load("texture.png").convert())
basic_wall_2 = pygame.surfarray.array2d(pygame.image.load("texture2.png").convert())
basic_wall_3 = pygame.surfarray.array2d(pygame.image.load("texture3.png").convert())

#Create Player
player = ((63, 63), 0, 60, 128)

#Create Frame Counter
clock = pygame.time.Clock()
font = pygame.font.SysFont("Monospace" , 16 , bold = False)

#Level Data
level = (
	((64, 64), (70, 64), 0.8, 0.2, 1, basic_wall_2),
	((70, 64), (70, 70), 0.8, 0.2, 1, basic_wall_2),
	((64, 64), (70, 70), 0.8, 0.2, 1, basic_wall_2),
)

dt = 0

while running:
	keys = pygame.key.get_pressed()

	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False

	#Player Movement
	player = (
		(player[PLAYER_POSITION][0] + (numpy.cos(numpy.radians(player[PLAYER_ANGLE])) * (keys[pygame.K_w] - keys[pygame.K_s]) * 4 * dt),
		player[PLAYER_POSITION][1] + (numpy.sin(numpy.radians(player[PLAYER_ANGLE])) * (keys[pygame.K_w] - keys[pygame.K_s]) * 4 * dt)),

		player[PLAYER_ANGLE] + ((keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * 128 * dt),
		player[PLAYER_VISION],
		player[PLAYER_DISTANCE],
	)

	scan_line(player, level, buffer)

	pygame.surfarray.blit_array(screen_surface, buffer)
	screen_surface.blit(font.render("FPS: " + str(int(clock.get_fps())), False, (255, 255, 255)), (0, 0))

	#This Is Unecessary In Closed Areas
	pygame.display.flip()
	buffer.fill(0)

	dt = clock.tick() / 1000