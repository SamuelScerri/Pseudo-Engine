import pygame
import numpy
import numba

#--------------------------------
#Enumerations
#--------------------------------


PLAYER_POSITION, PLAYER_ANGLE, PLAYER_VISION, PLAYER_DISTANCE, PLAYER_OFFSET = 0, 1, 2, 3, 4

WALL_POINT_A, WALL_POINT_B, WALL_FLOOR_HEIGHT, WALL_CEILING_HEIGHT, WALL_SEGMENT, WALL_TEXTURE = 0, 1, 2, 3, 4, 5


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
				numpy.power(posiiton[1] - checked_intersection[1], 2))

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
		pass


#--------------------------------
#Main Loop
#--------------------------------


pygame.init()

screen_surface = pygame.display.set_mode((256, 256), pygame.SCALED, vsync=True)
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
	)

	scan_line(player, level, buffer)

	pygame.surfarray.blit_array(screen_surface, buffer)
	screen_surface.blit(font.render("FPS: " + str(int(clock.get_fps())), False, (255, 255, 255)), (0, 0))

	#This Is Unecessary In Closed Areas
	pygame.display.flip()
	buffer.fill(0)

	dt = clock.tick() / 1000