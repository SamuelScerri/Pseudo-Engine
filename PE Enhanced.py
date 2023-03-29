import pygame
import numpy
import numba

class Player:
	def __init__(self, position, fov, view_distance):
		self.position = position
		self.angle = 0
		self.fov = fov
		self.view_distance = view_distance
		self.offset_y = 32

		
@numba.jit(nopython=True, nogil=True, cache=False, fastmath=True)
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

@numba.jit(nopython=True, nogil=True, cache=False, fastmath=True)
def clamp(value, minimum, maximum):
	return max(minimum, min(value, maximum))


@numba.jit(nopython=True, nogil=True, cache=False, fastmath=True)
def get_closest_wall(position, translated_angle, angle, translated_point, level):
	closest_wall = level[1]
	intersection_position = (0, 0)
	closest_distance = 0

	initial_request = True

	all_walls_intersected = []

	for wall in level:
		checked_intersection = check_intersection(translated_point, wall)

		if checked_intersection != (0, 0):
			closest_distance = numpy.sqrt(
				numpy.power(position[0] - checked_intersection[0], 2) +
				numpy.power(position[1] - checked_intersection[1], 2))

			all_walls_intersected.append((closest_distance, checked_intersection, wall))

	return sorted(all_walls_intersected, key=lambda tup: tup[0])


@numba.jit(nopython=True, nogil=True, cache=False, fastmath=True)
def normalize(direction):
	magnitude = numpy.sqrt(direction[0] * direction[0] + direction[1] * direction[1])
	if magnitude > 0:
		return direction[0] / magnitude, direction[1] / magnitude

	return (0, 0)


@numba.jit(nopython=True, nogil=True, cache=False, fastmath=True)
def collision_check(old_position, new_position, velocity, level):
	for wall in level:
		move_difference_segment = ((old_position[0], old_position[1]), (new_position[0], new_position[1]))
		checked_intersection = check_intersection(move_difference_segment, wall)

		if checked_intersection != (0, 0):
			return old_position

	return new_position


@numba.jit(nopython=True, nogil=True, cache=False, fastmath=True)
def lerp(a, b, t):
	return a + (b - a) * t


@numba.jit(nopython=True, nogil=True, cache=False, fastmath=True)
def mix(rgb_1, rgb_2):
	mixed_r = int(rgb_1[0] * rgb_2[0]) << 16
	mixed_g = int(rgb_1[1] * rgb_2[1]) << 8
	mixed_b = int(rgb_1[2] * rgb_2[2])

	return mixed_r + mixed_g + mixed_b


@numba.jit(nopython=True, nogil=True, cache=False, fastmath=True)
def convert_int_rgb(code):
	converted_r = (code >> 16) & 0xff
	converted_g = (code >> 8) & 0xff
	converted_b = code & 0xff

	return converted_r, converted_g, converted_b


@numba.jit(nopython=True, nogil=True, cache=False, fastmath=True)
def scan_line(position, angle, fov, view_distance, offset_y, level, floor, ceiling, buffer):
	#Get The Interval Angle To Loop Through Every X-Coordinate Correctly
	interval_angle = fov / buffer.shape[0]
	half_height = int(buffer.shape[1] / 2)

	for x in range(buffer.shape[0]):
		translated_angle = numpy.radians((angle - fov / 2) + interval_angle * x)

		translated_point = (
			(position[0], position[1]),
			(position[0] + view_distance * numpy.cos(translated_angle), position[1] + view_distance * numpy.sin(translated_angle)))

		intersected_walls = get_closest_wall(position, translated_angle, angle, translated_point, level)

		previous_top_wall_height = -0.1

		for wall in intersected_walls:
			distance = wall[0] * numpy.cos(translated_angle - numpy.radians(angle))
			intersection_position = wall[1]
			wall_reference = wall[2]


			if distance != 0:
				wall_height = numpy.floor(half_height / distance) * (buffer.shape[0] / buffer.shape[1])

				texture_distance = numpy.sqrt(
					numpy.power(wall_reference[0][0] - intersection_position[0], 2) +
					numpy.power(wall_reference[0][1] - intersection_position[1], 2)) % 1

				darkness = clamp(lerp(0, .5, 1 / distance), 0, 1)

				bottom_height = half_height + wall_height
				top_height = (half_height + wall_height) - 2 * wall_height * wall_reference[3]

				if previous_top_wall_height != -0.1:
					bottom_height = clamp(bottom_height, 0, previous_top_wall_height)
					top_height = clamp(top_height, 0, previous_top_wall_height)

				#Wall Casting
				for y in range(clamp(top_height, 0, buffer.shape[1]), clamp(bottom_height, 0, buffer.shape[1])):
					texture_rgb = convert_int_rgb(wall_reference[2][int(texture_distance * 64), int((y - (half_height - wall_height)) / wall_height * 32)])
					buffer[x, y] = mix(texture_rgb, (darkness, darkness, darkness))

				floor_range = buffer.shape[1]

				if previous_top_wall_height != -0.1:
					floor_range = clamp(floor_range, 0, previous_top_wall_height)

				#Floor Casting
				for y in range(clamp(bottom_height, 0, buffer.shape[1]), floor_range):
					floor_distance = buffer.shape[1] / (2 * y - buffer.shape[1])
					floor_distance /= numpy.cos(translated_angle - numpy.radians(angle))

					darkness = clamp(lerp(0, .5, 1 / floor_distance), 0, 1)

					translated_floor_point = (
						position[0] + floor_distance * numpy.cos(translated_angle) * (buffer.shape[0] / buffer.shape[1]),
						position[1] + floor_distance * numpy.sin(translated_angle) * (buffer.shape[0] / buffer.shape[1]))

					final_point = (
						int((translated_floor_point[0] * 64) % 64),
						int((translated_floor_point[1] * 64) % 64))

					floor_rgb = convert_int_rgb(floor[final_point[0], final_point[1]])
					ceiling_rgb = convert_int_rgb(ceiling[final_point[0], final_point[1]])

					buffer[x, y] = mix(floor_rgb, (darkness, darkness, darkness))
					#buffer[x, buffer.shape[1] - y] = mix(ceiling_rgb, (darkness, darkness, darkness))					

				#Ceiling Casting

				#ceiling_range = 0
				#for y in range(clamp(0, 0, buffer.shape[1]), top_height):
				#	floor_distance = buffer.shape[1] / (2 * y - buffer.shape[1])
				#	floor_distance /= numpy.cos(translated_angle - numpy.radians(angle))

				#	darkness = clamp(lerp(0, .5, 1 / floor_distance), 0, 1)

				#	translated_floor_point = (
				#		position[0] + floor_distance * numpy.cos(translated_angle) * (buffer.shape[0] / buffer.shape[1]),
				#		position[1] + floor_distance * numpy.sin(translated_angle) * (buffer.shape[0] / buffer.shape[1]))

				#	final_point = (
				#		int((translated_floor_point[0] * 64) % 64),
				#		int((translated_floor_point[1] * 64) % 64))

				#	floor_rgb = convert_int_rgb(floor[final_point[0], final_point[1]])
				#	ceiling_rgb = convert_int_rgb(ceiling[final_point[0], final_point[1]])

				#	buffer[x, y] = mix(floor_rgb, (darkness, darkness, darkness))


				previous_top_wall_height = top_height

		#closest_distance *= numpy.cos(translated_angle - numpy.radians(angle))

		#if closest_distance != 0:
		#	wall_height = numpy.floor(half_height / closest_distance) * (buffer.shape[0] / buffer.shape[1])
					
			#We Do % 1 To Repeat The Texture
		#	texture_distance = numpy.sqrt(
		#		numpy.power(closest_wall[0][0] - intersection_position[0], 2) +
		#		numpy.power(closest_wall[0][1] - intersection_position[1], 2)) % 1

		#	darkness = clamp(lerp(0, .5, 1 / closest_distance), 0, 1)

		#	top_height = (half_height + wall_height) - 2 * wall_height * closest_wall[3]

			#Wall Casting Code
		#	for y in range(clamp(top_height, 0, buffer.shape[1]), clamp(half_height + wall_height, 0, buffer.shape[1])):
		#		texture_rgb = convert_int_rgb(closest_wall[2][int(texture_distance * 64), int((y - (half_height - wall_height)) / wall_height * 32)])
		#		buffer[x, y] = mix(texture_rgb, (darkness, darkness, darkness))


			#Floor Casting Code
		#	for y in range(clamp(half_height + wall_height, 0, buffer.shape[1]), buffer.shape[1]):


pygame.init()

screen_surface = pygame.display.set_mode((128, 128), pygame.SCALED, vsync=False)
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
player = Player((68, 66), 60, 128)

#Create Frame Counter
clock = pygame.time.Clock()
font = pygame.font.SysFont("Monospace" , 16 , bold = False)


level = (
	((64, 64), (70, 64), basic_wall_1, 1.0),
	((70, 64), (70, 70), basic_wall_1, 1.0),
	((64, 64), (70, 70), basic_wall_1, 1.0),
	((69, 66), (70, 70), basic_wall_1, 0.4)
)

floor = basic_wall_2
ceiling = basic_wall_3

first_part_wall = None
second_part_wall = None

dt = 0

while running:
	keys = pygame.key.get_pressed()

	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False

		if event.type == pygame.MOUSEMOTION:
			player.angle += event.rel[0] * .2


		if event.type == pygame.KEYDOWN:
			down_keys = pygame.key.get_pressed()
			if down_keys[pygame.K_DOWN] or down_keys[pygame.K_UP]:
				player.fov += (down_keys[pygame.K_UP] - down_keys[pygame.K_DOWN])
				print("Changing FOV To:", player.fov)

		if event.type == pygame.MOUSEBUTTONDOWN:
			if pygame.mouse.get_pressed()[0]:
				if first_part_wall == None:
					first_part_wall = (int(player.position[0]), int(int(player.position[1])))
				else:
					second_part_wall = (int(player.position[0]), int(int(player.position[1])))
					new_wall = first_part_wall, second_part_wall, basic_wall_1
		
					level.append(new_wall)
		
					first_part_wall = None
					second_part_wall = None

	#Player Movement
	move_position = (
		player.position[0] + (numpy.cos(numpy.radians(player.angle)) * (keys[pygame.K_w] - keys[pygame.K_s]) * 8 * dt),
		player.position[1] + (numpy.sin(numpy.radians(player.angle)) * (keys[pygame.K_w] - keys[pygame.K_s]) * 8 * dt)
	)

	player.angle += (keys[pygame.K_d] - keys[pygame.K_a]) * 128 * dt

	#player.position = collision_check(player.position, move_position, 8 * dt, level)
	player.position = move_position
	scan_line(player.position, player.angle, player.fov, player.view_distance, player.offset_y, level, floor, ceiling, buffer)

	pygame.surfarray.blit_array(screen_surface, buffer)
	screen_surface.blit(font.render("FPS: " + str(int(clock.get_fps())), False, (255, 255, 255)), (0, 0))

	#This Is Unecessary In Closed Areas
	pygame.display.flip()
	buffer.fill(0)

	dt = clock.tick() / 1000