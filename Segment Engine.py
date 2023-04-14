import pygame
import numpy
import numba

import pymunk
import pymunk.pygame_util

#--------------------------------
#Enumerations
#--------------------------------


PLAYER_POSITION, PLAYER_ANGLE, PLAYER_VISION, PLAYER_DISTANCE, PLAYER_OFFSET = 0, 1, 2, 3, 4

WALL_POINT_A, WALL_POINT_B, WALL_FLOOR_HEIGHT, WALL_CEILING_HEIGHT, WALL_SEGMENT, WALL_TEXTURE, WALL_FLOOR_TEXTURE = 0, 1, 2, 3, 4, 5, 6

INTERSECTED_DISTANCE, INTERSECTED_POSITION, INTERSECTED_WALL = 0, 1, 2


#--------------------------------
#Functions
#--------------------------------


#Checks The Intersection Between Two Line Segments3
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


@numba.jit(nopython=True, nogil=True, cache=True, fastmath=True)
def lerp(a, b, t):
	return a + (b - a) * t


@numba.jit(nopython=True, nogil=True, cache=True, fastmath=True)
def normalize(direction):
	magnitude = numpy.sqrt(direction[0] * direction[0] + direction[1] * direction[1])
	if magnitude > 0:
		return (direction[0] / magnitude, direction[1] / magnitude)

	return (0, 0)


@numba.jit(nopython=True, nogil=True, cache=True, fastmath=True)
def mix(rgb_1, rgb_2):
	mixed_r = int(rgb_1[0] * rgb_2[0]) << 16
	mixed_g = int(rgb_1[1] * rgb_2[1]) << 8
	mixed_b = int(rgb_1[2] * rgb_2[2])

	return mixed_r + mixed_g + mixed_b


@numba.jit(nopython=True, nogil=True, cache=True, fastmath=True)
def convert_int_rgb(code):
	converted_r = (code >> 16) & 0xff
	converted_g = (code >> 8) & 0xff
	converted_b = code & 0xff

	return converted_r, converted_g, converted_b


#This Is Very Useful For Ceiling Casts & Making Functions More Generalized
@numba.jit(nopython=True, nogil=True, cache=True, fastmath=True)
def clamp_in_order(value, minimum, maximum):
	return max(minimum, min(value, maximum))


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

	segmented_order = []

	#This Will Order By Segments
	#We Are Going To Loop Backwards Here
	while len(all_walls_intersected) != 0:
		#print(len(all_walls_intersected))
		#current_wall_checked = all_walls_intersected.pop(len(all_walls_intersected) - 1)

		walls_obtained = [len(all_walls_intersected) - 1]
		offset = 0

		segmented_order.insert(0, all_walls_intersected[len(all_walls_intersected) - 1])

		#Find Any Other Wall With The Same Segment
		for i in range(len(all_walls_intersected) - 2, -1, -1):
			if all_walls_intersected[i][INTERSECTED_WALL][WALL_SEGMENT] == all_walls_intersected[len(all_walls_intersected) - 1][INTERSECTED_WALL][WALL_SEGMENT]:
				segmented_order.insert(0, all_walls_intersected[i])
				walls_obtained.append(i)

		for index in sorted(walls_obtained, reverse=True):
			del all_walls_intersected[index]

	return segmented_order


#Scan The Entire Screen From Left To Right & Render The Walls & Floors
@numba.jit(nopython=True, nogil=True, cache=True, fastmath=True)
def scan_line(player, level, buffer, tree_thing):
	#Obtain The Interval Angle To Rotate Correctly
	interval_angle = player[2] / buffer.shape[0]
	half_height = buffer.shape[1] / 2

	offset = 0

	for x in range(buffer.shape[0]):
		final_position = (0, 0, 0, 0)
		found = False

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

				#We Get All The Wall Heights To Be Drawn Later
				floor_height = (
					clamp_in_order((half_height + wall_height) - 2 * wall_height * (wall_reference[INTERSECTED_WALL][WALL_FLOOR_HEIGHT] + player[PLAYER_OFFSET]), 0, buffer.shape[1]),
					clamp_in_order((half_height + wall_height) - 2 * wall_height * (player[PLAYER_OFFSET]), 0, buffer.shape[1]))
					
				
				ceiling_height = (
					clamp_in_order((half_height - wall_height) + 2 * wall_height * (-player[PLAYER_OFFSET]), 0, buffer.shape[1]), 
					clamp_in_order((half_height - wall_height) + 2 * wall_height * (wall_reference[INTERSECTED_WALL][WALL_CEILING_HEIGHT] - player[PLAYER_OFFSET]), 0, buffer.shape[1]))
				
				cull_wall = False

				#If There Was A Previous Wall Already Rendered, We Clamp The Values To Avoid Overdraw
				if wall > 0:
					floor_height = (
						clamp_in_order(floor_height[0], previous_ceiling_height[1], previous_floor_height[0]),
						clamp_in_order(floor_height[1], previous_ceiling_height[1], previous_floor_height[0]))

					ceiling_height = (
						clamp_in_order(ceiling_height[0], previous_ceiling_height[1], previous_floor_height[0]),
						clamp_in_order(ceiling_height[1], previous_ceiling_height[1], previous_floor_height[0]))

					#If The Previous Wall Segment Was The Same, We Are Going To Draw The Floor Instead
					if wall_reference[INTERSECTED_WALL][WALL_SEGMENT] == intersected_walls[wall - 1][INTERSECTED_WALL][WALL_SEGMENT]:
						cull_wall = True

				floor_length = previous_floor_height[0]
				ceiling_length = previous_ceiling_height[1]

				#This Will Only Apply For The First Wall
				#Every Segment Requires 3 Or 4 Points, If There Is Not Another Point Detected Then It Is Guranteed That The Player Is Stepping On Top Of The Segment
				if wall == 0:
					if wall < len(intersected_walls) - 1:
						if wall_reference[INTERSECTED_WALL][WALL_SEGMENT] != intersected_walls[wall + 1][INTERSECTED_WALL][WALL_SEGMENT]:
							cull_wall = True
							floor_length = buffer.shape[1]
							ceiling_length = 0

							offset = wall_reference[INTERSECTED_WALL][WALL_FLOOR_HEIGHT]

					else:
						cull_wall = True
						floor_length = buffer.shape[1]
						ceiling_length = 0

						offset = wall_reference[INTERSECTED_WALL][WALL_FLOOR_HEIGHT]

				#Here We Will Draw The Walls
				if cull_wall == False:
					for y in range(floor_height[0], floor_height[1]):
						darkness = clamp_in_order(lerp(0, 1, 1 / fixed_distance), 0, 1)
						color_value = convert_int_rgb(wall_reference[INTERSECTED_WALL][WALL_TEXTURE][int(texture_distance * 64), int((y - ((half_height + wall_height) - 2 * wall_height * player[PLAYER_OFFSET])) / wall_height * 32)])

						buffer[x, y] = mix(color_value, (darkness, darkness, darkness))

					for y in range(ceiling_height[0], ceiling_height[1]):
						darkness = clamp_in_order(lerp(0, 1, 1 / fixed_distance), 0, 1)
						color_value = convert_int_rgb(wall_reference[INTERSECTED_WALL][WALL_TEXTURE][int(texture_distance * 64), int((y - ((half_height + wall_height) - 2 * wall_height * player[PLAYER_OFFSET])) / wall_height * 32)])

						buffer[x, y] = mix(color_value, (darkness, darkness, darkness))

				else:
					#We Render The Floor
					for y in range(floor_height[0], floor_length):
						interpolation = 2 * y - buffer.shape[1]

						if interpolation != 0:
							floor_distance = (buffer.shape[1] / interpolation) / numpy.cos(translated_angle - numpy.radians(player[PLAYER_ANGLE]))

							translated_floor_point = (
								player[PLAYER_POSITION][0] + floor_distance * numpy.cos(translated_angle) * (1 - (wall_reference[INTERSECTED_WALL][WALL_FLOOR_HEIGHT] + player[PLAYER_OFFSET]) * 2),
								player[PLAYER_POSITION][1] + floor_distance * numpy.sin(translated_angle) * (1 - (wall_reference[INTERSECTED_WALL][WALL_FLOOR_HEIGHT] + player[PLAYER_OFFSET]) * 2))

							darkness = clamp_in_order(lerp(0, 1, 1 / floor_distance), 0, 1)
							color_value = convert_int_rgb(wall_reference[INTERSECTED_WALL][WALL_FLOOR_TEXTURE][int((translated_floor_point[0] * 64) % 64), int((translated_floor_point[1] * 64) % 64)])

							buffer[x, y] = mix(color_value, (darkness, darkness, darkness))

					#And We Finally Draw The Ceiling
					for y in range(ceiling_length, ceiling_height[1]):
						interpolation = 2 * y - buffer.shape[1]

						if interpolation != 0:
							floor_distance = (buffer.shape[1] / interpolation) / numpy.cos(translated_angle - numpy.radians(player[PLAYER_ANGLE]))

							translated_floor_point = (
								player[PLAYER_POSITION][0] + floor_distance * numpy.cos(translated_angle) * -(1 - (wall_reference[INTERSECTED_WALL][WALL_CEILING_HEIGHT] - player[PLAYER_OFFSET]) * 2),
								player[PLAYER_POSITION][1] + floor_distance * numpy.sin(translated_angle) * -(1 - (wall_reference[INTERSECTED_WALL][WALL_CEILING_HEIGHT] - player[PLAYER_OFFSET]) * 2))

							darkness = clamp_in_order(lerp(0, 1, 1 / -floor_distance), 0, 1)
							color_value = convert_int_rgb(wall_reference[INTERSECTED_WALL][WALL_FLOOR_TEXTURE][int((translated_floor_point[0] * 64) % 64), int((translated_floor_point[1] * 64) % 64)])

							buffer[x, y] = mix(color_value, (darkness, darkness, darkness))
				
				dx = 66 - player[PLAYER_POSITION][0]
				dy = 70 - player[PLAYER_POSITION][1]

				dist = numpy.sqrt(dx * dx + dy * dy)

				theta = numpy.degrees(numpy.arctan2(-dy, dx))
				fixed_rotation = player[PLAYER_ANGLE] % 360

				y = (-fixed_rotation + (player[PLAYER_VISION] / 2) - theta)

				if y < -180:
					y += 360
			
				x_pos = y * (buffer.shape[0] / player[PLAYER_VISION])
				sprite_height = (half_height / dist)
				darkness = clamp_in_order(lerp(0, 1, 1 / dist), 0, 1)

				if x > x_pos - sprite_height and x < x_pos + sprite_height:
					if dist < intersected_walls[wall][INTERSECTED_DISTANCE]:
						if wall == 0:
							if not found:
								final_position = (clamp_in_order(half_height - sprite_height + 1, 0, buffer.shape[1]), clamp_in_order(half_height + sprite_height, 0, buffer.shape[1]), sprite_height, x_pos)
								found = True

						elif dist > intersected_walls[wall - 1][INTERSECTED_DISTANCE]:
							if not found:
								final_position = (clamp_in_order(half_height - sprite_height + 1, previous_ceiling_height[1], previous_floor_height[0]), clamp_in_order(half_height + sprite_height, previous_ceiling_height[1], previous_floor_height[0]), sprite_height, x_pos)
								found = True

							#This Means That The Object Is Directly In Front Of The Player
							#for y_loop in range(clamp_in_order(half_height - sprite_height + 1, 0, buffer.shape[1]), clamp_in_order(half_height + sprite_height, 0, buffer.shape[1])):
							#	if tree_thing[int((x - (x_pos + sprite_height)) / sprite_height * 32), int((y_loop - (half_height + sprite_height)) / sprite_height * 32)] != 9357180:
							#		color_value = convert_int_rgb(tree_thing[int((x - (x_pos + sprite_height)) / sprite_height * 32), int((y_loop - (half_height + sprite_height)) / sprite_height * 32)])
							#		buffer[x, y_loop] = mix(color_value, (darkness, darkness, darkness))			

					#	else:
					#		#If It Isn't In Front Of The Player, We Will Draw It Behind The Other Walls
					#		for y_loop in range(clamp_in_order(half_height - sprite_height + 1, previous_ceiling_height[1], previous_floor_height[0]), clamp_in_order(half_height + sprite_height, previous_ceiling_height[1], previous_floor_height[0])):
					#			if tree_thing[int((x - (x_pos + sprite_height)) / sprite_height * 32), int((y_loop - (half_height + sprite_height)) / sprite_height * 32)] != 9357180:
					#				color_value = convert_int_rgb(tree_thing[int((x - (x_pos + sprite_height)) / sprite_height * 32), int((y_loop - (half_height + sprite_height)) / sprite_height * 32)])
					#				buffer[x, y_loop] = mix(color_value, (darkness, darkness, darkness))



				#These Are Stored For Later Comparisions
				previous_floor_height = floor_height
				previous_ceiling_height = ceiling_height

		#We Will Draw The Sprites Here As Overlays
		for y_loop in range(final_position[0], final_position[1]):
			if tree_thing[int((x - (final_position[3] + final_position[2])) / final_position[2] * 32), int((y_loop - (half_height + final_position[2])) / final_position[2] * 32)] != 9357180:
				color_value = convert_int_rgb(tree_thing[int((x - (final_position[3] + final_position[2])) / final_position[2] * 32), int((y_loop - (half_height + final_position[2])) / final_position[2] * 32)])
				buffer[x, y_loop] = mix(color_value, (darkness, darkness, darkness))		

	return offset


#--------------------------------
#Main Loop
#--------------------------------


should_cap = False

#Physics
space = pymunk.Space()
space.gravity = (0, 0)

pygame.init()
pygame.mixer.init()

screen_surface = pygame.display.set_mode((256, 256), pygame.SCALED, vsync=False)
pygame.mouse.set_visible(False)
pygame.event.set_grab(True)

running = True

#Create Buffer
buffer = numpy.zeros((screen_surface.get_width(), screen_surface.get_height()), dtype=numpy.int32)
step_sound = pygame.mixer.Sound("Step.wav")
step_sound.set_volume(.2)

#Create Textures
#We Convert These To Arrays To Access The Texture Data Faster During The Rendering Process
basic_wall_1 = pygame.surfarray.array2d(pygame.image.load("texture.png").convert())
basic_wall_2 = pygame.surfarray.array2d(pygame.image.load("texture2.png").convert())
basic_wall_3 = pygame.surfarray.array2d(pygame.image.load("texture3.png").convert())
basic_wall_4 = pygame.surfarray.array2d(pygame.image.load("texture4.png").convert())
basic_wall_5 = pygame.surfarray.array2d(pygame.image.load("texture5.png").convert())

tree_thing = pygame.surfarray.array2d(pygame.image.load("tree.png").convert())
print(tree_thing)

#Create Player
player = ((66, 69), 0, 75, 128, 0)

#Create Frame Counter
#clock = pygame.time.Clock()
font = pygame.font.SysFont("Monospace" , 16 , bold = False)

offset = .5
offset2 = 1

#Level Data, This Is Temporary And Will Instead Load Through External Files In A Later Version
level = (
	((64.0, 71.0), (70.0, 71.0), 0.6, 0.0, 0, basic_wall_1, basic_wall_2),
	((70.0, 71.0), (70.0, 77.0), 0.6, 0.0, 0, basic_wall_1, basic_wall_2),
	((64.0, 71.0), (70.0, 77.0), 0.6, 0.0, 0, basic_wall_1, basic_wall_2),

	((64.0, 64.0), (70.0, 64.0), 0.2, 0.6, 1, basic_wall_2, basic_wall_3),
	((70.0, 64.0), (70.0, 70.0), 0.2, 0.6, 1, basic_wall_2, basic_wall_3),
	((64.0, 64.0), (70.0, 70.0), 0.2, 0.6, 1, basic_wall_2, basic_wall_3),

	((64.0, 64.0), (64.0, 71.0), 0.0, 0.2, 2, basic_wall_4, basic_wall_4),
	((64.0, 71.0), (70.0, 71.0), 0.0, 0.2, 2, basic_wall_4, basic_wall_4),
	((70.0, 71.0), (70.0, 70.0), 0.0, 0.2, 2, basic_wall_4, basic_wall_4),
	((70.0, 70.0), (64.0, 64.0), 0.0, 0.2, 2, basic_wall_4, basic_wall_4),

	((70.0, 71.0), (70.0, 70.0), 0.1, 0.0, 3, basic_wall_2, basic_wall_3),
	((70.0, 70.0), (70.5, 70.0), 0.1, 0.0, 3, basic_wall_2, basic_wall_3),
	((70.5, 70.0), (70.5, 71.0), 0.1, 0.0, 3, basic_wall_2, basic_wall_3),
	((70.5, 71.0), (70.0, 71.0), 0.1, 0.0, 3, basic_wall_2, basic_wall_3),

	((70.0 + offset, 71.0), (70.0 + offset, 70.0), 0.2, 0.0, 4, basic_wall_2, basic_wall_3),
	((70.0 + offset, 70.0), (70.5 + offset, 70.0), 0.2, 0.0, 4, basic_wall_2, basic_wall_3),
	((70.5 + offset, 70.0), (70.5 + offset, 71.0), 0.2, 0.0, 4, basic_wall_2, basic_wall_3),
	((70.5 + offset, 71.0), (70.0 + offset, 71.0), 0.2, 0.0, 4, basic_wall_2, basic_wall_3),

	((70.0 + offset2, 71.0), (70.0 + offset2, 70.0), 0.3, 0.0, 5, basic_wall_2, basic_wall_3),
	((70.0 + offset2, 70.0), (70.5 + offset2, 70.0), 0.3, 0.0, 5, basic_wall_2, basic_wall_3),
	((70.5 + offset2, 70.0), (70.5 + offset2, 71.0), 0.3, 0.0, 5, basic_wall_2, basic_wall_3),
	((70.5 + offset2, 71.0), (70.0 + offset2, 71.0), 0.3, 0.0, 5, basic_wall_2, basic_wall_3),

	((71.5, 71.0), (70.0, 71.0), 0.4, 0.0, 6, basic_wall_2, basic_wall_3),
	((70.0, 71.0), (70.0, 74.0), 0.4, 0.0, 6, basic_wall_2, basic_wall_3),
	((70.0, 74.0), (71.5, 72.0), 0.4, 0.0, 6, basic_wall_2, basic_wall_3),
	((71.5, 72.0), (71.5, 71.0), 0.4, 0.0, 6, basic_wall_2, basic_wall_3),
)

walls_physics_shape_information = []
walls_physics_body_information = []

offset = 0

#Adding The Walls To The Physics Engine
for wall in level:
	if wall[WALL_FLOOR_HEIGHT] > offset + .1:
		physics_geometry = pymunk.Body(body_type=pymunk.Body.STATIC)
		physics_segment = pymunk.Segment(physics_geometry, wall[WALL_POINT_A], wall[WALL_POINT_B], .2)

		space.add(physics_geometry, physics_segment)
		walls_physics_body_information.append(physics_geometry)
		walls_physics_shape_information.append(physics_segment)

draw_options = pymunk.pygame_util.DrawOptions(screen_surface)

mouse_velocity = 0

bobbing = 0
bobbing_strength = 0

#Create The Player Rigidbody,
#Friction Won't Matter Here As The Game's Logic Is Handled "Top-Down",
#Instead We Multiply The Velocity And Can Be Seen In The Physics Logic
player_body = pymunk.Body()
player_body.position = player[PLAYER_POSITION]
player_shape = pymunk.Circle(player_body, .2)
player_shape.mass = 1
player_shape.friction = 0

fps = 0

space.add(player_body, player_shape)
dt = 0
old_time = 0
time_between_physics = 0

interpolated_position = (0.0, 0.0)
interpolated_rotation = 0
rotation = player[PLAYER_ANGLE]

update_rate = 0

previous_time = 0
final_bobbing = 0
current_offset = 0
gravity_velocity = 0
should_jump = False

while running:
	keys = pygame.key.get_pressed()

	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False

		if event.type == pygame.MOUSEMOTION:
			mouse_velocity += event.rel[0] * .1

		if event.type == pygame.KEYDOWN:
			if pygame.key.get_pressed()[pygame.K_SPACE]:
				should_jump = True

	#print(player_body.position)

	#This Is So That Game Logic Will Not Be Tied To The Rendering Speed, We Can Also Now Do Interpolation
	while update_rate >= 1000 / 30:
		old_position = player_body.position
		old_rotation = rotation
		old_bobbing = final_bobbing

		current_offset = offset

		#if should_jump and current_offset == offset:
		#	gravity_velocity = .12
		#	should_jump = False

		#gravity_velocity -= .02
		#current_offset += gravity_velocity

		#if current_offset < offset:
			#current_offset = offset
			#gravity_velocity = 0


		#Direction Here Is Normalized For Diagonal Movement,
		#Without It Diagonal Movement Will Be Faster
		direction = normalize((
			(keys[pygame.K_w] - keys[pygame.K_s]),
			(keys[pygame.K_d] - keys[pygame.K_a])))

		#We Use Pymunk Here To Move The Player, This Will Allow Us To Collide With Any Obstacles
		player_shape.body.apply_impulse_at_local_point(
			((numpy.cos(numpy.radians(player[PLAYER_ANGLE])) * direction[0] + numpy.cos(numpy.radians(player[PLAYER_ANGLE] + 90)) * direction[1]) * .4,
			(numpy.sin(numpy.radians(player[PLAYER_ANGLE])) * direction[0] + numpy.sin(numpy.radians(player[PLAYER_ANGLE] + 90)) * direction[1]) * .4))
		player_shape.body.velocity *= .8
			
		rotation += mouse_velocity

		#The Smaller The Value, The Smaller The Bobbing. So If The Value Is 0, The Y-Offset Will Stay At Rest
		bobbing_strength = lerp(bobbing_strength, ((keys[pygame.K_w] - keys[pygame.K_s]) != 0 or (keys[pygame.K_d] - keys[pygame.K_a]) != 0), .4)

		bobbing += .8

		if bobbing > numpy.radians(360):
			bobbing -= numpy.radians(360)

			if ((keys[pygame.K_w] - keys[pygame.K_s]) != 0 or (keys[pygame.K_d] - keys[pygame.K_a]) != 0):
				step_sound.play()

		final_bobbing = -current_offset + (numpy.sin(bobbing) / 64) * bobbing_strength

		#We Step 16 Times For Better Collisions, In My Opinion Pymunk Should Not Be Restricted
		#To Discrete Collisions, And Continous Collisions Would Be Faster Than This Solution,
		#But It Is What It Is...
		for i in range(16):
			space.step(.01)

		if len(walls_physics_body_information) > 0:
			for i in range(len(walls_physics_body_information)):
				space.remove(walls_physics_body_information[i], walls_physics_shape_information[i])

			walls_physics_body_information.clear()
			walls_physics_shape_information.clear()

		for wall in level:
			if wall[WALL_FLOOR_HEIGHT] > current_offset + .2 or wall[WALL_CEILING_HEIGHT] > .2:
				physics_geometry = pymunk.Body(body_type=pymunk.Body.STATIC)
				physics_segment = pymunk.Segment(physics_geometry, wall[WALL_POINT_A], wall[WALL_POINT_B], .2)

				space.add(physics_geometry, physics_segment)
				walls_physics_body_information.append(physics_geometry)
				walls_physics_shape_information.append(physics_segment)

		#We Don't Reset To Zero In Case The Game Is Running Slow, This Is A Sort Of "Catch-Up"
		#Where If The Framerate Is 10, Then The Game Logic Will Run 3 More Times
		update_rate -= (1000 / 30)
		mouse_velocity = 0

		current_time = pygame.time.get_ticks()
		time_between_physics = current_time - old_time
		old_time = current_time

	if time_between_physics != 0:
		interpolated_position = (lerp(old_position[0], player_body.position[0], update_rate / time_between_physics), lerp(old_position[1], player_body.position[1], update_rate / time_between_physics))
		interpolated_rotation = lerp(old_rotation, rotation, update_rate / time_between_physics)
		interpolated_bobbing = lerp(old_bobbing, final_bobbing, update_rate / time_between_physics)
	else:
		interpolated_position = player_body.position
		interpolated_rotation = rotation
		interpolated_bobbing = final_bobbing

	player = (
		interpolated_position,
		interpolated_rotation,

		player[PLAYER_VISION],
		player[PLAYER_DISTANCE],

		interpolated_bobbing,
	)

	#print(player_body.position)

	#This Is Where The Magic Happens! We Will Also Get The Current Offset Of The Segment That The Player Is On, So That They Will Be Raised Accordingly
	offset = scan_line(player, level, buffer, tree_thing)
	pygame.surfarray.blit_array(screen_surface, buffer)

	screen_surface.blit(font.render("FPS: " + str(int(fps)), False, (255, 255, 255)), (0, 0))
	#print(interpolated_rotation)

	#Translate Position
	#dx = 66 - interpolated_position[0]
	#dy = 70 - interpolated_position[1]

	#dist = numpy.sqrt(dx * dx + dy * dy)

	#dist = numpy.sqrt(dx * dx + dy * dy)
	#theta = numpy.degrees(numpy.arctan2(-dy, dx))
	#interpolated_rotation %= 360

	#y = (-interpolated_rotation + (player[PLAYER_VISION] / 2) - theta)

	#if y < -180:
	#	y += 360
	#x = y * (512 / player[PLAYER_VISION])
	#tree_height = (256 / dist)

	#sized_thing = pygame.transform.scale(tree_thing, (tree_height, tree_height))

	#screen_surface.blit(sized_thing, (x - sized_thing.get_width() / 2, 256))
	pygame.display.flip()

	#This Is Unecessary In Closed Areas, But Performance Seems To Be Very Minimal, Might Be Removed After
	#Implementing Skyboxes
	buffer.fill(0)

	#We Increment By The Time It Took To Render & Update Everything
	#Whenever We Reach 33 Milliseconds (30 FPS), The Game Logic Will Execute

	new_time = pygame.time.get_ticks()
	update_rate += (new_time - previous_time)
	fps = 1 / ((new_time - previous_time) / 1000)

	#Why Not Multiply By Delta Time? Physics Needs To Be Consistent, And Using Delta Time
	#For Physics Is Not Good Practise, We Also Can Have An Easier Time Implement Logic,
	#As We Don't Have To Figure Out Different Solutions To Make Something Framerate Independent
	previous_time = new_time

pygame.quit()