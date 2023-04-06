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

	#This Will Order By Segments
	#Not Entirely Happy With This Algorithm But It Works
	if len(all_walls_intersected) > 0:
		single_value = -1
		found = False

		#Here The Walls Are Ordered By Segments
		for i in range(1, len(all_walls_intersected)):
			found = False

			for j in range(i, len(all_walls_intersected)):
				if all_walls_intersected[i - 1][INTERSECTED_WALL][WALL_SEGMENT] == all_walls_intersected[j][INTERSECTED_WALL][WALL_SEGMENT]:
					found = True
					all_walls_intersected[i], all_walls_intersected[j] = all_walls_intersected[j], all_walls_intersected[i]

			if single_value == -1 and not found:
				single_value = i

		if found == False:
			all_walls_intersected.insert(0, all_walls_intersected.pop(single_value))

	return all_walls_intersected


#Scan The Entire Screen From Left To Right & Render The Walls & Floors
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

					else:
						cull_wall = True
						floor_length = buffer.shape[1]
						ceiling_length = 0

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

				#These Are Stored For Later Comparisions
				previous_floor_height = floor_height
				previous_ceiling_height = ceiling_height


#--------------------------------
#Main Loop
#--------------------------------


should_cap = False

#Physics
space = pymunk.Space()
space.gravity = (0, 0)

pygame.init()
pygame.mixer.init()

screen_surface = pygame.display.set_mode((512, 512), pygame.SCALED | pygame.FULLSCREEN, vsync=True)
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

#Create Player
player = ((67, 63), 0, 75, 128, 0)

#Create Frame Counter
#clock = pygame.time.Clock()
#font = pygame.font.SysFont("Monospace" , 16 , bold = False)

#Level Data, This Is Temporary And Will Instead Load Through External Files In A Later Version
level = (
	((64, 71), (70, 71), 0.6, 0.0, 0, basic_wall_1, basic_wall_2),
	((70, 71), (70, 77), 0.6, 0.0, 0, basic_wall_1, basic_wall_2),
	((64, 71), (70, 77), 0.6, 0.0, 0, basic_wall_1, basic_wall_2),

	((64, 64), (64, 71), 0.0, 0.2, 2, basic_wall_4, basic_wall_4),
	((64, 71), (70, 71), 0.0, 0.2, 2, basic_wall_4, basic_wall_4),
	((70, 71), (70, 70), 0.0, 0.2, 2, basic_wall_4, basic_wall_4),
	((70, 70), (64, 64), 0.0, 0.2, 2, basic_wall_4, basic_wall_4),

	((64, 64), (70, 64), 0.2, 0.6, 1, basic_wall_2, basic_wall_3),
	((70, 64), (70, 70), 0.2, 0.6, 1, basic_wall_2, basic_wall_3),
	((64, 64), (70, 70), 0.2, 0.6, 1, basic_wall_2, basic_wall_3),
)

#Adding The Walls To The Physics Engine
for wall in level:
	if wall[WALL_FLOOR_HEIGHT] > 0:
		physics_geometry = pymunk.Body(body_type=pymunk.Body.STATIC)
		space.add(physics_geometry, pymunk.Segment(physics_geometry, wall[WALL_POINT_A], wall[WALL_POINT_B], .2))

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

while running:
	keys = pygame.key.get_pressed()

	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False

		if event.type == pygame.MOUSEMOTION:
			mouse_velocity += event.rel[0] * .05

		if event.type == pygame.KEYDOWN:
			if pygame.key.get_pressed()[pygame.K_SPACE]:
				if should_cap:
					should_cap = False
				else:
					should_cap = True

	#This Is So That Game Logic Will Not Be Tied To The Rendering Speed, We Can Also Now Do Interpolation
	while update_rate >= 1000 / 30:
		old_position = player_body.position
		old_rotation = rotation
		old_bobbing = final_bobbing

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

		final_bobbing = (numpy.sin(bobbing) / 64) * bobbing_strength

		#We Step 16 Times For Better Collisions, In My Opinion Pymunk Should Not Be Restricted
		#To Discrete Collisions, And Continous Collisions Would Be Faster Than This Solution,
		#But It Is What It Is...
		for i in range(16):
			space.step(.01)

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

	#This Is Where The Magic Happens!
	scan_line(player, level, buffer)
	pygame.surfarray.blit_array(screen_surface, buffer)

	#screen_surface.blit(font.render("FPS: " + str(int(clock.get_fps())), False, (255, 255, 255)), (0, 0))
	#screen_surface.blit(font.render("Logic Delta: " + str(int(time_between_physics)), False, (255, 255, 255)), (0, 10))
	pygame.display.flip()

	#This Is Unecessary In Closed Areas, But Performance Seems To Be Very Minimal, Might Be Removed After
	#Implementing Skyboxes
	buffer.fill(0)

	#We Increment By The Time It Took To Render & Update Everything
	#Whenever We Reach 33 Milliseconds (30 FPS), The Game Logic Will Execute

	new_time = pygame.time.get_ticks()
	update_rate += (new_time - previous_time)

	#Why Not Multiply By Delta Time? Physics Needs To Be Consistent, And Using Delta Time
	#For Physics Is Not Good Practise, We Also Can Have An Easier Time Implement Logic,
	#As We Don't Have To Figure Out Different Solutions To Make Something Framerate Independent
	previous_time = new_time

pygame.quit()