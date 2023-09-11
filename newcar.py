# This Code is Heavily Inspired By The YouTuber: Cheesy AI
# Code Changed, Optimized And Commented By: NeuralNine (Florian Dedov)
# This code has again been hoisted by the CGS Digital Innovation Department
# giving credit to the above authors for the benfit of our education in ML

import math
import random
import sys
import os

import neat
import pygame

# Constants
# WIDTH = 1880
# HEIGHT = 1480

WIDTH = 2500
HEIGHT = 1500

CAR_SIZE_X = 60
CAR_SIZE_Y = 60

maps = ['map.png', 'map2.png', 'map3.png', 'map4.png', 'map5.png']
chosen_map = maps[1]
BORDER_COLOR = (255, 255, 255, 255)  # Color To Crash on Hit

current_generation = 0  # Generation counter
"""
The Car Class
    The car class sets the parameters of each car, i.e. the numbers of radars, the starting position, as well as the fitness variable.
    Some of the functionality outlined below includes the initialisation (loading the sprite) and visualisation, checking whether it has collided with the road borders of the map, updates each frame of the pygame,
    setting the number of radars, whether it is still alive, and simulates how the car will run.

    In terms of the mechanisms of evolution, the population first determines the surroundings of the car using the distance
    of radars to calculate spatial information the neural network can work with. This information, a.k.a the input or first
    layer of neurons, is then connected through weightings of synapses, to a number n (0) layers of hidden nodes, eventually coming
    up with an output neuron, either turning the car left, right, decelerating, or if the model is uncertain, accelerate. After the each generation, the
    neural network is assigned a fitness variable, determing whether the output was successful and avoided a crash or increased
    distance. The most successful neural network models synapse weightings are then mimicked throughout the population of the next
    generation, creating a artifically selective environment. Initially the synapse weightings are random, however as the more
    successful neural networks survive longer and have their synapse weightings distributed into the population, thus evolving the
    population.
"""


class Car:
    """1. This Function:
        The car is visuallised through the car.png file, a constant determined by CAR_SIZE_X, CAR_SIZE_Y
        Each car is set to a constant starting position, orientation and speed, in order to create a fair test environment
        Sets a variable self.speed_set to determine whether the car has been setting their on speed later on. 
        Calculates the centre of the car, that will be used to draw radars from. 
        Sets an empty list for radars (our neurons), and radars to be drawn that will change as this is the independant variable
        Each car is either in a crashed state, or still alive state
        The dependant variable; fitness determined by a measure of distance over time.
    """

    def __init__(self):
        # Load Car Sprite and Rotate
        self.sprite = pygame.image.load("car.png").convert()  # Convert Speeds Up A Lot
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite

        # self.position = [690, 740] # Starting Position
        self.position = [830, 920]  # Starting Position
        self.angle = 0
        self.speed = 0

        self.speed_set = False  # Flag For Default Speed Later on

        self.center = [
            self.position[0] + CAR_SIZE_X / 2,
            self.position[1] + CAR_SIZE_Y / 2,
        ]  # Calculate Center

        self.radars = []  # List For Sensors / Radars
        self.drawing_radars = []  # Radars To Be Drawn

        self.alive = True  # Boolean To Check If Car is Crashed

        self.distance = 0  # Distance Driven
        self.time = 0  # Time Passed

    """ 2. This Function:
        Draws a screen in pygame, which is the car and its radars, optional to show the radars.
    """

    def draw(self, screen):
        screen.blit(self.rotated_sprite, self.position)  # Draw Sprite
        self.draw_radar(screen)  # OPTIONAL FOR SENSORS

    """ 3. This Function:
        For each radar, draws a line a and a circle, from the centre of the car
    """

    def draw_radar(self, screen):
        # Optionally Draw All Sensors / Radars
        for radar in self.radars:
            position = radar[0]
            pygame.draw.line(screen, (0, 255, 0), self.center, position, 1)
            pygame.draw.circle(screen, (0, 255, 0), position, 5)

    """ 4. This Function:
        Checks whether the car has crashed by whether it has touched the border colour of the map. 
    """

    def check_collision(self, game_map):
        self.alive = True
        for point in self.corners:
            # If Any Corner Touches Border Color -> Crash
            # Assumes Rectangle
            if game_map.get_at((int(point[0]), int(point[1]))) == BORDER_COLOR:
                self.alive = False
                break

    """ 5. This Function:
        Creates a check radar function, that will check the distance of each radar to the borders of the road in the map.
        Uses trigonometric ratios to calculate the distance using a length variable and the angle of the radar.
        Returns the coordinate variables of the radar, as well as the distance to the nearest border. 
        This will be used as our input neurons for each car's neural network. 

    """

    def check_radar(self, degree, game_map):
        length = 0
        x = int(
            self.center[0]
            + math.cos(math.radians(360 - (self.angle + degree))) * length
        )
        y = int(
            self.center[1]
            + math.sin(math.radians(360 - (self.angle + degree))) * length
        )

        # While We Don't Hit BORDER_COLOR AND length < 300 (just a max) -> go further and further
        while not game_map.get_at((x, y)) == BORDER_COLOR and length < 300:
            length = length + 1
            x = int(
                self.center[0]
                + math.cos(math.radians(360 - (self.angle + degree))) * length
            )
            y = int(
                self.center[1]
                + math.sin(math.radians(360 - (self.angle + degree))) * length
            )

        # Calculate Distance To Border And Append To Radars List
        dist = int(
            math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2))
        )
        self.radars.append([(x, y), dist])

    """ 6. This Function:
        This function sets the speed to an initial constant of 20, and allows them to speed up or down, which are two of the four output nodes.
        The sprite (car) updates itself with the correct angle, and moves it poisition according to the speed of the car and angle, making sure that the car does not go closer than 20 pixels to the edge.
        It will then add on the distance, and the time.
        The same for the Y value.
        It then calculates the new centre of the car, by dividing using the middlepoint between the car's position and size, for the x value and the y value.
        It then calculates the position of each corner of the car, by using the trigonmetric distance using half the lenght of the car (dividing it into 4 triangles) from the centre of the car.
        Finally it checks whether the car has collided with the game map, clears the values of the radars. 
    """

    def update(self, game_map):
        # Set The Speed To 20 For The First Time
        # Only When Having 4 Output Nodes With Speed Up and Down
        if not self.speed_set:
            self.speed = 20
            self.speed_set = True

        # Get Rotated Sprite And Move Into The Right X-Direction
        # Don't Let The Car Go Closer Than 20px To The Edge
        self.rotated_sprite = self.rotate_center(self.sprite, self.angle)
        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.position[0] = max(self.position[0], 20)
        self.position[0] = min(self.position[0], WIDTH - 120)

        # Increase Distance and Time
        self.distance += self.speed
        self.time += 1

        # Same For Y-Position
        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        self.position[1] = max(self.position[1], 20)
        self.position[1] = min(self.position[1], WIDTH - 120)

        # Calculate New Center
        self.center = [
            int(self.position[0]) + CAR_SIZE_X / 2,
            int(self.position[1]) + CAR_SIZE_Y / 2,
        ]

        # Calculate Four Corners
        # Length Is Half The Side
        length = 0.5 * CAR_SIZE_X
        left_top = [
            self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * length,
            self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * length,
        ]
        right_top = [
            self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * length,
            self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * length,
        ]
        left_bottom = [
            self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * length,
            self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * length,
        ]
        right_bottom = [
            self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * length,
            self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * length,
        ]
        self.corners = [left_top, right_top, left_bottom, right_bottom]

        # Check Collisions And Clear Radars
        self.check_collision(game_map)
        self.radars.clear()
        
        # From -90 To 120 With changing Step-Size Check Radar
        for d in range(-90, 120, 36):
            self.check_radar(d, game_map)

    """ 7. This Function:  
        In this function, the car gets data from each radar, by measuring the distance from the borders of the road in the map. 
        It initiallises a list of radars that it will return, and then changes the each according variable to the radar data inputted. 
        This also sets the number of radars, with the divisor in (radar[1] / n) determining the angle of the divisor, along with the stepsize in the function above. 
    """

    def get_data(self):
        # Get Distances To Border
        radars = self.radars
        return_values = [0,0,0,0,0,0,0]
        for i, radar in enumerate(radars):
            return_values[i] = int(radar[1] /36)

        return return_values

    """ 8. This Function:
        This function returns whether the car is alive or crashed. 
    """

    def is_alive(self):
        # Basic Alive Function
        return self.alive

    """ 9. This Function:
        Calculates the reward (how successful the car model was), by dividng the distance, by the time it stays alive. 
        This is important for determining which variables are successsful in the artifially selective environment. 
    """

    def get_reward(self):
        # Calculate Reward (Maybe Change?)
        # return self.distance / 50.0
        return self.distance / 50.0

    """ 10. This Function:
        This function graphically rotates the image of the car around its centre by using pygame rotate, by copying the original image, rotating it around its centre, and then trimming off excess pixels. 
    """

    def rotate_center(self, image, angle):
        # Rotate The Rectangle
        rectangle = image.get_rect()
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_rectangle = rectangle.copy()
        rotated_rectangle.center = rotated_image.get_rect().center
        rotated_image = rotated_image.subsurface(rotated_rectangle).copy()
        return rotated_image


""" This Function:
    This function will simulate the car's environment. 
    It starts with empty collections for the nets (each car's neural network) and cars.
    It then opens the display using pygame, in fullscreen.
    Then iterates through each genome, creating a neural network using neat, resourcing the config.txt file. 
    It applies a fitness variable, and creates a new car in the car array using the Car class.
    The function creates a clock using pygame, sets the font, and loads the map as a variable.
    It creates a generation counter, and adds to the counter for each time the simulation is run.
    Not good practice, but sets a counter to limit time. 
    When the user quits, it will exit the program. 
    While the simulation has not reached the maxium time limit or all of the cars in the existing geneeration have crashed:
        For each car, it will use the corresponding neural network, inputting neurons gained from the get_data function
            The neural net then returns either a 0, 1, 2, either turning left, right, or slows down respectively, and will speed up if the output does not fit any of the indexes
        It then checks the number of cars still alive
            Creates a counter for the number of cars alive
            It runs the is_alive() function for each car, checking whether the car is still alive, adding it to the total number of cars alive.
            For each car alive, it adds its reward from the get_reward() function to the genome's fitness, thus the longer the car survives on the track, the greater its fitness. 
        If there are no more cars alive, it ends the simulation.
        When the counter reaches 120, it will end automatically.
        The function draws each car on the game map.
        Then it displays the info, the number of generations, the number of cars still alive, and sets the display to 60fps. 
"""


def run_simulation(genomes, config):
    # Empty Collections For Nets and Cars
    nets = []
    cars = []

    # Initialize PyGame And The Display
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)

    # For All Genomes Passed Create A New Neural Network
    for i, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0

        cars.append(Car())

    # Clock Settings
    # Font Settings & Loading Map
    clock = pygame.time.Clock()
    generation_font = pygame.font.SysFont("Arial", 30)
    alive_font = pygame.font.SysFont("Arial", 20)
    game_map = pygame.image.load(chosen_map).convert()  # Convert Speeds Up A Lot

    global current_generation
    current_generation += 1

    # Simple Counter To Roughly Limit Time (Not Good Practice)
    counter = 0

    while True:
        # Exit On Quit Event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

        # For Each Car Get The Acton It Takes
        for i, car in enumerate(cars):
            output = nets[i].activate(car.get_data())
            choice = output.index(max(output))
            if choice == 0:
                car.angle += 10  # Left
            elif choice == 1:
                car.angle -= 10  # Right
            elif choice == 2:
                if car.speed - 2 >= 12:
                    car.speed -= 2  # Slow Down
            else:
                car.speed += 2  # Speed Up

        # Check If Car Is Still Alive
        # Increase Fitness If Yes And Break Loop If Not
        still_alive = 0
        for i, car in enumerate(cars):
            if car.is_alive():
                still_alive += 1
                car.update(game_map)
                genomes[i][1].fitness += car.get_reward()

        if still_alive == 0:
            break

        counter += 1
        if counter == 30 * 40:  # Stop After About 20 Seconds
            break

        # Draw Map And All Cars That Are Alive
        screen.blit(game_map, (0, 0))
        for car in cars:
            if car.is_alive():
                car.draw(screen)

        # Display Info
        text = generation_font.render(
            "Generation: " + str(current_generation), True, (0, 0, 0)
        )
        text_rect = text.get_rect()
        text_rect.center = (900, 450)
        screen.blit(text, text_rect)

        text = alive_font.render("Still Alive: " + str(still_alive), True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (900, 490)
        screen.blit(text, text_rect)

        pygame.display.flip()
        clock.tick(60)  # 60 FPS


""" 1. This Section:
    Finally the code loads the config.txt file, with the variables, default genome, default reproduction rate, species set, and stagnation.
         - Default Genome: The scaffold of each genome, determining the number of input and output nodes, the weights of each mutation and connection in the neural network, and the probability of adding or deleting mutations and connections
         - Default Species Set: Sets a threshold for measuring compatibility between species. 
         - Default Reproduction: Determins the number of the best models to copy into the next generation, depending on whether they are above or below the survival threshold. 
         - Default Stagnation: species_fitness_func: Specifies how to measure the fitness of a species, uses "max," meaning the fitness of a species is determined by the fitness of its best member. Sets a limit on how many generations a species can remain stagnant (no improvement) before it's considered for removal.
    It then creates a population based on the config file,
    Adds an reporter to the population giving statistics given. 
    Runs the simulatin for a maxium of 1000 generations.   
         """
if __name__ == "__main__":
    # Load Config
    config_path = "./config.txt"
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    # Create Population And Add Reporters
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Run Simulation For A Maximum of 25 Generations
    population.run(run_simulation, 25)
