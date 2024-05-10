import numpy as np
import matplotlib.pyplot as plt
import random
random.seed(13)
import copy
import csv
from game_ai.tetris_base import *
FILE_PATH = "F:\FCAI\AI\Second Semester\Cognitive Science\project\Cognitive_Course_Project\logfile.txt"
DATA_FILE_PATH = "F:\FCAI\AI\Second Semester\Cognitive Science\project\Cognitive_Course_Project\Data.csv"
NUM_CHROMOSOMES = 12
NUM_GENES = 9
ITERATIONS = 300
NUM_EVOLUTIONS = 1
MUT_RATE = 0.1
CROSSOVER_RATE = 0.3
MAX_SCORE = 400000

# ***********************************************************
# ***********************************************************


def Initialize_Chormosomes():
    """
    Initializes a list of chromosomes.

    Returns:
    - List: A list of randomly generated chromosomes.
    """
    # Empty list to store chromosomes
    chromosomes = []
    # Loop to create specified number of chromosomes
    for _ in range(NUM_CHROMOSOMES):
        # Generate a random list of genes for each chromosome
        chromosome = [round(random.uniform(-100, 100), 2)
                      for _ in range(NUM_GENES)]
        # Add the chromosome to the list of chromosomes
        chromosomes.append(chromosome)
    # Return the list of chromosomes
    return chromosomes
# ***********************************************************
# ***********************************************************


def obj_function(mx_hieght_cur, holes_cur, mx_hieght_nxt, holes_nxt, cleared_rows, piece_sides, floor_sides, wall_sides, score, chromosome):
    """
    Calculates the objective function value based on input features and weights specified in the chromosome.

    Args:
    - mx_hieght_cur: Current maximum height.
    - holes_cur: current Number of holes.
    - mx_hieght_nxt: Maximum height after current move.
    - holes_nxt: Number of holes after current move.
    - cleared_rows: Number of rows cleared by the current move.
    - piece_sides: Number of sides of the piece touching other blocks
    - floor_sides: Number of sides of the piece touching the floor
    - wall_sides: Number of sides of the piece touching the wall
    - score: Current game score
    - chromosome: List of weights

    Returns:
    - Objective function value
    """
    return chromosome[0] * mx_hieght_cur + chromosome[1] * holes_cur + \
        chromosome[2] * mx_hieght_nxt + chromosome[3] * holes_nxt + \
        chromosome[4] * cleared_rows + chromosome[5] * piece_sides + \
        chromosome[6] * floor_sides + chromosome[7] * wall_sides +\
        chromosome[8] * score
# ***********************************************************
# ***********************************************************


def calc_fitness(game_state):
    """
    Calculates the fitness of a given game state.

    Args:
    # game_state = [num_used_pieces, removed_lines, score, win]
    - game_state: A list representing the game state containing relevant information.
                  Index 1 contains the current score.
                  Index -1 indicates whether the game is over (True/False).

    Returns:
    - int: The fitness calculated based on the game state.
    """
    score = game_state[1]
    if (game_state[-1] == True):
        score += 500
    return score
# ***********************************************************
# ***********************************************************


def get_best_move(board, piece, score, chromo, display_piece=False):
    """
    Calculates the best move for the current piece on the board.

    Args:
    - board: The game board.
    - piece: The current piece to be placed on the board (falling piece).
    - score: The current score in the game.
    - chromo: The chromosome containing weights for the objective function.
    - display_piece: Boolean indicating whether to show the piece or not (default is False).

    Returns:
    - Tuple: The best X  and best rotation for the falling piece.
    """
    # Initialize variables to store the best move
    x_best = 0
    y_best = 0
    r_best = 0
    best_score = -9000000  # Initialize with  low value

    # Calculate the total holes and total blocks above holes before play
    # total_holes, total_blocking_bocks, total_sum_heights
    init_move_info = calc_initial_move_info(board)
    # print(f"init_move_info : {init_move_info}")

    # Iterate through every possible rotation of the piece
    for r in range(len(PIECES[piece['shape']])):
        # Iterate through every possible position on the board
        for x in range(-2, BOARDWIDTH-2):
            # Calculate movement information for the current move (falling piece)
            # [True, max_height, num_removed_lines, new_holes, new_blocking_blocks, piece_sides, floor_sides, wall_sides]
            movement_info = calc_move_info(board, piece, x, r,
                                           init_move_info[0],
                                           init_move_info[1])

            # Check if it's a valid movement
            if (movement_info[0]):
                # Calculate movement score using the objective function
                movement_score = obj_function(init_move_info[2], init_move_info[0], movement_info[1], movement_info[3],
                                              movement_info[2], movement_info[-3], movement_info[-2], movement_info[-1],  score, chromo)

                # Update best movement if the score is better
                if (movement_score > best_score):
                    best_score = movement_score
                    x_best = x
                    y_best = piece['y']
                    r_best = r
    # Adjust piece position based on whether to show the game or not
    if (display_piece):
        piece['y'] = y_best
    else:
        piece['y'] = -2  # Move the piece out of the visible area

    # Set the best X and rotation for the piece
    piece['x'] = x_best
    piece['rotation'] = r_best

    # Return the best X coordinate and rotation
    return x_best, r_best
# ***********************************************************
# ***********************************************************


def draw_game_on_screen(board, score, level, next_piece, falling_piece):
    """
    Draw the game on the screen.

    Args:
    - board: The game board.
    - score: The current score in the game.
    - level: The current level of the game.
    - next_piece: The next piece to appear in the game.
    - falling_piece: The current falling piece on the board.
    """
    # Fill the screen with the background color
    """Draw game on the screen"""
    DISPLAYSURF.fill(BGCOLOR)
    draw_board(board)  # Draw the game board
    draw_status(score, level)  # Draw the score and level information
    draw_next_piece(next_piece)  # Draw the next piece preview

    # If there is a falling piece, draw it
    if falling_piece != None:
        draw_piece(falling_piece)

    pygame.display.update()  # Update the display
    FPSCLOCK.tick(FPS)      # Control the frame rate
# ***********************************************************
# ***********************************************************


def run_single_chromo(chromo, max_score=MAX_SCORE, show=False):
    """
    Simulates a game using a single chromosome.

    Args:
    - chromosome: The chromosome containing weights.
    - max_score: The maximum score to achieve before ending the game (default is 90000).
    - show: Boolean indicating whether to display the game on screen or not (default is False).

    Returns:
    - List: A list representing the game state after the game session.
            The list contains:
            - The number of used pieces.
            - The number of lines removed at each step (list).
            - The final score.
            - Whether the game was won or not.
    """
    chromosome = copy.deepcopy(chromo)
    board = get_blank_board()
    last_fall_time = time.time()
    score = 0
    level, fall_freq = calc_level_and_fall_freq(score)
    falling_piece = get_new_piece()
    next_piece = get_new_piece()

    # Calculate best move for the falling piece
    # x_best, r_best
    get_best_move(board, falling_piece, score, chromosome)

    num_used_pieces = 0

    is_win = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Game exited by user")
                exit()

        if falling_piece == None:
            falling_piece = next_piece
            next_piece = get_new_piece()

            # Calculate best move for the falling piece
            # x_best, r_best
            get_best_move(board, falling_piece, score, chromosome)

            # Update number of used pieces and the score
            num_used_pieces += 1
            score += 1

            # Reset last_fall_time
            last_fall_time = time.time()

            if (not is_valid_position(board, falling_piece)):
                # GAME-OVER
                # Can't fit a new piece on the board, so game over.
                break

        if not show or time.time() - last_fall_time > fall_freq:
            if (not is_valid_position(board, falling_piece, adj_Y=1)):
                # Falling piece has landed, set it on the board
                add_to_board(board, falling_piece)

                # Bonus score for complete lines at once
                # 40   pts for 1 line
                # 120  pts for 2 lines
                # 300  pts for 3 lines
                # 1200 pts for 4 lines
                num_removed_lines = remove_complete_lines(board)
                if (num_removed_lines == 1):
                    score += 40
                elif (num_removed_lines == 2):
                    score += 120
                elif (num_removed_lines == 3):
                    score += 300
                elif (num_removed_lines == 4):
                    score += 1200
                level, fall_freq = calc_level_and_fall_freq(score)
                falling_piece = None

            else:
                # Piece did not land, just move the piece down
                falling_piece['y'] += 1
                last_fall_time = time.time()

        if (show):
            draw_game_on_screen(board, score, level, next_piece, falling_piece)

        if (score > max_score):
            is_win = True
            break

    # Save the game state
    game_state = [num_used_pieces, score, is_win]

    return game_state
# ***********************************************************
# ***********************************************************


def parent_selection(chromos, fitness_vals):
    """
    Selects parents from the population

    Args:
    - chromosomes: A list of chromosomes representing the population.
    - fitness: A list containing the fitness values of each chromosome.

    Returns:
    - List: A list of selected parent chromosomes.
    """
    chromosomes = copy.deepcopy(chromos)
    fitness = copy.deepcopy(fitness_vals)
    fitness = np.array(fitness)
    fitness_sum = sum(fitness)
    # Calculate the probabilities.
    fitness_probs = np.round(fitness/fitness_sum, 4)
    # Calculate cumulative probabilities.
    cumulative_sum = list()
    cum_sum = 0
    for i in range(len(fitness_probs)):
        cum_sum += fitness_probs[i]
        cumulative_sum.append(cum_sum)
    # Generate random numbers for selection
    R_probs = [random.random() for _ in range(NUM_CHROMOSOMES)]
    # Select parents using roulette wheel selection
    selected_pop = list()
    for R_num in R_probs:
        for i, cum_num in enumerate(cumulative_sum):
            if R_num <= cum_num:
                selected_pop.append(chromosomes[i])
                break

    return selected_pop
# ***********************************************************
# ***********************************************************


def crossover(pop):
    """
    Performs crossover operation on a population of chromosomes.

    Args:
    - population: A list of chromosomes representing the population.

    Returns:
    - List: A list of chromosomes after crossover operation.
    """
    population = copy.deepcopy(pop)
    crossover_population = []
    # Iterate over each chromosome in the population
    for chromo in population:
        # Generate a random number
        num = random.random()
        # Check if crossover should be performed based on crossover rate
        if (num > CROSSOVER_RATE):
            # Select a random parent from the population
            parent2 = random.choice(population)
            while (chromo == parent2):
                parent2 = random.choice(population)
            # Select a crossover point
            point = len(chromo)//2
            # Perform crossover by combining the first part of the current chromosome with the second part of the selected parent chromosome
            child = chromo[0:point] + parent2[point:]
            # Add the child chromosome to the crossover population
            crossover_population.append(child)
        else:
            crossover_population.append(chromo)
    return crossover_population
# ***********************************************************
# ***********************************************************


def mutation(pop):
    """
    Performs mutation operation on a population of chromosomes.

    Args:
    - population: A list of chromosomes representing the population.

    Returns:
    - List: A list of chromosomes after mutation operation.
    """
    population = copy.deepcopy(pop)
    # Iterate over each chromosome in the population
    for chromo in population:
        # Determine the number of mutation replacements for the current chromosome
        num_of_mutation_replacement = random.randint(0, len(chromo))
        # Perform mutation for the determined number of replacements
        for _ in range(num_of_mutation_replacement):
            # Select a random position for mutation replacement
            position_of_mutation_replacement = random.randint(0, len(chromo)-1)
            # Check if mutation should be performed based on mutation rate
            if random.random() < MUT_RATE:
                # Generate a random gene for mutation replacement
                random_gene = round(random.uniform(-100, 100), 4)
                # Perform mutation replacement at the selected position
                chromo[position_of_mutation_replacement] = random_gene
    return population
# ***********************************************************
# ***********************************************************


def replacement(chromos, fit):
    """
    Performs replacement operation in the genetic algorithm.

    Args:
    - chromosomes: A list of chromosomes representing the current population.
    - fitness: A list containing the fitness values of each chromosome.

    Returns:
    - updated chromosomes and fitness values after replacement.
    """
    new_chromosome = []
    chromosomes = copy.deepcopy(chromos)
    fitness = copy.deepcopy(fit)
    # Combine chromosomes with their fitness values
    for i in range(len(chromosomes)):
        t = [chromosomes[i], fitness[i]]
        new_chromosome.append(t)
    # Sort the combined list based on fitness values in descending order
    sorted_chromo = sorted(new_chromosome, key=lambda x: x[1], reverse=True)
    # Select the top half of the sorted list to survive
    sorted_chromo = sorted_chromo[:int(len(new_chromosome)/2)]

    # Separate chromosomes and fitness values
    chromosomes = []
    fitness = []
    for i in range(len(sorted_chromo)):
        chromo = sorted_chromo[i][0]
        fit = sorted_chromo[i][1]
        chromosomes.append(chromo)
        fitness.append(fit)
    return chromosomes, fitness
# ***********************************************************
# ***********************************************************


def clear_file():
    file_path = FILE_PATH
    with open(file_path, "w") as file:
        pass  # Using pass to do nothing, effectively clears the file


def write_to_file(chromosomes, fitness, i, evol):
    # Open a file in append mode (creates a new file if it doesn't exist)
    with open(FILE_PATH, "a") as file:
        # Append content to the file
        file.write(
            f" |iteration :  {i}\n |  chromosomes: \n |  {chromosomes}\n |  fitness: {fitness}\n |  best_score: {max(fitness)}\n")
        file.write(
            " |**************************************************************************\n")

        # Flush the buffer to ensure data is written to the file immediately
        file.flush()

# ***********************************************************
# ***********************************************************


def write_data_to_file(data):
    # Header for the CSV file
    header = ["Evolution", "Iteration", "Chromosome", "Fitness"]

    # Writing data to CSV file
    with open(DATA_FILE_PATH, 'w', newline='') as file:
        writer = csv.writer(file)

        # Write the header
        writer.writerow(header)

        # Write each row of data
        for row in data:
            writer.writerow(row)
            file.flush()

    print("CSV file created successfully.")


def run_game_ai():

    clear_file()  # clear log file
    csv_list = []  # create csv file to easy analyse

    # loop on NUM_EVOLUTIONS (10)
    for evol in range(NUM_EVOLUTIONS):
        cnt_max_score = 0
        curr_max_score = 0
        best_chromo = []
        best_fitness = []
        with open(FILE_PATH, "a") as file:
            # Append content to the file
            file.write(
                f"===========================================================================\n")
            file.write(f"Evolution : {evol}\n")

        # Initialize Chormosomes for the Evolution
        chromosomes = Initialize_Chormosomes()
        # save fitness for the Chormosomes
        Fitness_vals = list()

        # loop on Initialized Chormosomes
        for chromo in chromosomes:
            # run_single_chromo for each chromosome in Initialized Chormosomes and calc fitness values
            game_state = run_single_chromo(chromo)
            fitness_val = calc_fitness(game_state)
            # append fitness value for each Chormosomes on fitness values
            Fitness_vals.append(fitness_val)

        # loop on ITERATIONS (400)
        for i in range(ITERATIONS):
            if curr_max_score == NUM_CHROMOSOMES:
                print("ALL MAX SCORE")
            elif cnt_max_score < NUM_CHROMOSOMES:
                # get best (half) chromosomes and it's fitness
                best_chromo1, best_fitness1 = replacement(
                    chromosomes, Fitness_vals)

                # Apply selection , crossover and mutation
                parents = parent_selection(chromosomes, Fitness_vals)
                parents = crossover(parents)
                parents = mutation(parents)

                # empty Fitness_vals list to save the new fitness
                Fitness_vals = []

                # loop on parents
                print(f"Before FITNESS: ")
                print(f"cnt_max_score: {cnt_max_score}")
                print(f"best_chromo: {best_chromo}")
                for par in range(NUM_CHROMOSOMES):
                    print(f"parents[par]: {parents[par]} ==> ", end='')
                    if parents[par] in best_chromo:
                        print("Found")
                        fitness_val = best_fitness[best_chromo.index(
                            parents[par])]
                    else:
                        print("Not Found")
                        # run_single_chromo for each parent  and calc fitness values
                        game_state = run_single_chromo(parents[par])
                        fitness_val = calc_fitness(game_state)
                    # append fitness value for each parent on fitness values
                    Fitness_vals.append(fitness_val)

                # get best (half) parents and it's fitness
                best_chromo2, best_fitness2 = replacement(
                    parents, Fitness_vals)

                # concatinate the best_chromo1 + best_chromo2 and best_fitness1 + best_fitness2
                chromosomes = best_chromo1 + best_chromo2
                Fitness_vals = best_fitness1 + best_fitness2
                curr_max_score = 0
                for index in range(NUM_CHROMOSOMES):
                    if Fitness_vals[index] > MAX_SCORE:
                        curr_max_score += 1
                    if Fitness_vals[index] >= MAX_SCORE and chromosomes[index] not in best_chromo and cnt_max_score < NUM_CHROMOSOMES:
                        best_chromo.append(copy.deepcopy(chromosomes[index]))
                        best_fitness.append(copy.deepcopy(Fitness_vals[index]))
                        cnt_max_score += 1
                        print(f"cnt_max_score: {cnt_max_score}")
                print(f"curr_max_score: {curr_max_score}")
            else:
                chromosomes = best_chromo
                Fitness_vals = best_fitness
            # Add row for Each Chromosome
            for num in range(NUM_CHROMOSOMES):
                csv_row = []
                csv_row.append(evol)
                csv_row.append(i)
                csv_row.append(chromosomes[num])
                csv_row.append(Fitness_vals[num])
                csv_list.append(csv_row)
                # print(f"csv :{csv_row}")

            # write_to_file
            write_to_file(chromosomes, Fitness_vals, i, evol)

    write_data_to_file(csv_list)
