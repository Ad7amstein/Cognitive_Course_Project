import numpy as np
from game_ai.tetris_base import *

NUM_CHROMOSOMES = 12
NUM_GENES = 5
ITERATIONS = 400
MUT_RATE = 0.1
CROSSOVER_RATE = 0.3


def Initialize_Chormosomes():
    chromosomes = []
    for _ in range(NUM_CHROMOSOMES):
        chromosome = [round(random.uniform(-1000, 1000), 2) for _ in range(NUM_GENES)]
        chromosomes.append(chromosome)
    return chromosomes


def obj_function(mx_hieght_cur, holes_cur, mx_hieght_nxt, holes_nxt, cleared_rows, score, chromosome):
    return chromosome[0] * mx_hieght_cur + chromosome[1] * holes_cur + \
        chromosome[2] * mx_hieght_nxt + chromosome[3] * holes_nxt + \
        chromosome[4] * cleared_rows + chromosome[5] * score


def calc_fitness(game_state):
    score = game_state[2]
    return score


def calc_best_move(board, piece, chromo, show_game = False):
    best_X     = 0
    best_R     = 0
    best_Y     = 0
    best_score = -100000

    # Calculate the total the holes and blocks above holes before play
    init_move_info = calc_initial_move_info(board)
    # total_holes, total_blocking_bocks, total_sum_heights
    for r in range(len(PIECES[piece['rotation']])):
        # Iterate through every possible rotation
        for x in range(-2,BOARDWIDTH-2):
            #Iterate through every possible position
            # [True, max_height, num_removed_lines, new_holes, new_blocking_blocks, piece_sides, floor_sides, wall_sides]
            movement_info = calc_move_info(board, piece, x, r, \
                                                init_move_info[0], \
                                                init_move_info[1])

            # Check if it's a valid movement
            if (movement_info[0]):
                # Calculate movement score
                # mx_hieght_cur, holes_cur, mx_hieght_nxt, holes_nxt, cleared_rows, score, chromosome
                movement_score = obj_function(init_move_info[2], init_move_info[0], movement_info[1], movement_info[3], movement_info[2], 0, chromo)

                # Update best movement
                if (movement_score > best_score):
                    best_score = movement_score
                    best_X = x
                    best_R = r
                    best_Y = piece['y']

    if (show_game):
        piece['y'] = best_Y
    else:
        piece['y'] = -2

    piece['x'] = best_X
    piece['rotation'] = best_R

    return best_X, best_R


def run_single_chromo(chromosome, max_score = 20000, no_show = False):
    board            = get_blank_board()
    last_fall_time   = time.time()
    score            = 0
    level, fall_freq = calc_level_and_fall_freq(score)
    falling_piece    = get_new_piece()
    next_piece       = get_new_piece()

    # Calculate best move
    calc_best_move(board, falling_piece, chromosome)

    num_used_pieces = 0
    removed_lines   = [0,0,0,0] # Combos

    win   = False

    # Game loop
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print ("Game exited by user")
                exit()

        if falling_piece == None:
            # No falling piece in play, so start a new piece at the top
            falling_piece = next_piece
            next_piece    = get_new_piece()

            # Decide the best move based on your weights
            calc_best_move(board, falling_piece, chromosome)

            # Update number of used pieces and the score
            num_used_pieces += 1
            score           += 1

            # Reset last_fall_time
            last_fall_time = time.time()

            if (not is_valid_position(board, falling_piece)):
                # GAME-OVER
                # Can't fit a new piece on the board, so game over.
                break

        if no_show or time.time() - last_fall_time > fall_freq:
            if (not is_valid_position(board, falling_piece, adj_Y=1)):
                # Falling piece has landed, set it on the board
                add_to_board(board, falling_piece)

                # Bonus score for complete lines at once
                # 40   pts for 1 line
                # 120  pts for 2 lines
                # 300  pts for 3 lines
                # 1200 pts for 4 lines
                num_removed_lines = remove_complete_lines(board)
                if(num_removed_lines == 1):
                    score += 40
                    removed_lines[0] += 1
                elif (num_removed_lines == 2):
                    score += 120
                    removed_lines[1] += 1
                elif (num_removed_lines == 3):
                    score += 300
                    removed_lines[2] += 1
                elif (num_removed_lines == 4):
                    score += 1200
                    removed_lines[3] += 1

                falling_piece = None
            else:
                # Piece did not land, just move the piece down
                falling_piece['y'] += 1
                last_fall_time = time.time()

        # if (not no_show):
        #     draw_game_on_screen(board, score, level, next_piece, falling_piece,
        #                         chromosome)

        # Stop condition
        if (score > max_score):
            win   = True
            break

    # Save the game state
    game_state = [num_used_pieces, removed_lines, score, win]

    return game_state


def parent_selection(chromosomes, fitness):
    fitness_sum = fitness.sum()
    fitness_probs  = round(fitness/fitness_sum, 4)

    cumulative_sum = list()
    cum_sum = 0
    for i in range(len(fitness_probs)):
        cum_sum += fitness_probs[i]
        cumulative_sum.append(cum_sum)

    R_probs = [random.random() for _ in range(NUM_CHROMOSOMES)]
    selected_pop = list()
    for R_num in R_probs:
        for i, cum_num in enumerate(cumulative_sum):
            if R_num <= cum_num:
                selected_pop.append(chromosomes[i])
                break

    return selected_pop


def run_game_ai():
    chromosomes = Initialize_Chormosomes()
    Fitness_vals = list()
    for chromo in chromosomes:
        game_state = run_single_chromo(chromo)
        fitness_val = calc_fitness(game_state)
        Fitness_vals.append(fitness_val)

    for i in range(ITERATIONS):
        parents = parent_selection(chromosomes, Fitness_vals)









    # print(f"chromosomes: {chromosomes}\n")
    initial_move_info = calc_initial_move_info(board)
    # print(f"initial_move_info: {initial_move_info}")
    move_info = calc_move_info(board=board, piece=falling_piece, x=falling_piece['x'], r=falling_piece['rotation'],
                   total_holes_bef=initial_move_info[0], total_blocking_bloks_bef=initial_move_info[1])
    # print(f"move_info: {move_info}")
    while True:
        # Game Loop
        if (falling_piece == None):
            # No falling piece in play, so start a new piece at the top
            falling_piece = next_piece
            next_piece = get_new_piece()
            score += 1
            
            initial_move_info = calc_initial_move_info(board)
            print(f"initial_move_info: {initial_move_info}")
            move_info = calc_move_info(board=board, piece=falling_piece, x=falling_piece['x'], r=falling_piece['r'],
                   total_holes_bef=initial_move_info[0], total_blocking_bloks_bef=initial_move_info[1])
            print(f"move_info: {move_info}")
                # Check if it's a valid movement
            if (move_info[0]):
                # Calculate movement score
                # [True, max_height, num_removed_lines, new_holes, new_blocking_blocks, piece_sides, floor_sides, wall_sides]
                # obj_function(mx_hieght_cur, holes_cur, sum_hieghts_cur, mx_hieght_nxt, holes_nxt, sum_hieghts_nxt, cleared_rows, score, chromosome):
                movement_score = obj_function(move_info[1])
                # Reset last_fall_time
                last_fall_time = time.time()

                if (not is_valid_position(board, falling_piece)):
                    # GAME-OVER
                    # Can't fit a new piece on the board, so game over.
                    return

        # Check for quit
        check_quit()
    obj_function()
