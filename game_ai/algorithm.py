from game_ai.tetris_base import *

NUM_CHROMOSOMES = 12
NUM_GENES = 8
ITERATIONS = 400
MUT_RATE = 0.1
CROSSOVER_RATE = 0.3


def Initialize_Chormosomes():
    chromosomes = []
    for _ in range(NUM_CHROMOSOMES):
        chromosome = [round(random.uniform(-1000, 1000), 2) for _ in range(NUM_GENES)]
        chromosomes.append(chromosome)
    return chromosomes


def obj_function(mx_hieght_cur, holes_cur, sum_hieghts_cur, mx_hieght_nxt, holes_nxt, sum_hieghts_nxt, cleared_rows, score, chromosome):
    return chromosome[0] * mx_hieght_cur + chromosome[1] * holes_cur + chromosome[2] * sum_hieghts_cur + \
        chromosome[3] * mx_hieght_nxt + chromosome[4] * holes_nxt + chromosome[5] * sum_hieghts_nxt + \
        chromosome[6] * cleared_rows + chromosome[7] * score


def fitness(obj_val):
    return round(1/obj_val, 4)


def run_game_ai():
    # Setup variables
    board = get_blank_board()
    last_movedown_time = time.time()
    last_moveside_time = time.time()
    last_fall_time = time.time()
    moving_down = False  # note: there is no movingUp variable
    moving_left = False
    moving_right = False
    score = 0
    level, fall_freq = calc_level_and_fall_freq(score)

    falling_piece = get_new_piece()
    next_piece = get_new_piece()

    chromosomes = Initialize_Chormosomes()
    total_holes_bef, total_blocking_bloks_bef = calc_initial_move_info(board)
    initial_move_info = calc_initial_move_info(board)
    move_info = calc_move_info(board=board, piece=falling_piece, x=falling_piece['x'], r=falling_piece['r'],
                   total_holes_bef=total_holes_bef, total_blocking_bloks_bef=total_blocking_bloks_bef)
    while True:
        # Game Loop
        if (falling_piece == None):
            # No falling piece in play, so start a new piece at the top
            falling_piece = next_piece
            next_piece = get_new_piece()
            score += 1
            
            total_holes_bef, total_blocking_bloks_bef = calc_initial_move_info(board)
            initial_move_info = calc_initial_move_info(board)
            move_info = calc_move_info(board=board, piece=falling_piece, x=falling_piece['x'], r=falling_piece['r'],
                   total_holes_bef=total_holes_bef, total_blocking_bloks_bef=total_blocking_bloks_bef)

            # [True, max_height, num_removed_lines, new_holes, new_blocking_blocks, piece_sides, floor_sides, wall_sides]
            # obj_function(mx_hieght_cur, holes_cur, sum_hieghts_cur, mx_hieght_nxt, holes_nxt, sum_hieghts_nxt, cleared_rows, score, chromosome):
            obj_function(move_info[1])
            # Reset last_fall_time
            last_fall_time = time.time()

            if (not is_valid_position(board, falling_piece)):
                # GAME-OVER
                # Can't fit a new piece on the board, so game over.
                return

        # Check for quit
        check_quit()
    obj_function()
