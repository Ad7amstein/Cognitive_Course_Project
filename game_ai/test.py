import random
random.seed(42)
from game_ai.tetris_base import *
FILE_PATH = "F:\FCAI\AI\Second Semester\Cognitive Science\project\Cognitive_Course_Project\log_file.txt"
NUM_CHROMOSOMES = 14
NUM_GENES = 9
ITERATIONS = 400
MUT_RATE = 0.1
CROSSOVER_RATE = 0.3

#***********************************************************
#***********************************************************
def obj_function(mx_hieght_cur, holes_cur, mx_hieght_nxt, holes_nxt, cleared_rows, piece_sides, floor_sides, wall_sides , score ,chromosome ):
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
           chromosome[4] * cleared_rows  + chromosome[5] * piece_sides + \
           chromosome[6] * floor_sides   + chromosome[7] * wall_sides +\
           chromosome[8] * score
#***********************************************************
#***********************************************************
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
    if (game_state[-1] == True) :
        score += 500
    return score
#***********************************************************
#***********************************************************
def get_best_move(board, piece, score ,chromo, display_piece = False):
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
    x_best     = 0
    y_best     = 0
    r_best     = 0
    best_score = -9000000  # Initialize with  low value

    # Calculate the total holes and total blocks above holes before play
    # total_holes, total_blocking_bocks, total_sum_heights
    init_move_info = calc_initial_move_info(board)
    # print(f"init_move_info : {init_move_info}")

    # Iterate through every possible rotation of the piece
    for r in range(len(PIECES[piece['shape']])):
        # Iterate through every possible position on the board
        for x in range(-2,BOARDWIDTH-2):
            # Calculate movement information for the current move (falling piece)
            # [True, max_height, num_removed_lines, new_holes, new_blocking_blocks, piece_sides, floor_sides, wall_sides]
            movement_info = calc_move_info(board, piece, x, r, \
                                                init_move_info[0], \
                                                init_move_info[1])

            # Check if it's a valid movement
            if (movement_info[0]):
                # Calculate movement score using the objective function
                movement_score = obj_function(init_move_info[2], init_move_info[0], movement_info[1], movement_info[3], movement_info[2], movement_info[-3], movement_info[-2], movement_info[-1] ,  score , chromo)

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
        piece['y'] = -2 # Move the piece out of the visible area

    # Set the best X and rotation for the piece
    piece['x'] = x_best
    piece['rotation'] = r_best

    # Return the best X coordinate and rotation
    return x_best, r_best
#***********************************************************
#***********************************************************
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
    draw_status(score, level) # Draw the score and level information
    draw_next_piece(next_piece) # Draw the next piece preview

    # If there is a falling piece, draw it
    if falling_piece != None:
        draw_piece(falling_piece)

    pygame.display.update() # Update the display
    FPSCLOCK.tick(FPS)      # Control the frame rate
#***********************************************************
#***********************************************************
def run_single_chromo(chromosome, max_score = 100000, show = True):
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
    num_piece = 600
    board            = get_blank_board()
    last_fall_time   = time.time()
    score            = 0
    level, fall_freq = calc_level_and_fall_freq(score)
    falling_piece    = get_new_piece()
    next_piece       = get_new_piece()

    # Calculate best move for the falling piece
    # x_best, r_best
    get_best_move(board, falling_piece, score ,chromosome)

    num_used_pieces = 0


    is_win   = False
    while True:
        # if (num_used_pieces >= 600) :
        #     break
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print ("Game exited by user")
                exit()

        if falling_piece == None:
            falling_piece = next_piece
            next_piece    = get_new_piece()

            # Calculate best move for the falling piece
            # x_best, r_best
            get_best_move(board, falling_piece, score ,chromosome)

            # Update number of used pieces and the score
            print(f"num_used_pieces : {num_used_pieces}")
            num_used_pieces += 1
            score           += 1

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
                if(num_removed_lines == 1):
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
            is_win   = True
            break

    # Save the game state
    game_state = [num_used_pieces, score, is_win]

    return game_state
#***********************************************************
#***********************************************************
def test_main():

    # chromo = [-71.1966, 65.7304, -22.4267, -93.6919, -3.1324, 49.4697, -37.7855, 40.6373, 53.1462]
    # run_single_chromo(chromo)
    chromo = [-71.1966, 65.7304, -22.4267, -93.6919, -3.1324, 49.4697, -37.7855, 40.6373, 53.1462]
    run_single_chromo(chromo)

test_main()