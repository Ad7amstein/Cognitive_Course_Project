from game_ai.tetris_base import main
from game_ai.tetris_base import MANUAL_GAME
from game_ai.tetris_base import run_game
from game_ai.algorithm import run_game_ai

##############################################################################
# MAIN GAME
##############################################################################
def main():
    main()
    if (MANUAL_GAME):
        run_game()
    else:
        run_game_ai()
    

if __name__ == "__main__":
    main()