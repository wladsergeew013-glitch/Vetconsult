import curses
import random
import time

WIDTH = 10
HEIGHT = 20

# Tetromino shapes represented as lists of lists
SHAPES = [
    [[1, 1, 1, 1]],                            # I
    [[1, 1],
     [1, 1]],                                   # O
    [[0, 1, 0],
     [1, 1, 1]],                                # T
    [[1, 0, 0],
     [1, 1, 1]],                                # J
    [[0, 0, 1],
     [1, 1, 1]],                                # L
    [[1, 1, 0],
     [0, 1, 1]],                                # S
    [[0, 1, 1],
     [1, 1, 0]]                                 # Z
]

COLORS = [1, 2, 3, 4, 5, 6, 7]


def rotate(shape):
    """Rotate the shape clockwise."""
    return [list(row) for row in zip(*shape[::-1])]


def create_board():
    return [[0] * WIDTH for _ in range(HEIGHT)]


def check_collision(board, shape, offset):
    off_y, off_x = offset
    for y, row in enumerate(shape):
        for x, cell in enumerate(row):
            if cell:
                new_x = x + off_x
                new_y = y + off_y
                if new_x < 0 or new_x >= WIDTH or new_y >= HEIGHT:
                    return True
                if new_y >= 0 and board[new_y][new_x]:
                    return True
    return False


def merge_piece(board, shape, offset, color):
    off_y, off_x = offset
    for y, row in enumerate(shape):
        for x, cell in enumerate(row):
            if cell and 0 <= y + off_y < HEIGHT and 0 <= x + off_x < WIDTH:
                board[y + off_y][x + off_x] = color


def clear_lines(board):
    new_board = [row for row in board if any(cell == 0 for cell in row)]
    cleared = HEIGHT - len(new_board)
    for _ in range(cleared):
        new_board.insert(0, [0] * WIDTH)
    return new_board, cleared


def draw_board(stdscr, board, score, level):
    stdscr.clear()
    for y, row in enumerate(board):
        for x, cell in enumerate(row):
            if cell:
                stdscr.addstr(y + 1, x * 2 + 1, "[]", curses.color_pair(cell))
            else:
                stdscr.addstr(y + 1, x * 2 + 1, "  ")
    stdscr.addstr(0, 0, f"Score: {score} Level: {level}")
    stdscr.refresh()


def game(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(100)

    for i in range(1, 8):
        curses.init_pair(i, i, 0)

    board = create_board()
    current_shape = random.choice(SHAPES)
    current_color = random.choice(COLORS)
    offset = [0, WIDTH // 2 - len(current_shape[0]) // 2]
    score = 0
    level = 1
    drop_speed = 0.5
    last_drop = time.time()

    while True:
        key = stdscr.getch()
        if key == ord('q'):
            break
        elif key == curses.KEY_LEFT:
            new_offset = [offset[0], offset[1] - 1]
            if not check_collision(board, current_shape, new_offset):
                offset = new_offset
        elif key == curses.KEY_RIGHT:
            new_offset = [offset[0], offset[1] + 1]
            if not check_collision(board, current_shape, new_offset):
                offset = new_offset
        elif key == curses.KEY_DOWN:
            new_offset = [offset[0] + 1, offset[1]]
            if not check_collision(board, current_shape, new_offset):
                offset = new_offset
        elif key == curses.KEY_UP:
            rotated = rotate(current_shape)
            if not check_collision(board, rotated, offset):
                current_shape = rotated

        if time.time() - last_drop > drop_speed:
            new_offset = [offset[0] + 1, offset[1]]
            if not check_collision(board, current_shape, new_offset):
                offset = new_offset
            else:
                merge_piece(board, current_shape, offset, current_color)
                board, cleared = clear_lines(board)
                score += cleared * 100
                level = score // 1000 + 1
                drop_speed = max(0.1, 0.5 - (level - 1) * 0.05)
                current_shape = random.choice(SHAPES)
                current_color = random.choice(COLORS)
                offset = [0, WIDTH // 2 - len(current_shape[0]) // 2]
                if check_collision(board, current_shape, offset):
                    break
            last_drop = time.time()

        temp_board = [row[:] for row in board]
        merge_piece(temp_board, current_shape, offset, current_color)
        draw_board(stdscr, temp_board, score, level)

    stdscr.nodelay(False)
    stdscr.addstr(HEIGHT // 2, WIDTH - 4, "Game Over")
    stdscr.addstr(HEIGHT // 2 + 1, WIDTH - 6, "Press any key")
    stdscr.getch()


if __name__ == "__main__":
    curses.wrapper(game)
