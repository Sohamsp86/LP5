
def print_board(board):
    for row in board:
        print(" ".join('Q' if x == 1 else '.' for x in row))

def is_safe(board, row, col):
    for i in range(row):
        if board[i][col] == 1:
            return False
    for i, j in zip(range(row-1, -1, -1), range(col-1, -1, -1)):
        if board[i][j] == 1:
            return False
    for i, j in zip(range(row-1, -1, -1), range(col+1, 8)):
        if board[i][j] == 1:
            return False
    return True

def solve(board, row):
    if row == 8:
        print_board(board)
        return True
    for col in range(8):
        if is_safe(board, row, col):
            board[row][col] = 1
            if solve(board, row + 1):
                return True
            board[row][col] = 0
    return False

def main():
    board = [[0 for _ in range(8)] for _ in range(8)]
    first_row = int(input("Enter the row (0-7) for the first queen: "))
    first_col = int(input("Enter the column (0-7) for the first queen: "))
    board[first_row][first_col] = 1
    print("\nPlacing the first queen at position ({},{})".format(first_row, first_col))
    print("The initial board setup is:")
    print_board(board)
    print("\n")
    if not solve(board, 0):
        print("\nNo solution exists for the given starting position.")

if __name__ == "__main__":
    main()
