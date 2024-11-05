def print_board(board):
    for row in board:
        print(" ".join('Q' if x == 1 else '.' for x in row))
    print("\n")  # Add a newline for better readability

def is_safe(board, row, col):
    # Check this column on the upper side
    for i in range(row):
        if board[i][col] == 1:
            return False
    # Check the upper diagonal on the left side
    for i, j in zip(range(row - 1, -1, -1), range(col - 1, -1, -1)):
        if board[i][j] == 1:
            return False
    # Check the upper diagonal on the right side
    for i, j in zip(range(row - 1, -1, -1), range(col + 1, 8)):
        if board[i][j] == 1:
            return False
    return True

def solve(board, row):
    if row == 8:
        return True
    for col in range(8):
        if is_safe(board, row, col):
            board[row][col] = 1
            print(f"Placing queen at position ({row}, {col}):")
            print_board(board)  # Print the board after placing the queen
            if solve(board, row + 1):
                return True
            board[row][col] = 0  # Backtrack
            print(f"Removing queen from position ({row}, {col}):")
            print_board(board)  # Print the board after removing the queen
    return False

def main():
    board = [[0 for _ in range(8)] for _ in range(8)]
    
    # The first queen's position can be hardcoded for demonstration.
    # You can modify this part to specify the starting position directly if needed.
    board[0][0] = 1  # Example: placing the first queen at (0, 0)
    
    # Start solving from the next row after the first queen's row
    if not solve(board, 1):
        print("\nNo solution exists for the given starting position.")

if __name__ == "__main__":
    main()
