def knapsack_dp(values, weights, capacity):
    n = len(values)
    
    # Initialize the DP table with 0s
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

    # Populate the DP table
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w], values[i - 1] + dp[i - 1][w - weights[i - 1]])
            else:
                dp[i][w] = dp[i - 1][w]

    # Backtrack to find which items are included in the knapsack
    total_value = dp[n][capacity]
    w = capacity
    items_included = []
    
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            items_included.append(i - 1)  # Store index (i-1) of the item
            w -= weights[i - 1]

    # Display the DP table
    print("DP Table (Max Value Table):")
    for row in dp:
        print(" ".join(f"{x:2}" for x in row))

    # Display the items included and the total value
    print("Items Included in Knapsack (Item indices starting from 0):", items_included)
    print("Total Value in Knapsack:", total_value)

# Input
if __name__ == "__main__":
    num_items = int(input("Enter the number of items: "))
    capacity = int(input("Enter the capacity of the knapsack: "))
    
    values = []
    weights = []
    
    for i in range(num_items):
        weight = int(input(f"Enter the weight of item {i + 1}: "))
        value = int(input(f"Enter the value of item {i + 1}: "))
        weights.append(weight)
        values.append(value)

    # Solve knapsack problem using dynamic programming
    knapsack_dp(values, weights, capacity)
