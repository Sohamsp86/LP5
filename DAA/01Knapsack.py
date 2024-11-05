def knapsack(values, weights, capacity):
    n = len(values)
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

    # Build the DP table
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w], values[i - 1] + dp[i - 1][w - weights[i - 1]])
            else:
                dp[i][w] = dp[i - 1][w]

    return dp[n][capacity]

# Example usage
if __name__ == "__main__":
    # Define values and weights of items
    values = [60, 100, 120]
    weights = [10, 20, 30]
    knapsack_capacity = 50

    max_value = knapsack(values, weights, knapsack_capacity)
    print(f"Maximum value in knapsack: {max_value}")
