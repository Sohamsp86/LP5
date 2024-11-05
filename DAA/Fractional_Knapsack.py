class Item:
    def __init__(self, value, weight, id):
        self.value = value
        self.weight = weight
        self.id = id
        self.value_per_weight = value / weight

def fractional_knapsack(capacity, items):
    # Sort items by value per weight in descending order
    items.sort(key=lambda x: x.value_per_weight, reverse=True)
    
    total_value = 0.0
    knapsack_contents = []

    for item in items:
        if capacity >= item.weight:
            # Take the whole item
            capacity -= item.weight
            total_value += item.value
            knapsack_contents.append((item.id, item.value, item.weight, 1))
        else:
            # Take a fraction of the item
            fraction = capacity / item.weight
            total_value += item.value * fraction
            knapsack_contents.append((item.id, item.value * fraction, capacity, fraction))
            capacity = 0  # Knapsack is full
            break

    return total_value, knapsack_contents

# Input and output formatting
def main():
    n = int(input("Enter the number of items: "))
    capacity = float(input("Enter the capacity of the knapsack: "))
    
    items = []
    for i in range(1, n + 1):
        weight = float(input(f"Enter the weight of item {i}: "))
        value = float(input(f"Enter the value of item {i}: "))
        items.append(Item(value, weight, i))
    
    total_value, knapsack_contents = fractional_knapsack(capacity, items)
    
    # Display the results
    print("Item ID | Value | Weight | Fraction of Item Taken")
    print("--------------------------------------------------")
    for id, value, weight, fraction in knapsack_contents:
        print(f"{id} | {value:.1f} | {weight:.1f} | {fraction}")
    
    print(f"Total Value in Knapsack: {total_value}")

# Run the main function
main()
