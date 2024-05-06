import matplotlib.pyplot as plt

def plot_lists(list1, list2):
    # Plotting the lists
    plt.plot(list1, label='List 1')
    plt.plot(list2, label='List 2')

    # Adding labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Plot of Two Lists')

    # Adding legend
    plt.legend()

    # Showing the plot
    plt.show()

# Example lists
list1 = [1, 2, 3, 4, 5]
list2 = [5, 4, 3, 2, 1]

# Call the function
plot_lists(list1, list2)
