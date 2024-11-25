import numpy as np
import matplotlib.pyplot as plt

def iterate(a, initial_y, iterations):
    y = initial_y
    for i in range(iterations):
        # Apply the first equation
        x = y/(a-a*y+y)
        print(f"Iteration: {x}")  # Output the result of the first equation
        
        # Now use the result as the new value of x
        y = x  

        # Apply the second equation (y = x, so needed y is also x)
        print(f"Next value of x: {y}")  # Output the next value

# Define the functions
def func1(x, a):
    return (a * x) / (1 + (a - 1) * x)


def func2(x):
    return x


# Example usage
a = 2.0826  # Replace with your desired value
initial_y = 0.912  # Starting value of x
iterations = 9  # Number of iterations

iterate(a, initial_y, iterations)
# Parameters
a = 2  # Replace with your desired value
x_values = np.linspace(-1, 3, 100)  # Values for x-axis

# Calculate y values for each function
y1_values = func1(x_values, a)
y2_values = func2(x_values)

plt.rcParams['font.sans-serif'] = ['SimHei']  # or any other font supporting Chinese
plt.rcParams['axes.unicode_minus'] = False  # Ensures that minus signs are displayed correctly

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x_values, y1_values, label=f'相平衡曲线', color='blue')
plt.plot(x_values, y2_values, label='操作线', color='red', linestyle='--')

plt.scatter([0.912, 0.318], [0.912, 0.318], color='red', zorder=5)  # Add green markers


# Formatting the plot
plt.title('乙醇正丙醇相平衡图')
plt.xlabel('x')
plt.ylabel('y')
plt.axhline(0, color='black',linewidth=0.5, ls='--')
plt.axvline(0, color='black',linewidth=0.5, ls='--')
plt.grid()
plt.legend()
plt.xlim(0, 1)
plt.ylim(0, 1)

# Show the plot
plt.show()
