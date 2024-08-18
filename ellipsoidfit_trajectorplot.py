import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Load data from Excel
file_path = '/home/ramya/micelledimention/test_c.xlsx'
data = pd.read_excel(file_path)

# Print column names to check
print(data.columns)

# If necessary, strip any leading/trailing spaces
data.columns = data.columns.str.strip()

# Group the data by the correct frame column name
grouped = data.groupby('frame')

# Function to compute the ellipsoid parameters
def fit_ellipsoid(x, y, z):
    def ellipsoid_error(params):
        a, b, c, x0, y0, z0 = params
        return np.sum(((x - x0)**2 / a**2) + ((y - y0)**2 / b**2) + ((z - z0)**2 / c**2) - 1)**2

    # Initial guess for the ellipsoid parameters
    x_mean, y_mean, z_mean = np.mean(x), np.mean(y), np.mean(z)
    initial_guess = [np.std(x), np.std(y), np.std(z), x_mean, y_mean, z_mean]

    result = minimize(ellipsoid_error, initial_guess, method='L-BFGS-B', bounds=[(0, None)] * 6)
    return result.x

# Prepare lists to capture a, b, c values for all frames
a_values = []
b_values = []
c_values = []

# Group the data by the frame
grouped = data.groupby('frame')

# Iterate over each frame
for frame, group in grouped:
    # Extract x, y, z for the current frame
    x = group['x'].values
    y = group['y'].values
    z = group['z'].values
    
    # Fit ellipsoid to the current frame's data
    params = fit_ellipsoid(x, y, z)
    a, b, c, _, _, _ = params
    
    # Ensure that a, b, and c are in non-increasing order
    a, b, c = sorted([a, b, c], reverse=True)
    
    # Append the values to the lists
    a_values.append(a)
    b_values.append(b)
    c_values.append(c)

# Convert lists to numpy arrays for easier plotting
a_values = np.array(a_values)
b_values = np.array(b_values)
c_values = np.array(c_values)

# Plot the ellipsoid parameters over time
plt.figure(figsize=(10, 6))
plt.plot(a_values, label='a (Longest Axis)')
plt.plot(b_values, label='b (Middle Axis)')
plt.plot(c_values, label='c (Shortest Axis)')
plt.xlabel('Frame')
plt.ylabel('Ellipsoid Axis Length')
plt.title('Ellipsoid Parameters Over Time')
plt.legend()
plt.show()
