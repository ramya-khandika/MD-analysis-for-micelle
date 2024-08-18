import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize

# Load data from Excel
file_path = '/home/ramya/seventy.xlsx'
data = pd.read_excel(file_path)
x = data['x'].values
y = data['y'].values
z = data['z'].values

# 3D scatter plot of the original points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, color='blue', label='Original Points')

# Function to compute the ellipsoid parameters
def fit_ellipsoid(x, y, z):
    def ellipsoid_error(params):
        a, b, c, x0, y0, z0 = params
        return np.sum(np.abs(((x - x0)**2 / a**2) + ((y - y0)**2 / b**2) + ((z - z0)**2 / c**2) - 1))

    # Initial guess for the ellipsoid parameters
    x_mean, y_mean, z_mean = np.mean(x), np.mean(y), np.mean(z)
    initial_guess = [np.std(x), np.std(y), np.std(z), x_mean, y_mean, z_mean]

    result = minimize(ellipsoid_error, initial_guess, method='L-BFGS-B', bounds=[(0.1, None)]*3 + [(None, None)]*3)
    return result.x

# Fit ellipsoid to data
params = fit_ellipsoid(x, y, z)
a, b, c, x0, y0, z0 = params

# Create ellipsoid points
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
u, v = np.meshgrid(u, v)
x_ellipsoid = a * np.cos(u) * np.sin(v) + x0
y_ellipsoid = b * np.sin(u) * np.sin(v) + y0
z_ellipsoid = c * np.cos(v) + z0

# Plot ellipsoid
ax.plot_wireframe(x_ellipsoid, y_ellipsoid, z_ellipsoid, color='green', alpha=0.5)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('3D Ellipsoid Fit to Data')

plt.legend()
plt.show()

def calculate_residuals(x, y, z, a, b, c, x0, y0, z0):
    return ((x - x0)**2 / a**2) + ((y - y0)**2 / b**2) + ((z - z0)**2 / c**2) - 1

# Calculate residuals
residuals = calculate_residuals(x, y, z, a, b, c, x0, y0, z0)

# Mean Squared Error
mse = np.mean(residuals**2)

# Root Mean Squared Error
rmse = np.sqrt(mse)

# Print errors
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')

# Check if each point satisfies the ellipsoid equation
def check_points(x, y, z, a, b, c, x0, y0, z0):
    for i in range(len(x)):
        value = (x[i] - x0)**2 / a**2 + (y[i] - y0)**2 / b**2 + (z[i] - z0)**2 / c**2
        if np.isclose(value, 1, atol=1e-2):  # Adjust the tolerance as needed
            print(f'Point ({x[i]}, {y[i]}, {z[i]}) satisfies the ellipsoid equation (value = {value:.2f})')
        else:
            print(f'Point ({x[i]}, {y[i]}, {z[i]}) does NOT satisfy the ellipsoid equation (value = {value:.2f})')

# Check and print points
check_points(x, y, z, a, b, c, x0, y0, z0)

# Plot residuals
fig, ax = plt.subplots()
ax.hist(residuals, bins=50, color='blue', alpha=0.7)
ax.set_title('Residuals Histogram')
ax.set_xlabel('Residual Value')
ax.set_ylabel('Frequency')
plt.show()

# Scatter plot of residuals
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(x, y, z, c=residuals, cmap='viridis')
fig.colorbar(scatter, ax=ax, label='Residual Value')
ax.set_title('3D Scatter Plot with Residuals')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()
