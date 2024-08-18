import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize

# Load data from Excel
file_path = '/home/ramya/seventy40ns.xlsx'
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
        return np.sum(((x - x0)**2 / a**2) + ((y - y0)**2 / b**2) + ((z - z0)**2 / c**2) - 1)**2

    # Initial guess for the ellipsoid parameters
    x_mean, y_mean, z_mean = np.mean(x), np.mean(y), np.mean(z)
    initial_guess = [np.std(x), np.std(y), np.std(z), x_mean, y_mean, z_mean]

    result = minimize(ellipsoid_error, initial_guess, method='L-BFGS-B', bounds=[(0, None)] * 6)
    return result.x

# Fit ellipsoid to data
params = fit_ellipsoid(x, y, z)
a, b, c, x0, y0, z0 = params
# Ensure that a, b, and c are in non-increasing order
a, b, c = sorted([a, b, c], reverse=True)

# Print the values of a, b, and c
print(f'a: {a}')
print(f'b: {b}')
print(f'c: {c}')

# Calculate eccentricity
eccentricity = np.sqrt(1 - (c**2 / a**2))
print(f'Eccentricity: {eccentricity:.3f}')

# Calculate volume of the ellipsoid
volume = (4/3) * np.pi * a * b * c
print(f'Volume of the ellipsoid: {volume:.3f}')

# Calculate surface area of the ellipsoid (using approximation)
p = 1.6075
surface_area = 4 * np.pi * ((a*b)**p + (b*c)**p + (c*a)**p)**(1/p)
print(f'Surface area of the ellipsoid (approx.): {surface_area:.3f}')

# Create ellipsoid points
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
u, v = np.meshgrid(u, v)
x_ellipsoid = a * np.cos(u) * np.sin(v) + x0
y_ellipsoid = b * np.sin(u) * np.sin(v) + y0
z_ellipsoid = c * np.cos(v) + z0

# Plot ellipsoid
ax.plot_wireframe(x_ellipsoid, y_ellipsoid, z_ellipsoid, color='green', alpha=0.5)

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
