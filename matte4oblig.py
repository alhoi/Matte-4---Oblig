

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Funksjon for numerisk løsning
def laplace_jacobi(Lx, Ly, Nx, Ny, V0, V1, num_iterations):
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)

    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)

    X, Y = np.meshgrid(x, y)

    V = np.zeros((Nx, Ny))
    V[0, :] = V0
    V[-1, :] = V0
    V[:, 0] = V1
    V[:, -1] = V1

    for _ in range(num_iterations):
        V[1:-1, 1:-1] = 0.25 * (V[:-2, 1:-1] + V[2:, 1:-1] + V[1:-1, :-2] + V[1:-1, 2:])

    return X, Y, V

# Parametere
Lx = 1.0
Ly = 1.0
Nx = 50
Ny = 50
V0 = 5.0   # Elektrisk potensial ved x = 0 og x = L_x
V1 = 0.0   # Elektrisk potensial ved y = 0 og y = L_y
num_iterations = 1000

# Bruker den definerte funksjonen til å løse Laplace-ligningen numerisk
X, Y, ElectricPotential = laplace_jacobi(Lx, Ly, Nx, Ny, V0, V1, num_iterations)

# Her lages plottet
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X-akse - Lederens lengde')
ax.set_ylabel('Y-akse - Lederens bredde')
ax.set_zlabel('Z-akse - Elektrisk potensial')
ax.set_title('Løsning av Laplace-ligningen')

# Funksjon for å oppdatere plottet i hver animasjonsframe
def update(frame):
    ax.clear()
    ax.set_xlabel('x - Lederens lengde')
    ax.set_ylabel('y - Lederens bredde')
    ax.set_zlabel('z - Elektrisk potensial')
    ax.set_title(f'Løsning av Laplace-ligningen')

    # Løs Laplace-ligningen numerisk med økt antall iterasjoner for hvert frame
    X, Y, ElectricPotential = laplace_jacobi(Lx, Ly, Nx, Ny, V0, V1, frame * 10)

    # Plot den oppdaterte løsningen
    ax.plot_surface(X, Y, ElectricPotential, cmap='viridis')

# Her opprettes animasjon
num_frames = 50
animation = FuncAnimation(fig, update, frames=num_frames, interval=200)

plt.show()