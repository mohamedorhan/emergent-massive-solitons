import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq

# ==============================
# Physical & Numerical Parameters
# ==============================
Nx = 2048
Lx = 200.0
dx = Lx / Nx

dt = 0.01
Nt = 12000

c = 1.0        # signal speed
lam = 1.0      # lambda
v = 1.0        # vacuum expectation value

m = np.sqrt(2 * lam * v**2)   # analytic mass

x = np.linspace(-Lx/2, Lx/2, Nx)

# ==============================
# Analytic Kink Initial Condition
# ==============================
phi = v * np.tanh(m * x / (np.sqrt(2) * c))
phi_t = np.zeros_like(phi)

# ==============================
# Periodic Laplacian
# ==============================
def laplacian(f):
    return (np.roll(f,1) - 2*f + np.roll(f,-1)) / dx**2

# ==============================
# Energy Functional
# ==============================
def energy(phi, phi_t):
    grad = (np.roll(phi,-1) - np.roll(phi,1)) / (2*dx)
    U = 0.25 * lam * (phi**2 - v**2)**2
    density = 0.5 * phi_t**2 + 0.5 * c**2 * grad**2 + U
    return np.sum(density) * dx

E0 = energy(phi, phi_t)

# ==============================
# Time Evolution (Leapfrog)
# ==============================
energy_log = []

for n in range(Nt):
    acc = c**2 * laplacian(phi) - lam * phi * (phi**2 - v**2)
    phi_t += dt * acc
    phi += dt * phi_t
    
    if n % 20 == 0:
        energy_log.append(energy(phi, phi_t))

# ==============================
# Linear Perturbation for Dispersion
# ==============================
phi += 0.01 * np.random.randn(Nx)

sig = phi.copy()
spec = np.abs(fft(sig))**2
k = fftfreq(Nx, dx)

# ==============================
# OUTPUT FIGURES
# ==============================
plt.figure(figsize=(14,4))

plt.subplot(1,3,1)
plt.plot(x, phi)
plt.title("Relativistic ϕ⁴ Kink Solution")
plt.xlabel("x")
plt.ylabel("ϕ")

plt.subplot(1,3,2)
plt.plot(energy_log)
plt.title("Energy Conservation")
plt.xlabel("time step")
plt.ylabel("E")

plt.subplot(1,3,3)
plt.plot(k[k>0], spec[k>0])
plt.title("Dispersion Spectrum |ϕ(k)|²")
plt.xlabel("k")
plt.ylabel("Power")
plt.xlim(0,1)

plt.tight_layout()
plt.show()

print("\n======================================")
print("ANALYTIC RESULT:")
print("m² = 2 λ v² =", 2*lam*v**2)
print("m  =", m)
print("======================================")
print("NUMERICAL:")
print("Energy drift =", abs(energy_log[-1]-energy_log[0])/energy_log[0])
print("======================================")
