import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


# Sidebar inputs for interactive parameters
power_mW = 1020  # Typical LED power in mW

theta = st.sidebar.number_input('Semi-angle at half power (theta)', min_value=0, max_value=90, value=45)
efficacy_lm_per_W = st.sidebar.number_input('Luminous efficay of LED (lm/W)', min_value=200, max_value=1000, value=683)  # in lumens per watt
h = st.sidebar.number_input('Height of LEDs from receiver plane (cm)', min_value=1, max_value=100, value=10)
lx = st.sidebar.number_input('Detector width (cm)', min_value=1, max_value=100, value=10)
ly = st.sidebar.number_input('Detector length (cm)', min_value=1, max_value=100, value=10)
R1 = st.sidebar.number_input('Radius of LED circle (cm)', min_value=1, max_value=50, value=10)
num_LEDs = st.sidebar.number_input('Number of LEDs on the circle', min_value=1, max_value=50, value=1)
beta = st.sidebar.number_input('Tilt angle (beta) of LEDs toward center (degrees)', min_value=0, max_value=90, value=0)


# Function to calculate I_0
def calculate_I_0(power_mW, efficacy_lm_per_W, theta):
    """Calculate the center luminous intensity I_0 in candela (cd)."""
    luminous_flux_lm = (power_mW / 1000) * efficacy_lm_per_W  # Convert radiant power to luminous flux (in lumens)
    solid_angle = 2 * np.pi * (1 - np.cos(np.radians(theta)))  # Approximation for solid angle in steradians
    I_0 = luminous_flux_lm / solid_angle  # center Luminous intensity formula
    return I_0

# Function to calculate total illuminance E_lux_total
def calculate_E_lux_total(I_0, theta, h, beta, R1, lx, ly, num_LEDs):
    """Calculate the total illuminance on the detector plane from multiple LEDs."""
    
    beta_rad = np.radians(beta)
    m = -np.log10(2) / np.log10(np.cos(np.radians(theta)))  # Lambertian order of emission
    
    # Room grid
    Nx = lx * 25  # number of grid points along x-axis
    Ny = ly * 25  # number of grid points along y-axis
    x = np.linspace(-lx / 2, lx / 2, int(Nx))
    y = np.linspace(-ly / 2, ly / 2, int(Ny))
    XR, YR = np.meshgrid(x, y)  # receiver plane grid

    # Initialize total received power to zero
    E_lux_total = np.zeros_like(XR)

    # Calculate alpha angles for LEDs
    alphas = np.linspace(0, 360, num_LEDs, endpoint=False)
    alphas_rad = np.radians(alphas)

    for alpha_rad in alphas_rad:
        # LED position on a circle of radius R1
        x_led = R1 * np.cos(alpha_rad)
        y_led = R1 * np.sin(alpha_rad)

        # Distance from LED to each point on the detector plane
        tran_distanc_rec = np.sqrt((XR - x_led)**2 + (YR - y_led)**2 + h**2)

        # Direction from LED to center of detector (0,0)
        dir_to_center_x = -x_led / np.sqrt(x_led**2 + y_led**2)
        dir_to_center_y = -y_led / np.sqrt(x_led**2 + y_led**2)

        # Cosine of irradiance angle
        cos_phi = ((h / tran_distanc_rec) * np.cos(beta_rad)) + (
                    (dir_to_center_x * (XR - x_led) + dir_to_center_y * (YR - y_led)) / tran_distanc_rec * np.sin(beta_rad))

        # Ensure cos_phi values are within [-1,1]
        cos_phi = np.clip(cos_phi, -1, 1)

        # Received power from this LED
        E_lux = (I_0 * cos_phi ** m) / (tran_distanc_rec ** 2)

        # Accumulate the received power
        E_lux_total += E_lux
    
    return XR, YR, E_lux_total

# Calculate I_0 using the function
I_0 = calculate_I_0(power_mW, efficacy_lm_per_W, theta)

# Calculate E_lux_total using the function
XR, YR, E_lux_total = calculate_E_lux_total(I_0, theta, h, beta, R1, lx, ly, num_LEDs)

# Plotting
fig = plt.figure(figsize=(18, 4.5))

# 2D subplot
ax2 = fig.add_subplot(132)
contour = ax2.contourf(XR, YR, E_lux_total, 50, cmap='viridis')
fig.colorbar(contour, ax=ax2, fraction=0.046, pad=0.04)
ax2.set_xlabel('X (cm)')
ax2.set_ylabel('Y (cm)')
ax2.set_title('Distribution of illuminance (lx) in 2D view')
ax2.set_aspect('equal')
# Set scale according to lx and ly
ax2.set_xlim([-lx / 2, lx / 2])
ax2.set_ylim([-ly / 2, ly / 2])
ax2.set_aspect('equal')  # Maintain equal aspect ratio

# Show the plots in Streamlit
st.pyplot(fig)