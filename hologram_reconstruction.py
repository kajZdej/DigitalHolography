import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from PIL import Image
import scipy.io


# Function to load and normalize the hologram image
def load_hologram(image_path):
    # take just one color channel (e.g. red channel)
    img = Image.open(image_path)
    # convert to array
    img = np.array(img)

    return np.array(img) / 255.0  # Normalize to [0, 1]

def crop_center(img, cropx, cropy):
    y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]

def propIR(u1, pitch = 2.2e-06, wavelength = 6.328e-07, z = 0.17211, shiftx = 0, shifty = 0, signal = -1):
    """
    Impulse Response Reconstruction Method for holographic data.

    Parameters:
        u1 (ndarray): Hologram Plane Field (2D numpy array)
        pitch (float): Pixel Pitch (meters)
        wavelength (float): Wavelength (meters)
        z (float): Propagation Distance (meters)
        shiftx (float): Horizontal shift parameter (fraction of image width)
        shifty (float): Vertical shift parameter (fraction of image height)
        signal (int): +1 or -1, determines reconstruction type

    Returns:
        u2 (ndarray): Object Plane Field (2D numpy array)
    """
    # Constants and dimensions
    k = 2 * np.pi / wavelength  # Wave number
    M, N = u1.shape  # Dimensions of the hologram
    dx, dy = pitch, pitch  # Pixel pitch
    Lx, Ly = N * pitch, M * pitch  # Physical dimensions of the hologram
    
    # Generate x and y coordinates with shifts
    x = np.linspace(-Lx / 2 + shiftx * N * dx, Lx / 2 + shiftx * N * dx - dx, N)
    y = np.linspace(-Ly / 2 + shifty * M * dy, Ly / 2 + shifty * M * dy - dy, M)
    X, Y = np.meshgrid(x, y)  # Meshgrid for coordinates
    
    # Impulse response function
    h = (-signal * 1j) * np.exp((signal * 1j * k / (2 * z)) * (X**2 + Y**2))
    
    # Multiply hologram with the impulse response
    U2 = h * u1
    
    # Perform inverse FFT to reconstruct
    u2 = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(U2)))  # Centered IFFT
    u2 = np.abs(u2)  # Get the magnitude
    return u2


def propASM(u1, pitch, wavelength, z):
    """
    Angular Spectrum Method for propagating a wavefield.
    
    Parameters:
        u1 : ndarray
            Hologram plane field (input complex field)
        pitch : float
            Pixel pitch (distance between adjacent pixels)
        wavelength : float
            Wavelength of the light
        z : float
            Propagation distance (can be positive or negative)
    
    Returns:
        u2 : ndarray
            Object plane field (output complex field)
    """
    import numpy as np

    # Get dimensions of the input field
    M, N = u1.shape
    
    # Calculate spatial frequencies
    dx = pitch
    dy = pitch
    Lx = N * pitch
    Ly = M * pitch

    fx = np.fft.fftshift(np.fft.fftfreq(N, d=dx))  # freq coordinates (x)
    fy = np.fft.fftshift(np.fft.fftfreq(M, d=dy))  # freq coordinates (y)

    FX, FY = np.meshgrid(fx, fy)
    
    # Transfer function
    H = np.exp(2j * np.pi * z * np.sqrt((1 / wavelength)**2 - FX**2 - FY**2))
    H[np.isnan(H)] = 0  # Handle evanescent waves
    
    # Perform FFT-based propagation
    U1 = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(u1)))  # Forward FFT
    U2 = U1 * np.conj(H)  # Apply transfer function
    u2 = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(U2)))  # Inverse FFT

    return u2

# Update function for the slider
def update(val):
    z = z_slider.val
    reconstructed_image = propIR(input, z=z, pitch = 4.8e-06, wavelength = 4.73e-07)
    im.set_data(reconstructed_image)
    ax.set_title(f"Reconstructed Plane at z = {z:.3f} m")
    fig.canvas.draw_idle()
    
data = scipy.io.loadmat('Dice1_Hol_v2.mat')


"""input = data['u1']

slika = propIR(input, pitch = 2.2e-06, wavelength = 6.328e-07, z = 0.17211, shiftx = 0, shifty = 0, signal = -1)
plt.imshow(slika, cmap='gray')
plt.show()"""

"""# Load the hologram
image_path = "0_1_a_-5.png"  # Replace with your image filename
hologram = load_hologram(image_path)"""


# Initial parameters
z_min, z_max = -0.2, 0.2  # Reconstruction range in meters
z_init = 0.00741  # Initial reconstruction distance

input = load_hologram('deepDices2k-AP\deepDices2k_ampli.bmp')
input = input[:, :, 0]
print(input.shape)

# Set up the figure
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
reconstructed_image = propIR(input, pitch = 4.8e-06, wavelength = 6.4e-07, z = z_init, shiftx = 0, shifty = 0, signal = -1)

im = ax.imshow(reconstructed_image, cmap="gray", extent=(0, input.shape[1], 0, input.shape[0]))
ax.set_title(f"Reconstructed Plane at z = {z_init:.3f} m")
ax.set_xlabel("Pixels")
ax.set_ylabel("Pixels")

# Slider for reconstruction distance
ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor="lightgray")
z_slider = Slider(ax_slider, "z (m)", z_min, z_max, valinit=z_init)


z_slider.on_changed(update)

plt.show()  