import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_image(image_path):
    """Load an image as grayscale."""
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

def estimate_wavelength(hologram, pixel_pitch):
    """
    Estimate the wavelength of light from fringe spacing.
    :param hologram: 2D hologram image.
    :param pixel_pitch: Pixel size in meters.
    :return: Estimated wavelength in meters.
    """
    # Fourier transform of the hologram
    fft_holo = np.fft.fftshift(np.fft.fft2(hologram))
    magnitude_spectrum = np.abs(fft_holo)
    
    # Detect the dominant fringe frequency
    y_center, x_center = np.array(magnitude_spectrum.shape) // 2
    radial_profile = magnitude_spectrum[y_center, :]  # Take horizontal profile
    peak_freq = np.argmax(radial_profile[x_center:]) + x_center
    
    # Convert peak frequency to spatial frequency
    spatial_freq = peak_freq / (hologram.shape[1] * pixel_pitch)
    
    # Estimate wavelength
    wavelength = 1 / spatial_freq
    return wavelength

def estimate_distance(hologram, brightfield, wavelength, pixel_pitch, z_range):
    """
    Estimate object-to-sensor distance by matching reconstructed phase.
    :param hologram: 2D hologram image.
    :param brightfield: Brightfield image of the same object.
    :param wavelength: Wavelength of light (meters).
    :param pixel_pitch: Pixel size in meters.
    :param z_range: Range of distances to search (meters).
    :return: Best distance (meters).
    """
    def reconstruct(hologram, z):
        """Reconstruct the hologram at distance z."""
        k = 2 * np.pi / wavelength
        ny, nx = hologram.shape
        fx = np.fft.fftfreq(nx, d=pixel_pitch)
        fy = np.fft.fftfreq(ny, d=pixel_pitch)
        fx, fy = np.meshgrid(fx, fy)
        fsq = fx**2 + fy**2
        # Clip values to avoid negative values inside the sqrt
        sqrt_term = np.sqrt(np.clip(1 - (wavelength**2) * fsq, 0, None))
        H = np.exp(1j * k * z * sqrt_term)
        fft_holo = np.fft.fft2(hologram)
        return np.fft.ifft2(fft_holo * H)

    best_distance = None
    min_error = float("inf")
    
    for z in z_range:
        reconstructed = np.abs(reconstruct(hologram, z))
        error = np.mean((reconstructed - brightfield) ** 2)  # Mean squared error
        if error < min_error:
            min_error = error
            best_distance = z
    
    return best_distance

def main():
    # Load images
    hologram_path = "/mnt/data/home/antonn/SEMINAR/DigitalHolography/images/imageh.png"
    brightfield_path = "/mnt/data/home/antonn/SEMINAR/DigitalHolography/images/imageb.png"
    hologram = load_image(hologram_path)
    brightfield = load_image(brightfield_path)

    # Define parameters
    pixel_pitch = 6.45e-6  # Pixel size in meters

    # Step 1: Estimate Wavelength
    estimated_wavelength = estimate_wavelength(hologram, pixel_pitch)
    print(f"Estimated Wavelength: {estimated_wavelength:.2e} meters")

    # Step 2: Estimate Distance
    z_range = np.linspace(0.001, 0.05, 100)  # Search range for z (1 mm to 50 mm)
    estimated_distance = estimate_distance(hologram, brightfield, estimated_wavelength, pixel_pitch, z_range)
    print(estimated_distance)
    #print(f"Estimated Distance: {estimated_distance:.2e} meters")

if __name__ == "__main__":
    main()
