import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import rotate, resize

# ----------------------------
# 1. Radon Transform
# ----------------------------
def radon_transform(image, steps):
    """Compute sinogram of the image."""
    n = image.shape[0]
    sinogram = np.zeros((n, steps), dtype=np.float64)
    for s in range(steps):
        rotated = rotate(image, -s * 180 / steps, resize=False, order=1, mode='constant')
        sinogram[:, s] = np.sum(rotated, axis=0)
    return sinogram

# ----------------------------
# 2. Pad Sinogram to Square
# ----------------------------
def sinogram_circle_to_square(sinogram):
    diagonal = int(np.ceil(np.sqrt(2) * sinogram.shape[0]))
    pad = diagonal - sinogram.shape[0]
    old_center = sinogram.shape[0] // 2
    new_center = diagonal // 2
    pad_before = new_center - old_center
    pad_width = ((pad_before, pad - pad_before), (0, 0))
    return np.pad(sinogram, pad_width, mode='constant', constant_values=0)


###Filters
def ram_lak_filter(freqs):
    return 2 * np.abs(freqs)

def shepp_logan_filter(freqs):
    ramp = 2 * np.abs(freqs)
    f = np.abs(freqs)
    with np.errstate(divide='ignore', invalid='ignore'):
        return ramp * np.sinc(f / 2)

def cosine_filter(freqs):
    ramp = 2 * np.abs(freqs)
    f = np.abs(freqs)
    f_max = 0.5
    window = np.cos(np.pi * f / (2 * f_max))
    window[f > f_max] = 0
    return ramp * window

def hamming_filter(freqs):
    ramp = 2 * np.abs(freqs)
    f = np.abs(freqs)
    f_max = 0.5
    window = 0.54 + 0.46 * np.cos(np.pi * f / f_max)
    window[f > f_max] = 0
    return ramp * window

def hann_filter(freqs):
    ramp = 2 * np.abs(freqs)
    f = np.abs(freqs)
    f_max = 0.5
    window = 0.5 + 0.5 * np.cos(np.pi * f / f_max)
    window[f > f_max] = 0
    return ramp * window


# ----------------------------
# 3. Filtered Backprojection
# ----------------------------
def iradon_transform(radon_image, theta, filter_type='ram-lak', interpolation='linear'):
    output_size = radon_image.shape[0]
    radon_image = sinogram_circle_to_square(radon_image)
    th = np.deg2rad(theta)  # convert to radians

    # pad to next power of 2 for FFT
    proj_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * radon_image.shape[0]))))
    pad_width = ((0, proj_size_padded - radon_image.shape[0]), (0, 0))
    img = np.pad(radon_image, pad_width, mode='constant', constant_values=0)

    # Create frequency axis 
    freqs = np.fft.fftfreq(proj_size_padded).reshape(-1, 1)

    # Select filter based on filter_type
    if filter_type == 'ram-lak':
        filt = ram_lak_filter(freqs)
    elif filter_type == 'shepp-logan':
        filt = shepp_logan_filter(freqs)
    elif filter_type == 'cosine':
        filt = cosine_filter(freqs)
    elif filter_type == 'hamming':
        filt = hamming_filter(freqs)
    elif filter_type == 'hann':
        filt = hann_filter(freqs)
    else:
        filt = ram_lak_filter(freqs)   # default

    # Apply filter in Fourier domain
    projection_fft = np.fft.fft(img, axis=0) * filt
    radon_filtered = np.real(np.fft.ifft(projection_fft, axis=0))
    radon_filtered = radon_filtered[:radon_image.shape[0], :]

    # Backprojection 
    reconstructed = np.zeros((output_size, output_size), dtype=np.float64)
    X, Y = np.mgrid[0:output_size, 0:output_size]
    xpr = X - output_size // 2
    ypr = Y - output_size // 2
    mid_index = radon_image.shape[0] // 2

    for i in range(len(theta)):
        t = ypr * np.cos(th[i]) - xpr * np.sin(th[i])
        x = np.arange(radon_filtered.shape[0]) - mid_index

        if interpolation == 'linear':
            backproj = np.interp(t, x, radon_filtered[:, i], left=0, right=0)
        else:
            from scipy.interpolate import interp1d
            interpolant = interp1d(x, radon_filtered[:, i], kind=interpolation,
                                   bounds_error=False, fill_value=0)
            backproj = interpolant(t)
        reconstructed += backproj

    # Mask outside circle
    radius = output_size // 2
    mask = (xpr**2 + ypr**2) <= radius**2
    reconstructed[~mask] = 0

    # Normalize
    return reconstructed * np.pi / (2 * len(th))

# ----------------------------
# 4. Main
# ----------------------------
# Load image
imagename = r"sagittalslice.jpg"  # replace with your image path
image = imread(imagename)
if image.ndim == 3:
    image = rgb2gray(image)
image = resize(image, (220, 220), anti_aliasing=True)

# Radon Transform
steps = 220
sinogram = radon_transform(image, steps)


# Theta angles
theta = np.linspace(0., 180., steps, endpoint=False)

##FIlters
filters_to_test = ['ram-lak', 'shepp-logan', 'cosine', 'hamming', 'hann']
reconstructions = {}

for f in filters_to_test:
    recon = iradon_transform(sinogram, theta, filter_type=f, interpolation='cubic')
    reconstructions[f] = recon



# ----------------------------
# 5. Plot results
# ----------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.set_title("Original Image")
ax1.imshow(image, cmap='gray')
ax1.axis('off')

ax2.set_title("Sinogram")
ax2.imshow(sinogram, cmap='gray', aspect='auto',
           extent=(0, 180, 0, sinogram.shape[0]))
ax2.set_xlabel("Projection angle (deg)")
ax2.set_ylabel("Projection position (pixels)")

plt.show()
# Reconstruction error
error = reconstructions['ram-lak'] - image
# Reconstruction error
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.set_title("FBP Reconstruction")
ax1.imshow(reconstructions['ram-lak'], cmap='gray')
ax1.axis('off')

ax2.set_title("Reconstruction Error")
imkwargs = dict(vmin=-0.2, vmax=0.2)
ax2.imshow(error, cmap='gray', **imkwargs)
plt.show()


# Plot image, reconstructions, and difference maps
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
filters = list(reconstructions.keys())
axes = axes.ravel()

# Original
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original')
axes[0].axis('off')

# Reconstructions
for i, f in enumerate(filters, start=1):
    axes[i].imshow(reconstructions[f], cmap='gray')
    axes[i].set_title(f)
    axes[i].axis('off')




other_filters = [f for f in filters_to_test if f != 'ram-lak']


#Original minus Filtered
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()
v_range = 0.15 

for i, f in enumerate(filters_to_test):
    # Compute difference: Original - Reconstruction
    diff = image - reconstructions[f]
    
    im = axes[i].imshow(diff, cmap='gray', vmin=-v_range, vmax=v_range)
    axes[i].set_title(f'Original minus {f.capitalize()}')
    axes[i].axis('off')
    
    # Add colorbars to interpret the error magnitude
    plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

# Hide the last subplot 
if len(filters_to_test) < 6:
    axes[5].axis('off')

plt.suptitle('Error Maps: Original Image minus Reconstructions', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for i, f in enumerate(other_filters):
    # Compute difference: Ram-Lak minus the current filter
    diff = reconstructions['ram-lak'] - reconstructions[f]
    
    # Use gray map
    im = axes[i].imshow(diff, cmap='gray', vmin=-0.05, vmax=0.05)
    axes[i].set_title(f'Ram-Lak minus {f.capitalize()}')
    axes[i].axis('off')
    
    fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

plt.suptitle('Difference Maps: Visualizing Filtering Effects', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# plot intensity profiles along a row for original and reconstructions
row_idx = 110  # choose a row with sharp features
plt.figure(figsize=(10, 6))
plt.plot(image[row_idx, :], 'k-', linewidth=2, label='Original')
for f in filters:
    plt.plot(reconstructions[f][row_idx, :], '--', label=f)
plt.xlabel('Pixel column')
plt.ylabel('Intensity')
plt.legend()
plt.title('Horizontal profile comparison')
plt.grid(True)
plt.show()

# Error metrics
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def compute_metrics(original, reconstructed):
    rmse = np.sqrt(np.mean((original - reconstructed)**2))
    psnr = peak_signal_noise_ratio(original, reconstructed, data_range=1.0)
    ssim = structural_similarity(original, reconstructed, data_range=1.0)
    return rmse, psnr, ssim

print(f"{'Filter':<12} {'RMSE':<8} {'PSNR (dB)':<10} {'SSIM':<6}")
print("-" * 40)
for f in filters:
    rmse, psnr, ssim = compute_metrics(image, reconstructions[f])
    print(f"{f:<12} {rmse:.4f}   {psnr:.2f}      {ssim:.4f}")
