import matplotlib.pyplot as plt

# Epochs
epochs = list(range(1, 31))

# Losses for KJRD-Net (First Run)
losses_kjrd = [
    640.2638, 600.7680, 573.0477, 573.2656, 575.4436, 573.7127, 580.3729, 573.8301, 583.8171, 572.1400,
    578.6041, 572.2867, 573.0339, 578.3259, 574.4535, 571.3945, 573.2612, 578.8617, 579.3572, 579.8905,
    585.1076, 588.3886, 574.1136, 582.6640, 568.4469, 574.4657, 569.3032, 570.8053, 576.7416, 579.6465
]

# Losses for Diffusion-based KJRD-Net (Second Run)
losses_diffusion = [
    620.3514, 577.3925, 582.5332, 577.0408, 578.1359, 576.3933, 578.5372, 578.0438, 575.0026, 576.4584,
    581.0842, 579.0326, 576.0211, 574.8461, 577.5991, 573.0597, 574.0848, 573.0062, 569.2625, 573.0711,
    571.2607, 572.2465, 576.5461, 568.8587, 572.5080, 579.7039, 568.9212, 570.5675, 569.0497, 568.0357
]

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(epochs, losses_kjrd, marker='o', linestyle='-', color='blue', label='KJRD-Net')
plt.plot(epochs, losses_diffusion, marker='o', linestyle='-', color='green', label='Diffusion-based KJRD-Net')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.xticks(epochs)
plt.legend()
plt.tight_layout()
plt.show()
