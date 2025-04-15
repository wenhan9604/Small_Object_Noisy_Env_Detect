import cv2, os
import numpy as np

def generate(in_path, out_path):
    os.makedirs(out_path, exist_ok=True)
    print('applying haze to images')
    for filename in os.listdir(in_path):
        img_path = os.path.join(in_path, filename)
        image = cv2.imread(img_path)
        height, width, _ = image.shape
        image = cv2.resize(image,(width // 2,height // 2), interpolation=cv2.INTER_LINEAR)
        direction = np.random.choice(['top', 'bottom', 'none'])
        hazed_image = apply_haze(image=image, direction=direction)

        cv2.imwrite(os.path.join(out_path,filename), hazed_image)


def generate_haze_map(height, width, t_range=(0.2, 0.4), direction='top'):
    # Base linear gradient (stronger haze at top or bottom)
    if direction == 'top':
        base = np.linspace(t_range[0], t_range[1], height).reshape(-1, 1)
    elif direction == 'bottom':
        base = np.linspace(t_range[1], t_range[0], height).reshape(-1, 1)
    else:
        base = np.full((height, 1), np.mean(t_range))  # flat if unknown

    gradient = np.tile(base, (1, width)).astype(np.float32)

    # Add some smooth noise
    noise = np.random.normal(loc=0.0, scale=0.02, size=(height, width)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (101, 101), 0)

    # Combine gradient + noise and clip to range
    t_map = gradient + noise
    t_map = np.clip(t_map, t_range[0], t_range[1])
    return t_map


def apply_haze(image, A_range=(0.9, 1.0), t_range=(0.3, 0.5), direction='top'):
    height, width, _ = image.shape

    # --- Atmospheric Light with slight spatial variation (bluish fog) ---
    A = np.zeros_like(image).astype(np.float32)
    A[:, :, 0] = np.random.uniform(220, 255, size=(height, width))  # Blue
    A[:, :, 1] = np.random.uniform(220, 255, size=(height, width))  # Green
    A[:, :, 2] = np.random.uniform(220, 245, size=(height, width))  # Red

    # Optional: smooth the variations to make them more natural
    for i in range(3):
        A[:, :, i] = cv2.GaussianBlur(A[:, :, i], (101, 101), 0)

    # --- Transmission Map with gradient + noise ---
    t_map = generate_haze_map(height, width, t_range, direction)
    t_map_3ch = np.repeat(t_map[:, :, np.newaxis], 3, axis=2)

    # --- Blend image and haze ---
    hazed_image = image.astype(np.float32) * t_map_3ch + A * (1 - t_map_3ch)
    hazed_image = np.clip(hazed_image, 0, 255).astype(np.uint8)

    return hazed_image


if __name__ == "__main__":
    generate(
        in_path='./dataset/dota_train_part1',
        out_path='./dataset/dota_train_part1_hazed'
    )
    