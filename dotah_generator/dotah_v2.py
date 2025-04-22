import cv2, os
import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

def process(filename, in_path, out_path, subfolder):
    img_path=os.path.join(in_path,subfolder,"images",filename)
    image = cv2.imread(img_path)
    height, width, _ = image.shape
    image = cv2.resize(image,(width // 2,height // 2), interpolation=cv2.INTER_LINEAR)
    direction = np.random.choice(['top', 'bottom', 'none'])
    hazed_image = apply_haze(image=image, direction=direction)

    cv2.imwrite(os.path.join(out_path,subfolder,'images',filename),hazed_image)

    ###### process labels, do not remove. Labels need to be half-sized too.
    if subfolder != "test":
        label_filename=os.path.join(in_path,subfolder,"labelTxt",filename.replace(".png",".txt"))
        new_annotation=[]
        header=[]
        try:
            with open(label_filename,'r') as f:
                # print(f"Processing: {filename}")
                lines=[line.strip().split() for line in f.readlines()]
                # print(lines[0])
                header.append(lines[0][0])
                header.append(lines[1][0])
                
                for i in range(2,len(lines)):
                    line=lines[i]

                    points=np.array([float(x) for x in line[:-2]])
                    points /= 2.0
                    points=points.astype(int)
                    points_list=list(points)
                    points_list=[str(x) for x in points_list]
                    points_list.append(line[-2])
                    points_list.append(line[-1])

                    # new_annotation.append(f"{" ".join(points_list)}")
            with open(os.path.join(out_path,subfolder,"labelTxt",filename.replace(".png",".txt")),"w") as f:            
                f.write('\n'.join([*header,*new_annotation]))
        except:
            print(f"No label found for {filename}")

def generate(in_path, out_path,subfolder):
    os.makedirs(os.path.join(out_path,subfolder,"images"),exist_ok=True)
    os.makedirs(os.path.join(out_path,subfolder,"labelTxt"),exist_ok=True)
    
    # Get all image filenames
    image_dir = os.path.join(in_path, subfolder, "images")
    filenames = [f for f in os.listdir(image_dir) if f.endswith(".png")]
    
    # Parallel execution with progress bar
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(process)(filename, in_path, out_path, subfolder)
        for filename in filenames
    )

    # Handle failed files
    failed_files = [f for f in results if f is not None]
    if failed_files:
        print(f"Failed to process {len(failed_files)} files: {failed_files}")
        
        


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


def apply_haze(image, A_range=(0.9, 1.0), t_range=(0.5, 0.7), direction='top'):
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

def generate_haze_map_gaussian(kernel_size, numb_patches, sigma, image_shape):
    '''
    Plan:
    - Create a haze map that has patches of varying intensity. Each patch's intensity is defined by gaussian distribution 
    - Will stride across the map, and apply a bit-wise operation of gaussian mask onto the patch 
    - Lastly, add the patch back to the haze map 
    '''

    map_height = image_shape[1]
    map_width = image_shape[0]

    stride_height = np.ceil(map_height / numb_patches).astype(np.int32)
    stride_width = np.ceil(map_width / numb_patches).astype(np.int32)

    # Create a standard gaussian kernel
    gaussian_1D = cv2.getGaussianKernel(kernel_size, sigma)

    gaussian_2D = gaussian_1D * gaussian_1D.T

    print("\n2D Gaussian kernel:")
    print(f"Shape: {gaussian_2D.shape}")
    print(gaussian_2D)
    

    haze_map = np.ones(image_shape, dtype=np.float64)

    # print("\n haze_map:")
    # print(f"Shape: {haze_map.shape}")
    # print(haze_map)

    for y in range(0, map_height, stride_height):
        for x in range(0, map_width, stride_width):
            
            #Get patch 
            patch = haze_map[y:y+kernel_size, x:x+kernel_size]

            # print("\n Patch:")
            # print(f"Shape: {patch.shape}")
            # print(patch)

            #Get randomized gaussian kernel 
            # gaussian_1D = cv2.getGaussianKernel(kernel_size, sigma)

            # gaussian_2D = gaussian_1D * gaussian_1D.T

            # print("\n2D Gaussian kernel:")
            # print(gaussian_2D)

            patch *= gaussian_2D

            haze_map[y:y+kernel_size, x:x+kernel_size] = patch

    return haze_map


def generate_train_and_val(in_path,out_path):
    print("Procesing train set")
    generate(in_path,out_path,"train")
    print("Processing validation set")
    generate(in_path,out_path,"val")
    print("Processing test set")
    generate(in_path,out_path,"test")

if __name__ == "__main__":
    # generate_train_and_val(
    #     in_path='../raw_data/dota_orig',
    #     out_path='../raw_data/dota_hazed'
    # )

    haze_map = generate_haze_map_gaussian(5, 1, 0.9, (256, 256))
    plt.imshow(haze_map, cmap='gray')
    plt.waitforbuttonpress()
    plt.close('all')

