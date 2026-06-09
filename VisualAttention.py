import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def main():
    # Specify the image and attention map paths
    LR_image_path = "./inference_image.png"
    attention_path = "./results/Attention_map/PFT_light_attention_map_w32_L10.npy"

    # Define the crop region for the LR image (left, upper, right, lower)
    LR_crop_box = (64, 32, 96, 64)  # Select the first window (32x32 region)
    
    # Load and crop the LR image
    LR_image = Image.open(LR_image_path).convert("RGBA")
    cropped_image = LR_image.crop(LR_crop_box)

    # Parameters for selecting specific attention map information
    batch_index = 10
    num_heads_index = 0
    query_index = 367
    visual_scale = 0.01        # Scale factor for better visualization

    # Load the attention map (shape: batch*windows, num_heads, n, n)
    attn = np.load(attention_path)

    # Extract the attention vector for the selected batch, head, and query
    attention_map = attn[batch_index][num_heads_index][query_index]

    # Reshape the attention vector into a 2D attention map
    attention_map_reshaped = attention_map.reshape((32, 32))

    # Scale attention values and clip them into [0, 1] range
    scaled_attention_map = np.clip(attention_map_reshaped / visual_scale, 0, 1)

    # Create a transparent overlay image with the same size as cropped image
    overlay = Image.new('RGBA', cropped_image.size, (255, 0, 0, 0))

    # Apply the attention map directly to the overlay
    for i in range(32):
        for j in range(32):
            alpha = int(scaled_attention_map[i, j] * 255)
            overlay.putpixel((j, i), (255, 0, 0, alpha))  # Red color with variable transparency

    # Calculate pixel coordinates for the query location inside 32x32
    x_index = query_index % 32
    y_index = query_index // 32

    # Highlight the query location with a small red rectangle
    draw = ImageDraw.Draw(cropped_image)
    draw.rectangle(
        (x_index - 1, y_index - 1, x_index + 1, y_index + 1),
        outline="red",
        width=1
    )

    # Save results
    os.makedirs("./visualization_results", exist_ok=True)

    overlay_save_path = f"./visualization_results/overlay_attention_map_b{batch_index}_h{num_heads_index}_q{query_index}.png"
    LR_crop_save_path = f"./visualization_results/LR_crop_image_with_query_b{batch_index}_h{num_heads_index}_q{query_index}.png"

    # Save the overlay image
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    ax.imshow(overlay)
    ax.axis('off')
    fig.savefig(overlay_save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # Save the LR_crop image with query highlight
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    ax.imshow(cropped_image)
    ax.axis('off')
    fig.savefig(LR_crop_save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    print(f"Overlay attention map saved to {overlay_save_path}")
    print(f"LR_crop image with query location saved to {LR_crop_save_path}")

if __name__ == "__main__":
    main()
