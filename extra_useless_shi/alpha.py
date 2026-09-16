from PIL import Image
import sys
import os

def split_alpha_channel(image_path, output_dir=None):
    """
    Splits an image into its RGB and Alpha channel components.
    Saves the alpha channel as a separate grayscale image.
    """
    img = Image.open(image_path)

    # Ensure image has an alpha channel
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    r, g, b, a = img.split()

    # Prepare output paths
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = output_dir or os.path.dirname(image_path) or "."

    rgb_path = os.path.join(output_dir, f"{base_name}_rgb.png")
    alpha_path = os.path.join(output_dir, f"{base_name}_alpha.png")

    # Save RGB image (without alpha)
    Image.merge("RGB", (r, g, b)).save(rgb_path)

    # Save Alpha channel as grayscale image
    a.save(alpha_path)

    print(f"RGB image saved to:   {rgb_path}")
    print(f"Alpha channel saved to: {alpha_path}")

    return rgb_path, alpha_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python split_alpha.py <image_path> [output_dir]")
        sys.exit(1)

    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    split_alpha_channel(image_path, output_dir)