import cv2
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np


def convert_png_to_jpeg(png_path, output_directory, new_name=None,
                        background_color=(255, 255, 255)):
    """
    Convert a PNG image to JPEG format and save it to the specified directory. If no new name is specified, the
    original name will be used. The new name should not include the file extension.

    Parameters
    ----------
    png_path : Path
        Path to the PNG image.
    output_directory : Path
        Directory where the JPEG image will be saved.
    new_name : str, optional
        New name for the JPEG image (without the file extension).
    background_color : tuple, optional
        RGB color to fill the alpha channel, default is white (255, 255, 255).
    """
    png = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)

    if png is None:
        print(f"Failed to load image: {png_path}")
        return

    if png.shape[2] == 4:
        alpha_channel = png[:, :, 3]
        rgb_channels = png[:, :, :3]

        background = np.full_like(rgb_channels, background_color)
        final_rgb = (rgb_channels * (alpha_channel / 255.0)[:, :, None] +
                     background * (1 - (alpha_channel / 255.0)[:, :, None]))
        final_rgb = final_rgb.astype(np.uint8)
    else:
        final_rgb = png[:, :, :3]

    if new_name is None:
        new_name = png_path.stem + ".jpg"
    else:
        new_name = new_name + ".jpg"

    jpeg_path = output_directory / new_name
    cv2.imwrite(str(jpeg_path), final_rgb, [cv2.IMWRITE_JPEG_QUALITY, 100])


def get_images(input_folder):
    """
    Get all PNG images in the specified directory and its subdirectories.

    Parameters
    ----------
    input_folder : Path
        The directory to search for PNG images.

    Returns
    -------
    list of Path
    """
    return list(input_folder.glob("**/*.png"))


if __name__ == "__main__":
    png_images = get_images(Path(
        r"\\wsl.localhost\Ubuntu\home\ethan\rustow\deskewing_datasets\images\cudl_images\segmented_images\images_01"))
    output_dir = Path(r"C:\Users\ethan_haque\Desktop\tmp\jpg_images")
    with ProcessPoolExecutor(max_workers=8) as executor:
        list(executor.map(convert_png_to_jpeg, png_images, [output_dir] * len(png_images)))
