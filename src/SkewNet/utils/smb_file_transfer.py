import logging
import argparse
from smbclient import register_session, scandir, open_file
import cv2
import numpy as np
import os


def get_credentials(credentials_file):
    """
    Read the username and password from a file.

    Parameters
    ----------
    credentials_file : str
        The path to the credentials file.
    
    Returns
    -------
    username : str
        The username.
    password : str
        The password.
    """
    username = ""
    password = ""
    try:
        with open(credentials_file, "r") as file:
            lines = file.readlines()
            username = lines[0].strip()
            password = lines[1].strip()
    except Exception as e:
        logging.error(f"Error reading credentials file: {e}")

    return username, password


def setup_cli():
    """Sets up the command line interface for the script.

    Returns
    -------
    argparse.ArgumentParser
        The parser object.
    """
    parser = argparse.ArgumentParser(description="Tool for transferring files to and from an SMB share.")
    parser.add_argument("server", type=str, help="The server name or IP address.")
    parser.add_argument("share", type=str, help="The name of the SMB share.")
    parser.add_argument("-A", "--credfile", type=str, help="File containing username and password.")
    parser.add_argument("-u", "--username", type=str, default="", help="The username to access the SMB share.")
    parser.add_argument("-p", "--password", type=str, default="", help="The password to access the SMB share.")

    return parser


def download_file_to_memory(share_path, file_path):
    """
    Download a file from an SMB share to memory.

    Parameters
    ----------
    share_path : str
        The path to the share.
    file_path : str
        The path to the file on the share.

    Returns
    -------
    bytes
        The file contents.
    """
    with open_file(rf"{share_path}\{file_path}", "rb") as file:
        return file.read()
    

def download_file_to_disk(share_path, file_path, destination_path):
    """
    Download a file from an SMB share to disk.

    Parameters
    ----------
    share_path : str
        The path to the share.
    file_path : str
        The path to the file on the share.
    destination_path : str
        The path to save the file to.
    """
    with open_file(rf"{share_path}\{file_path}", "rb") as file:
        with open(destination_path, "wb") as destination:
            destination.write(file.read())


def create_file_index(share_path, start_directory):
    """
    Recursively traverse the directories and create an index of all the files on the share starting from the specified path.

    Parameters
    ----------
    share_path : str
        The base path of the share.
    start_directory : str
        The directory on the share to start indexing from (relative to the share path).

    Returns
    -------
    file_index : list of str
        The list of all files on the share.
    """
    file_index = []
    start_path = rf"{share_path}\{start_directory}"

    for file_info in scandir(start_path):
        if file_info.is_file():
            file_index.append(start_directory + "\\" + file_info.name)
        elif file_info.is_dir():
            file_index.extend(create_file_index(share_path, start_directory + "\\" + file_info.name))

    return file_index



def filter_files_by_extension(file_index, extensions):
    """
    Filter files by their extensions.

    Parameters
    ----------
    file_index : list of str
        The list of all files.
    extensions : tuple of str
        The extensions to filter by.

    Returns
    -------
    list of str
        The filtered list of files.
    """
    return [f for f in file_index if f.lower().endswith(extensions)]


def exclude_files_starting_with(file_index, character):
    """
    Exclude files that start with a specific character.

    Parameters
    ----------
    file_index : list of str
        The list of all files.
    character : str
        The character to exclude.

    Returns
    -------
    list of str
        The filtered list of files.
    """
    return [f for f in file_index if not os.path.basename(f).startswith(character)]


def filter_unique_images(file_index, priority_order):
    """
    Keep only one copy of each image, based on a priority order of extensions.

    Parameters
    ----------
    file_index : list of str
        The list of all image files.
    priority_order : tuple of str
        The priority order of extensions.

    Returns
    -------
    list of str
        The filtered list of image files.
    """
    image_dict = {}
    for file in sorted(file_index, key=lambda x: priority_order.index(os.path.splitext(x)[-1].lower())):
        name, _ = os.path.splitext(os.path.basename(file))
        if name not in image_dict:
            image_dict[name] = file

    return list(image_dict.values())



def save_file_index(file_index, destination_path):
    """
    Save the file index to disk.

    Parameters
    ----------
    file_index : list of str
        The file index to save.
    destination_path : str
        The path to save the file index to.
    """
    file_index.sort()
    with open(destination_path, "w", encoding="utf-8") as file:
        output = "\n".join(file_index)
        file.write(output)


def convert_and_save_image(file_contents, output_path):
    """
    Convert the file contents to a JPEG image and save it to the specified path.

    Parameters
    ----------
    file_contents : bytes
        The contents of the image file.
    output_path : str
        The path to save the converted image to.
    """
    array = np.frombuffer(file_contents, dtype=np.uint8)
    img = cv2.imdecode(array, cv2.IMREAD_COLOR)
    cv2.imwrite(output_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def process_image_file(file, share_path, output_directory):
    try:
        file_contents = download_file_to_memory(share_path, file)
        
        # Create the same directory structure in the output directory
        output_path = os.path.join(output_directory, os.path.splitext(file)[0] + '.jpg')
        # Replace backslashes with forward slashes for Linux
        output_path = output_path.replace("\\", "/")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        convert_and_save_image(file_contents, output_path)
        logging.info(f"Processed {file}")

    except Exception as e:
        logging.error(f"Error processing file {file}: {e}")


def main():
    parser = setup_cli()
    args = parser.parse_args()

    if args.credfile:
        args.username, args.password = get_credentials(args.credfile)

    if args.username and args.password:
        register_session(args.server, username=args.username, password=args.password)
    else:
        register_session(args.server)

    share_path = rf"\\{args.server}\{args.share}"
    directory_paths_on_share = [
        "EVE_DRIVE",
        # "cairogeniza",
    ]

    file_index = []
    for path in directory_paths_on_share:
        partial_index = create_file_index(share_path, path)

        image_extensions = ('.tif', '.tiff', '.jpeg', '.jpg', '.png', '.jp2')
        partial_index = filter_files_by_extension(partial_index, image_extensions)
        partial_index = exclude_files_starting_with(partial_index, '.')
        partial_index = filter_unique_images(partial_index, image_extensions)

        file_index.extend(partial_index)


    # save images to output directory, converting to jpeg, while keeping the directory structure
    output_directory = "/scratch/gpfs/RUSTOW/test"

    for file in file_index:
        file_contents = download_file_to_memory(share_path, file)
        
        # Create the same directory structure in the output directory
        output_path = os.path.join(output_directory, os.path.splitext(file)[0] + '.jpg')
        # Replace backslashes with forward slashes for Linux
        output_path = output_path.replace("\\", "/")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        convert_and_save_image(file_contents, output_path)


if __name__ == "__main__":
    main()
