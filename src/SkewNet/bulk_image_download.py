import argparse
import multiprocessing
import os
import time

import psutil
from utils import image_utils
from utils import smb_file_transfer as smb


def setup_cli():
    """Sets up the command line interface for the script.

    Returns
    -------
    argparse.ArgumentParser
        The parser object.
    """
    parser = argparse.ArgumentParser(description="Tool for transferring files to and from an SMB share.")
    parser.add_argument("server", type=str, help="The server name or IP address.")
    parser.add_argument("share", type=str, help="The share name.")
    parser.add_argument("-A", "--credfile", type=str, help="File containing username and password.")
    parser.add_argument("-u", "--username", type=str, default="", help="The username to access the SMB share.")
    parser.add_argument("-p", "--password", type=str, default="", help="The password to access the SMB share.")

    return parser


def worker(ouput_directory, queue, process_function, *args):
    """
    Process the downloaded files.

    Parameters
    ----------
    ouput_directory : str
        The directory to save the downloaded files to.
    queue : multiprocessing.Queue
        The queue to get the downloaded files from.
    process_function : callable
        The function to process the downloaded files.
    """
    while True:
        item = queue.get()
        if item is None:
            break
        file, file_contents = item

        process_function(file_contents, rf"{ouput_directory}\{file}", *args)


def get_file_extension_priority(file, extensions):
    """
    Get the priority of the file extension in the list of extensions.
    The priority is the index of the extension in the list.
    If the extension is not in the list, the priority is the length of the list.

    Parameters
    ----------
    file : str
        The path to the file.
    extensions : list of str
        The list of extensions to check.

    Returns
    -------
    int
        The priority of the file extension.
    """
    extension = os.path.splitext(file)[-1].lower()
    return extensions.index(extension) if extension in extensions else len(extensions)


def create_image_index(share_path, directory_paths_on_share):
    """
    Create an index of all the image files on the share.

    Parameters
    ----------
    share_path : str
        The path to the share.
    directory_paths_on_share : list of str
        The paths to the directories on the share.

    Returns
    -------
    list of str
        The index of all the image files on the share.
    """
    file_index = []
    for path in directory_paths_on_share:
        partial_index = smb.create_file_index(share_path, path)

        image_extensions = (".tif", ".tiff", ".jpeg", ".jpg", ".png", ".jp2")
        partial_index = smb.filter_files_by_extension(partial_index, image_extensions)
        partial_index = smb.exclude_files_starting_with(partial_index, ".")
        partial_index = smb.filter_unique_items(
            partial_index,
            image_utils.get_file_name_without_extension,
            lambda file: get_file_extension_priority(file, image_extensions),
        )

        file_index.extend(partial_index)

    return file_index


def process_image(file_contents, output_path, largest_dimension):
    """
    Process the downloaded image files.

    Parameters
    ----------
    file_contents : bytes
        The contents of the file.
    output_path : str
        The path to save the file to.
    largest_dimension : int
        The largest dimension of the image.
    """
    output_path = output_path.replace("\\", "/")
    output_path = os.path.join(os.path.dirname(output_path), os.path.basename(output_path).split(".")[0] + ".jpg")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image_utils.convert_bytes_to_jpeg_and_resize(file_contents, str(output_path), largest_dimension)


def download_files(file_index, share_path, queue):
    """
    Download the files from the share.

    Parameters
    ----------
    file_index : list of str
        The index of the files to download.
    share_path : str
        The path to the share.
    queue : multiprocessing.Queue
        The queue to put the downloaded files in.

    Returns
    -------
    int
        The total size of the downloaded files.
    """
    total_size_downloaded = 0
    start_time = time.time()

    for i, file in enumerate(file_index):
        file_contents = smb.download_file_to_memory(share_path, file)
        total_size_downloaded += len(file_contents)
        queue.put((file, file_contents))

        elapsed_time = time.time() - start_time
        print_download_info(i, elapsed_time, total_size_downloaded, file_index)

    return total_size_downloaded


def print_download_info(i, elapsed_time, total_size_downloaded, file_index):
    """
    Print the download information.

    Parameters
    ----------
    i : int
        The index of the current file.
    elapsed_time : float
        The elapsed time since the start of the download.
    total_size_downloaded : int
        The total size of the downloaded files.
    file_index : list of str
        The index of the files to download.

    Returns
    -------
    int
        The total size of the downloaded files.
    """
    if elapsed_time > 0:  # Avoid division by zero
        download_speed = total_size_downloaded / elapsed_time / (1024 * 1024)  # MB per second
        info_string = (
            f"Speed: {download_speed:.2f} MB/s, "
            f"Downloaded: {total_size_downloaded / (1024 * 1024):.2f} MB, "
            f"Time: {elapsed_time:.2f} s, "
            f"Memory: {psutil.virtual_memory().percent}%, "
            f"CPU: {psutil.cpu_percent()}%, "
            f"Files: {i + 1}/{len(file_index)}"
        )
        print(info_string, end="\r")


def main():
    args = setup_cli().parse_args()
    smb.create_connection(args.server, args.username, args.password, args.credfile)

    output_directory = "/scratch/gpfs/RUSTOW/test"
    share_path = rf"\\{args.server}\{args.share}"
    directory_paths_on_share = ["EVE_DRIVE"]

    largest_dimension = 1024
    num_workers = 16

    file_index = create_image_index(share_path, directory_paths_on_share)

    queue = multiprocessing.Queue()
    workers = [
        multiprocessing.Process(target=worker, args=(output_directory, queue, process_image, largest_dimension))
        for _ in range(num_workers)
    ]

    for w in workers:
        w.start()

    download_files(file_index, share_path, queue)

    # Signal workers that no more files to process
    for _ in range(num_workers):
        queue.put(None)

    for w in workers:
        w.join()


if __name__ == "__main__":
    main()
