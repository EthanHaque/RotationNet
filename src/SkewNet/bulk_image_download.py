import argparse
from utils import smb_file_transfer as smb
from utils import image_utils
import multiprocessing
import os
import time


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


def worker(ouput_directory, queue, process_function):
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

        process_function(file_contents, rf"{ouput_directory}\{file}")


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


def main():
    """Main function for the script."""
    parser = setup_cli()
    args = parser.parse_args()

    smb.create_connection(args.server, args.username, args.password, args.credfile)

    share_path = rf"\\{args.server}\{args.share}"
    directory_paths_on_share = [
        "EVE_DRIVE",
        # "cairogeniza",
    ]

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

    output_directory = "/scratch/gpfs/RUSTOW/test"

    queue = multiprocessing.Queue()
    largest_dimension = 1000

    def process_function(file_contents, output_path):
        output_path = output_path.replace("\\", "/")
        output_path = os.path.join(os.path.dirname(output_path), os.path.basename(output_path).split(".")[0] + ".jpg")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image_utils.convert_bytes_to_jpeg_and_resize(file_contents, str(output_path), largest_dimension)

    # Creating separate processes for processing the downloaded files
    num_workers = 2
    workers = [
        multiprocessing.Process(target=worker, args=(output_directory, queue, process_function))
        for _ in range(num_workers)
    ]
    for w in workers:
        w.start()


    total_size_downloaded = 0  # Total size of downloaded files in bytes
    start_time = time.time()  # Start time of downloading
    for file in file_index:
        file_contents = smb.download_file_to_memory(share_path, file)
        file_size = len(file_contents)
        total_size_downloaded += file_size
        queue.put((file, file_contents))

        # Calculate and print metrics
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time > 0:  # Avoid division by zero
            download_speed = total_size_downloaded / elapsed_time / (1024 * 1024)  # MB per second
            print(f"Current download speed: {download_speed:.2f} MB/s, Total size downloaded: {total_size_downloaded / (1024 * 1024):.2f} MB, Queue size: {queue.qsize()}", end="\r")


    # Adding a sentinel value to the queue for each worker to signal that there are no more files to process
    for _ in range(num_workers):
        queue.put(None)

    # Wait for all worker processes to complete
    for w in workers:
        w.join()


if __name__ == "__main__":
    main()
