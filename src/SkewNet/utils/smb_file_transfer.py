import logging
import os
from smbclient import open_file, register_session, scandir


def _get_credentials(credentials_file):
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

    Examples
    --------
    >>> _get_credentials("credentials.txt")
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


def create_connection(server, username="", password="", credentials_file=""):
    """
    Register an SMB session.

    Parameters
    ----------
    server : str
        The server name or IP address.
    username : str, optional
        The username to access the SMB share.
    password : str, optional
        The password to access the SMB share.
    credentials_file : str, optional
        File containing username and password.
    
    Examples
    --------
    >>> create_connection("server1", "user1", "pass1")
    >>> create_connection("server2", credentials_file="credentials.txt")
    >>> create_connection("123.123.123.123", username="user3", password="pass3")
    """
    if credentials_file:
        if username or password:
            logging.warning("Credentials file and username/password specified. Using credentials file.")
        username, password = _get_credentials(credentials_file)

    if username and password:
        register_session(server, username=username, password=password)
    else:
        register_session(server)


def clean_path(path):
    """
    Clean a path to be compatible with SMB.

    Parameters
    ----------
    path : str
        The path to clean.

    Returns
    -------
    str
        The cleaned path.

    Examples
    --------
    >>> clean_path("directory1\\directory2\\file1.txt")
    """
    if path.startswith("\\") or path.endswith("\\"):
        path = path.strip("\\")
    if path.startswith("/") or path.endswith("/"):
        path = path.strip("/")
    path = path.replace("/", "\\")

    return path


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

    Examples
    --------
    >>> download_file_to_memory("\\\\server1\\share1", "directory1\\file1.txt")
    """
    file_path = clean_path(file_path)

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

    Examples
    --------
    >>> download_file_to_disk("\\\\server1\\share1", "directory1\\file1.txt", "C:\\Users\\user1\\file1.txt")
    """
    file_path = clean_path(file_path)
    with open_file(rf"{share_path}\{file_path}", "rb") as file:
        with open(destination_path, "wb") as destination:
            destination.write(file.read())


def list_directory(share_path, directory_path):
    """
    List the contents of a directory on an SMB share.

    Parameters
    ----------
    share_path : str
        The path to the share.
    directory_path : str
        The path to the directory on the share.

    Returns
    -------
    list of str
        The list of files and directories in the specified directory.

    Examples
    --------
    >>> list_directory("\\\\server1\\share1", "directory1")
    """
    directory_path = clean_path(directory_path)
    return [file_info.name for file_info in scandir(rf"{share_path}\{directory_path}")]


def create_file_index(share_path, start_directory):
    """
    Recursively traverse the directories and create an index of all 
    the files on the share starting from the specified path.

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
    start_directory = clean_path(start_directory)

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


def filter_unique_items(items, key_function, priority_function):
    """
    Keep only one copy of each item, based on a priority function.

    Parameters
    ----------
    items : list
        The list of all items.
    key_function : callable
        A function that takes an item and returns a key for grouping.
    priority_function : callable
        A function that takes an item and returns a value for sorting within groups.

    Returns
    -------
    list
        The filtered list of items.
    """
    item_dict = {}
    for item in sorted(items, key=priority_function):
        key = key_function(item)
        if key not in item_dict:
            item_dict[key] = item

    return list(item_dict.values())


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


# def main():
#     parser = setup_cli()
#     args = parser.parse_args()

#     if args.credfile:
#         args.username, args.password = _get_credentials(args.credfile)

#     if args.username and args.password:
#         register_session(args.server, username=args.username, password=args.password)
#     else:
#         register_session(args.server)

#     share_path = rf"\\{args.server}\{args.share}"
#     directory_paths_on_share = [
#         "EVE_DRIVE",
#         # "cairogeniza",
#     ]

#     def get_file_name_without_extension(file):
#         name, _ = os.path.splitext(os.path.basename(file))
#         return name

#     def get_file_extension_priority(file, extensions):
#         extension = os.path.splitext(file)[-1].lower()
#         return extensions.index(extension) if extension in extensions else len(extensions)

#     image_extensions = (".tif", ".tiff", ".jpeg", ".jpg", ".png", ".jp2")

#     file_index = []
#     for path in directory_paths_on_share:
#         partial_index = create_file_index(share_path, path)
#         partial_index = filter_files_by_extension(partial_index, image_extensions)
#         partial_index = exclude_files_starting_with(partial_index, ".")

#         partial_index = filter_unique_items(
#             partial_index,
#             key_function=get_file_name_without_extension,
#             priority_function=lambda file: get_file_extension_priority(file, image_extensions),
#         )

#         file_index.extend(partial_index)

#     print(len(file_index))

    # save images to output directory, converting to jpeg, while keeping the directory structure
    # output_directory = "/scratch/gpfs/RUSTOW/test"

    # for file in file_index:
    #     file_contents = download_file_to_memory(share_path, file)

    #     # Create the same directory structure in the output directory
    #     output_path = os.path.join(output_directory, os.path.splitext(file)[0] + '.jpg')
    #     # Replace backslashes with forward slashes for Linux
    #     output_path = output_path.replace("\\", "/")
    #     os.makedirs(os.path.dirname(output_path), exist_ok=True)

    #     convert_and_save_image(file_contents, output_path)


# if __name__ == "__main__":
#     main()
