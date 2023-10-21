import logging
import argparse
from smbclient import register_session, scandir, open_file


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
        print(f"Error reading credentials file: {e}")

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
    with open_file(f"{share_path}\\{file_path}", "rb") as file:
        return file.read()


def create_file_index(share_path):
    """
    Recursively traverse the directories and create an index of all the files on the share.

    Parameters
    ----------
    directory_path : str
        The path of the directory to start indexing from.

    Returns
    -------
    file_index : list of str
        The list of all files on the share.
    """
    file_index = []

    for file_info in scandir(share_path):
        if file_info.is_file():
            file_index.append(f"{share_path}/{file_info.name}")
        elif file_info.is_dir():
            file_index.extend(create_file_index(f"{share_path}/{file_info.name}"))

    return file_index


def main():
    parser = setup_cli()
    args = parser.parse_args()

    if args.credfile:
        args.username, args.password = get_credentials(args.credfile)

    # Register session if credentials are provided
    if args.username and args.password:
        register_session(args.server, username=args.username, password=args.password)

    share_path = f"\\\\{args.server}\\{args.share}"
    # file_index = create_file_index(share_path)

    # for file in file_index:
    #     print(file)

    import time
    start = time.time()
    file_contents = download_file_to_memory(share_path, "text1G.txt")
    end = time.time()
    print(f"Time taken: {end - start}")
    # print(file_contents)

if __name__ == "__main__":
    main()
