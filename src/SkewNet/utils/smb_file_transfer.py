import logging
import argparse
import uuid
from io import BytesIO
from smbprotocol.connection import Connection
from smbprotocol.session import Session
from smbprotocol.tree import TreeConnect
from smbprotocol.open import (
    CreateDisposition,
    CreateOptions,
    FilePipePrinterAccessMask,
    ImpersonationLevel,
    Open,
    ShareAccess,
)
from smbprotocol.file_info import FileInformationClass
import cv2
import numpy as np


def create_smb_connection(server, port, username, password, share):
    """
    Create a connection to an SMB server.

    Parameters
    ----------
    server : str
        The IP address or hostname of the SMB server.
    port : int
        The port number to connect to, typically 445.
    username : str
        The username for authentication.
    password : str
        The password for authentication.
    share : str
        The SMB share path.

    Returns
    -------
    connection : smbprotocol.connection.Connection
        The SMB connection object.
    session : smbprotocol.session.Session
        The SMB session object.
    tree : smbprotocol.tree.TreeConnect
        The SMB tree connect object.
    """
    try:
        connection = Connection(uuid.uuid4(), server, port)
        connection.connect()

        session = Session(connection, username, password)
        session.connect()

        tree = TreeConnect(session, share)
        tree.connect()

        logging.info(f"Successfully connected to {share} on {server}:{port} as {username}.")
        return connection, session, tree

    except Exception as error:
        logging.error(f"Failed to connect to {server}:{port} - {str(error)}")
        return None, None, None
    

def create_file_index(tree, directory_path):
    """
    Recursively traverse the directories and create an index of all the files on the share.

    Parameters
    ----------
    tree : smbprotocol.tree.TreeConnect
        The tree connect instance associated with the SMB share.
    directory_path : str
        The path of the directory to start indexing from.

    Returns
    -------
    file_index : list of str
        The list of all files on the share.
    """
    file_index = []
    try:
        dir_open = Open(tree, directory_path)
        dir_open.create(
            ImpersonationLevel.Impersonation,
            FilePipePrinterAccessMask.GENERIC_READ,
            0,
            ShareAccess.FILE_SHARE_READ,
            CreateDisposition.FILE_OPEN,
            CreateOptions.FILE_DIRECTORY_FILE
        )

        info_class = FileInformationClass.FILE_ID_BOTH_DIR_INFORMATION
        buffer = dir_open.query_directory(info_class)
        contents = [item['file_name'].get_value().decode('utf-16-le') for item in buffer]

        for item in contents:
            if item in ['.', '..']:
                continue  # Skip current and parent directory entries
            
            item_path = f"{directory_path}/{item}"
            if item_path.endswith('/'):  # It's a directory
                file_index.extend(create_file_index(tree, item_path))
            else:
                file_index.append(item_path)

        dir_open.close()

        logging.info(f"Indexed directory {directory_path}.")
        return file_index

    except Exception as error:
        logging.error(f"An error occurred while indexing {directory_path}: {str(error)}")
        return file_index  # Return the index built so far


def download_file_to_memory(tree, file_path):
    """
    Download a file from an SMB share to memory.

    This function downloads a specified file from an SMB share and stores it in memory. 
    If the file is successfully downloaded, it returns the file's bytes; otherwise, 
    it handles the exception and returns None.

    Parameters
    ----------
    tree : smbprotocol.tree.TreeConnect
        The tree connect instance associated with the SMB share.
    file_path : str
        The path of the file to be downloaded from the SMB share.

    Returns
    -------
    file_bytes : bytes or None
        The bytes of the downloaded file if successful, None otherwise.

    Raises
    ------
    Exception
        Handles any exception that occurs during the file download process and logs the error.
    """
    try:
        file_open = Open(tree, file_path)
        file_open.create(
            ImpersonationLevel.Impersonation,
            FilePipePrinterAccessMask.GENERIC_READ,
            0,
            ShareAccess.FILE_SHARE_READ,
            CreateDisposition.FILE_OPEN,
            CreateOptions.FILE_NON_DIRECTORY_FILE
        )

        file_bytes = BytesIO()

        offset = 0
        while offset < file_open.end_of_file:
            length = file_open.connection.max_read_size
            data = file_open.read(offset, length)
            file_bytes.write(data)
            offset += length
        
        # Reset the file pointer to the beginning of the file
        file_bytes.seek(0)

        file_open.close()

        logging.info(f"Downloaded file {file_path} to memory.")
        return file_bytes

    except Exception as error:
        logging.error(f"An error occurred while downloading {file_path}: {str(error)}")
        return None


def list_directory_contents(tree, directory_path):
    """
    List the contents of a directory on an SMB share.

    Parameters
    ----------
    tree : smbprotocol.tree.TreeConnect
        The tree connect instance associated with the SMB share.
    directory_path : str
        The path of the directory to list the contents of.

    Returns
    -------
    contents : list of str
        The list of contents of the directory.
    """
    try:
        dir_open = Open(tree, directory_path)
        dir_open.create(
            ImpersonationLevel.Impersonation,
            FilePipePrinterAccessMask.GENERIC_READ,
            0,
            ShareAccess.FILE_SHARE_READ,
            CreateDisposition.FILE_OPEN,
            CreateOptions.FILE_DIRECTORY_FILE
        )

        info_class = FileInformationClass.FILE_NAMES_INFORMATION     
        buffer = dir_open.query_directory("*", info_class)
        contents = [item['file_name'].get_value().decode('utf-16-le') for item in buffer]
        contents = [item for item in contents if item not in ['.', '..']]

        dir_open.close()

        logging.info(f"Listed contents of directory {directory_path}.")
        return contents

    except Exception as error:
        logging.error(f"An error occurred while listing contents of {directory_path}: {str(error)}")
        return None


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
        with open(credentials_file, 'r') as file:
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
    parser = argparse.ArgumentParser(description='Tool for transferring files to and from an SMB share.')
    parser.add_argument('server', type=str, help='The name of the SMB share.')
    parser.add_argument('share', type=str, help='The name of the SMB share.')
    parser.add_argument('-A', '--credfile', type=str, help='File containing username and password.')
    parser.add_argument('-u', '--username', type=str, default='', help='The username to access the SMB share.')
    parser.add_argument('-P', '--port', type=int, default=445, help='The port number to connect to, typically 445.')
    parser.add_argument('-p', '--password', type=str, default='', help='The password to access the SMB share.')

    return parser


def main():
    parser = setup_cli()
    args = parser.parse_args()

    if args.credfile:
        args.username, args.password = get_credentials(args.credfile)
    
    connection, session, tree = create_smb_connection(args.server, args.port, args.username, args.password, args.share)
    
    if connection and session and tree:
        # content = download_file_to_memory(tree, r"cairogeniza\Krengel\00003b\Krengel_003b_r.tif")
        # np_array = np.frombuffer(content.getvalue(), dtype=np.uint8)
        # image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        # cv2.imwrite("/scratch/gpfs/RUSTOW/tmp.jpg", image)

        directory_contents = list_directory_contents(tree, r"cairogeniza\Krengel\00003b")
        print(directory_contents)

        # index = create_file_index(tree, r"cairogeniza\Krengel\00003b")
        # print(index)

        # tree.disconnect()
        # session.disconnect()
        # connection.disconnect()
    

if __name__ == "__main__":
    main()