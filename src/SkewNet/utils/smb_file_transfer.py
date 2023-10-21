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
    


def download_file_to_memory(tree, file_path):
    """
    Download a file from an SMB share to memory.

    This function downloads a specified file from an SMB share and stores it in memory. 
    It uses an established SMB connection, session, and tree to access the file. 
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

        file_size = file_open.file_attributes['end_of_file']
        file_bytes = file_open.read(0, file_size)
        file_open.close()

        logging.info(f"Downloaded file {file_path} to memory.")
        return file_bytes

    except Exception as error:
        logging.error(f"An error occurred while downloading {file_path}: {str(error)}")
        # return None
        raise error


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
        download_file_to_memory(tree, "test.txt")

    tree.disconnect()
    session.disconnect()
    connection.disconnect()


if __name__ == "__main__":
    main()
