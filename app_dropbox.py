"""
app_dropbox.py

This module manages Dropbox integration for document access.
Provides:
- Authentication with Dropbox API using refresh tokens
- Listing folders and PDF files from Dropbox
- Downloading files from Dropbox
- Uploading files to Dropbox
- Converting downloaded files to file-like objects for processing
"""

import os
import dropbox
from dropbox.exceptions import AuthError as DropboxOAuth2TokenError
import logging
import streamlit as st
import io
from dotenv import load_dotenv
import time

# Load environment variables to ensure we have access to all Dropbox credentials
load_dotenv()

# Get Dropbox credentials from environment variables
def get_dropbox_credentials():
    """Get Dropbox credentials from environment variables"""
    credentials = {
        "app_key": os.getenv("DROPBOX_APPKEY"),
        "app_secret": os.getenv("DROPBOX_APPSECRET"),
        "refresh_token": os.getenv("DROPBOX_REFRESH_TOKEN"),
        "access_token": os.getenv("DROPBOX_TOKEN")
    }
    
    # Log available credentials (without showing the actual values)
    available = [k for k, v in credentials.items() if v]
    logging.info(f"Available Dropbox credentials: {available}")
    
    return credentials

# Initialize Dropbox client
def initialize_dropbox_client():
    """
    Initialize and validate Dropbox client using credentials from environment variables.
    This is the main entry point for Dropbox integration.
    
    Returns:
        Dropbox: A configured Dropbox client or None if not available
    """
    try:
        app_key = os.getenv('DROPBOX_APPKEY')
        app_secret = os.getenv('DROPBOX_APPSECRET')
        refresh_token = os.getenv('DROPBOX_REFRESH_TOKEN')
        
        if not all([app_key, app_secret, refresh_token]):
            missing = []
            if not app_key: missing.append("DROPBOX_APPKEY")
            if not app_secret: missing.append("DROPBOX_APPSECRET")
            if not refresh_token: missing.append("DROPBOX_REFRESH_TOKEN")
            
            logging.warning(f"Dropbox not fully configured. Missing: {', '.join(missing)}")
            return None
            
        logging.info("Initializing Dropbox client with refresh token")
        dbx = dropbox.Dropbox(
            app_key=app_key,
            app_secret=app_secret,
            oauth2_refresh_token=refresh_token
        )
        
        # Test connection to ensure it works
        account = dbx.users_get_current_account()
        logging.info(f"Successfully authenticated with Dropbox as {account.name.display_name}")
        return dbx
    except DropboxOAuth2TokenError as e:
        logging.error(f"Dropbox authentication error: {str(e)}")
        return None
    except Exception as e:
        logging.error(f"Error initializing Dropbox client: {str(e)}")
        return None

# Initialize Dropbox client with proper token refresh
def get_dropbox_client():
    """
    Initialize Dropbox client with token refresh capability.
    Uses access token first (simpler method), falls back to refresh token.
    """
    credentials = get_dropbox_credentials()
    
    # Try using access token first (simpler method)
    if credentials["access_token"]:
        try:
            logging.info("Initializing Dropbox client with access token")
            dbx = dropbox.Dropbox(credentials["access_token"])
            # Test the connection
            dbx.users_get_current_account()
            logging.info("Successfully authenticated with Dropbox using access token")
            return dbx
        except DropboxOAuth2TokenError as e:
            logging.error(f"Dropbox token expired or invalid (access token method): {str(e)}")
            logging.info("Attempting to refresh token...") # Added logging
        except Exception as e:
            logging.error(f"Error initializing Dropbox with access token: {str(e)}")

    logging.info("Checking for refresh token credentials...") # Added logging - check if we reach this point
    # Fall back to refresh token if access token not available or failed
    if credentials["refresh_token"] and credentials["app_key"] and credentials["app_secret"]:
        logging.info("Refresh token, app key, and app secret are available.") # Added logging
        try:
            logging.info("Initializing Dropbox client with refresh token")
            dbx = dropbox.Dropbox(
                app_key=credentials["app_key"],
                app_secret=credentials["app_secret"],
                oauth2_refresh_token=credentials["refresh_token"]
            )
            # Test the connection
            dbx.users_get_current_account()
            logging.info("Successfully authenticated with Dropbox using refresh token")
            return dbx
        except Exception as e:
            logging.error(f"Error initializing Dropbox with refresh token: {str(e)}")
    else:
        logging.warning("Refresh token, app key, or app secret is missing.") # Added logging
    
    # If we get here, neither method worked
    logging.error("Failed to authenticate with Dropbox")
    return None

# List folders in Dropbox
def list_dropbox_folders(dbx=None):
    """
    List all folders in Dropbox
    
    Args:
        dbx: Optional Dropbox client
        
    Returns:
        list: List of folder paths
    """
    if not dbx:
        dbx = get_dropbox_client()
    if not dbx:
        return []
        
    try:
        # Use empty string for root folder
        result = dbx.files_list_folder("")  # Root folder is specified as empty string, not "/"
        folders = [entry.path_display for entry in result.entries if isinstance(entry, dropbox.files.FolderMetadata)]
        
        # Continue listing if there are more entries
        while result.has_more:
            result = dbx.files_list_folder_continue(result.cursor)
            folders.extend([entry.path_display for entry in result.entries if isinstance(entry, dropbox.files.FolderMetadata)])
            
        return folders
    except Exception as e:
        logging.error(f"Error listing Dropbox folders: {str(e)}")
        return []

# List PDF files in a Dropbox folder
def list_dropbox_pdf_files(folder_path="/", dbx=None):
    """
    List all PDF files in a Dropbox folder
    
    Args:
        folder_path: Path to folder in Dropbox
        dbx: Optional Dropbox client
        
    Returns:
        list: List of PDF file paths
    """
    if not dbx:
        dbx = get_dropbox_client()
    if not dbx:
        return []
        
    try:
        # Dropbox API requires empty string for root folder
        api_path = "" if folder_path == "/" else folder_path
        
        result = dbx.files_list_folder(api_path, recursive=True)
        pdf_files = [entry.path_display for entry in result.entries 
                    if isinstance(entry, dropbox.files.FileMetadata) 
                    and entry.path_display.lower().endswith('.pdf')]
        
        # Continue listing if there are more entries
        while result.has_more:
            result = dbx.files_list_folder_continue(result.cursor)
            pdf_files.extend([entry.path_display for entry in result.entries 
                            if isinstance(entry, dropbox.files.FileMetadata) 
                            and entry.path_display.lower().endswith('.pdf')])
            
        return pdf_files
    except Exception as e:
        logging.error(f"Error listing Dropbox PDF files: {str(e)}")
        return []

# Download a file from Dropbox
def download_dropbox_file(file_path, dbx=None):
    if not dbx:
        dbx = get_dropbox_client()
    if not dbx:
        return None
    
    try:
        # For files_download: Must have leading slash
        if not file_path.startswith("/"):
            api_file_path = f"/{file_path}"
        else:
            api_file_path = file_path
            
        logging.info(f"Downloading Dropbox file: {api_file_path}")
        
        try:
            md, res = dbx.files_download(path=api_file_path)
            data = res.content
            logging.info(f"Successfully downloaded {api_file_path}")
            return data
        except Exception as download_error:
            logging.error(f"Download error: {str(download_error)}")
            return None
    except Exception as e:
        logging.error(f"Error downloading Dropbox file: {str(e)}")
        st.error(f"Error downloading Dropbox file: {str(e)}")
        return None

# Upload a file to Dropbox
def upload_to_dropbox(file_data, file_name, folder_path="/", dbx=None):
    if not dbx:
        dbx = get_dropbox_client()
    if not dbx:
        return False
    
    try:
        # Always upload to root level 
        logging.info(f"Uploading file {file_name} to root level in Dropbox")
        
        # Upload the file to root with leading slash for path
        file_path = f"/{file_name}"
        logging.info(f"Uploading file to Dropbox: {file_path}")
        
        dbx.files_upload(
            file_data, 
            file_path,  # Use root path with leading slash
            mode=dropbox.files.WriteMode.overwrite
        )
        logging.info(f"Successfully uploaded {file_name} to Dropbox root")
        return True
    except Exception as e:
        logging.error(f"Error uploading to Dropbox: {str(e)}")
        st.error(f"Error uploading to Dropbox: {str(e)}")
        return False

# Create a file-like object from Dropbox file data
def create_file_like_object(file_data, file_name):
    if not file_data:
        return None
    
    try:
        file_like = io.BytesIO(file_data)
        file_like.name = file_name  # Add name attribute
        return file_like
    except Exception as e:
        logging.error(f"Error creating file-like object: {str(e)}")
        return None

# Check if Dropbox is configured
def is_dropbox_configured():
    """Check if Dropbox credentials are properly configured"""
    credentials = get_dropbox_credentials()
    if credentials["app_key"] and credentials["app_secret"] and credentials["refresh_token"]:
        return True
    return False

# Save a file to Dropbox
def save_file_to_dropbox(file_obj, path):
    """
    Save a file to Dropbox
    
    Args:
        file_obj: File object (must have read method)
        path: Dropbox path where the file should be saved
        
    Returns:
        bool: Success indicator
    """
    try:
        # Get Dropbox client
        dbx = get_dropbox_client()
        if not dbx:
            logging.error("Could not initialize Dropbox client")
            return False
        
        # Read file data
        file_data = file_obj.read()
        
        # Extract filename from path
        filename = os.path.basename(path)
        folder_path = os.path.dirname(path)
        if not folder_path:
            folder_path = "/"
            
        # Upload to Dropbox
        return upload_to_dropbox(file_data, filename, folder_path, dbx)
    except Exception as e:
        logging.error(f"Error saving file to Dropbox: {str(e)}")
        return False

# Helper function to set up Dropbox using OAuth flow (call this manually to get tokens)
def setup_dropbox_oauth():
    """
    Run this function in a Python script to set up Dropbox OAuth.
    You'll need to copy and paste the authorization code from the browser.
    Then update your .env file with the tokens.
    """
    from dropbox import DropboxOAuth2FlowNoRedirect
    
    app_key = input("Enter your Dropbox app key: ").strip()
    app_secret = input("Enter your Dropbox app secret: ").strip()
    
    auth_flow = DropboxOAuth2FlowNoRedirect(
        app_key,
        consumer_secret=app_secret,
        token_access_type='offline',
        scope=['files.metadata.read', 'files.content.read', 'files.content.write', 'account_info.read']
    )
    
    authorize_url = auth_flow.start()
    print("1. Go to: " + authorize_url)
    print("2. Click \"Allow\" (you might have to log in first).")
    print("3. Copy the authorization code.")
    auth_code = input("Enter the authorization code here: ").strip()
    
    try:
        oauth_result = auth_flow.finish(auth_code)
        print("\nDROPBOX_APPKEY =", app_key)
        print("DROPBOX_APPSECRET =", app_secret)
        print("DROPBOX_REFRESH_TOKEN =", oauth_result.refresh_token)
        print("DROPBOX_TOKEN =", oauth_result.access_token)
        print("\nScopes:", oauth_result.scope)
        print("Token expires at:", oauth_result.expires_at)
        print("\nCopy these values to your .env file")
        return oauth_result
    except Exception as e:
        print('Error: %s' % (e,))
        return None

# Create a folder in Dropbox
def create_dropbox_folder(folder_path, dbx=None):
    """
    Create a folder in Dropbox if it doesn't exist
    
    Args:
        folder_path: Path for the folder (with or without leading slash)
        dbx: Dropbox client (optional)
        
    Returns:
        bool: Success indicator
    """
    if not dbx:
        dbx = get_dropbox_client()
    if not dbx:
        return False
    
    # Ensure folder path has leading slash
    if not folder_path.startswith("/"):
        folder_path = f"/{folder_path}"
    
    try:
        # Check if folder already exists
        try:
            dbx.files_get_metadata(folder_path)
            logging.info(f"Folder {folder_path} already exists")
            return True
        except Exception:
            # Folder doesn't exist, try to create it
            logging.info(f"Creating folder {folder_path}")
            dbx.files_create_folder_v2(folder_path)
            logging.info(f"Successfully created folder {folder_path}")
            return True
    except Exception as e:
        logging.error(f"Error creating folder {folder_path}: {str(e)}")
        if st:  # Check if streamlit is available
            st.error(f"Error creating Dropbox folder: {str(e)}")
        return False

# Delete a file from Dropbox
def delete_dropbox_file(file_path, dbx=None):
    """
    Delete a file from Dropbox
    
    Args:
        file_path: Path to the file (with or without leading slash)
        dbx: Dropbox client (optional)
        
    Returns:
        bool: Success indicator
    """
    if not dbx:
        dbx = get_dropbox_client()
    if not dbx:
        return False
    
    # Ensure file path has leading slash
    if not file_path.startswith("/"):
        file_path = f"/{file_path}"
    
    try:
        # Delete the file
        logging.info(f"Deleting file {file_path}")
        dbx.files_delete_v2(file_path)
        logging.info(f"Successfully deleted file {file_path}")
        return True
    except Exception as e:
        logging.error(f"Error deleting file {file_path}: {str(e)}")
        if st:  # Check if streamlit is available
            st.error(f"Error deleting Dropbox file: {str(e)}")
        return False

# Get a file from Dropbox and return it as a file-like object
def get_file_from_dropbox(file_path, dbx=None):
    """
    Download a file from Dropbox and return it as a file-like object
    
    Args:
        file_path: Path to file in Dropbox
        dbx: Optional Dropbox client
        
    Returns:
        BytesIO: File-like object or None if download failed
    """
    if not dbx:
        dbx = initialize_dropbox_client()
    if not dbx:
        logging.error("Could not initialize Dropbox client")
        return None
        
    try:
        # Make sure path starts with /
        if not file_path.startswith('/'):
            file_path = f"/{file_path}"
            
        logging.info(f"Downloading file from Dropbox: {file_path}")
        metadata, response = dbx.files_download(file_path)
        
        # Create file-like object from content
        file_content = response.content
        file_obj = io.BytesIO(file_content)
        
        # Set the filename attribute
        file_obj.name = os.path.basename(file_path)
        
        logging.info(f"Successfully downloaded {len(file_content)} bytes from {file_path}")
        return file_obj
    except Exception as e:
        logging.error(f"Error downloading file from Dropbox: {str(e)}")
        return None

# For handling dropbox on-demand feature
def set_dropbox_on_demand(enabled=True):
    """
    Configure Dropbox to only connect when necessary
    
    Args:
        enabled: Whether to enable on-demand mode
    """
    return True

# Get Dropbox usage statistics
def get_dropbox_usage_stats():
    """
    Get usage statistics for Dropbox
    
    Returns:
        dict: Statistics about Dropbox usage
    """
    return {
        "enabled": is_dropbox_configured(),
        "files_available": True if is_dropbox_configured() else False
    }
