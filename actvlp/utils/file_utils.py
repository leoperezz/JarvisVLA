import os
import json
import shutil
import pathlib
from typing import Union
import rich

def load_json_file(file_path: Union[str, pathlib.Path], data_type="dict"):
    """
    Load a JSON file from the given path.

    Args:
        file_path (Union[str, pathlib.Path]): Path to the JSON file.
        data_type (str): Expected data type of the JSON content ("dict" or "list").

    Returns:
        dict or list: Loaded JSON content. Returns an empty dictionary or list if the file does not exist.
    """
    if isinstance(file_path, pathlib.Path):
        file_path = str(file_path)  # Convert pathlib.Path to string

    # Initialize an empty object based on the specified data type
    if data_type == "dict":
        json_file = dict()
    elif data_type == "list":
        json_file = list()
    else:
        raise ValueError("Invalid data type. Expected 'dict' or 'list'.")

    # Check if the file exists
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding="utf-8") as f:
                json_file = json.load(f)  # Load JSON content
        except IOError as e:
            rich.print(f"[red]Failed to open file {file_path}: {e}[/red]")
        except json.JSONDecodeError as e:
            rich.print(f"[red]Error parsing JSON file {file_path}: {e}[/red]")
    else:
        rich.print(f"[yellow]File {file_path} does not exist. Returning an empty file...[/yellow]")

    return json_file

def dump_json_file(json_file, file_path: Union[str, pathlib.Path], indent=4, if_print=True, if_backup=True, if_backup_delete=False):
    """
    Save data to a JSON file with optional backup and logging.

    Args:
        json_file (dict or list): The JSON data to save.
        file_path (Union[str, pathlib.Path]): Path to save the JSON file.
        indent (int): Indentation level for formatting the JSON file (default is 4).
        if_print (bool): Whether to print status messages (default is True).
        if_backup (bool): Whether to create a backup before writing (default is True).
        if_backup_delete (bool): Whether to delete the backup after writing (default is False).
    """
    if isinstance(file_path, pathlib.Path):
        file_path = str(file_path)  # Convert pathlib.Path to string

    backup_path = file_path + ".bak"  # Define the backup file path

    # Create a backup if the file exists and backup is enabled
    if os.path.exists(file_path) and if_backup:
        shutil.copy(file_path, backup_path)

    before_exist = os.path.exists(file_path)  # Check if the file existed before writing

    try:
        # Write JSON data to file
        with open(file_path, 'w', encoding="utf-8") as f:
            json.dump(json_file, f, indent=indent, ensure_ascii=False)

        # Print status messages
        if before_exist and if_print:
            rich.print(f"[yellow]Updated {file_path}[/yellow]")
        elif if_print:
            rich.print(f"[green]Created {file_path}[/green]")

    except IOError as e:
        # Restore from backup if writing fails
        if os.path.exists(backup_path) and if_backup:
            shutil.copy(backup_path, file_path)
            if if_print:
                rich.print(f"[red]Failed to write {file_path}. Restored from backup: {e}[/red]")
        else:
            if if_print:
                rich.print(f"[red]Failed to write {file_path} and no backup available: {e}[/red]")

    finally:
        # Cleanup: Delete the backup file if required
        if if_backup:
            if os.path.exists(backup_path) and if_backup_delete:
                os.remove(backup_path)
            elif not os.path.exists(backup_path) and not if_backup_delete:  # If the file was initially empty, create a backup
                shutil.copy(file_path, backup_path)