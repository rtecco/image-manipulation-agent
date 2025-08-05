#!/usr/bin/env python3
"""
REPL script to load the most recent state file and start an interactive session.
"""

import os
import sys
import dill
import tempfile
import glob
from pathlib import Path
from typing import Dict, Any, Optional
import code
from PIL import Image
from datetime import datetime

def find_most_recent_state_file() -> Optional[Path]:
    """Find the most recent state.pkl file in temporary directories."""
    temp_dir = Path(tempfile.gettempdir())
    
    # Look for state.pkl files in all subdirectories of temp
    state_files = list(temp_dir.glob("**/state.pkl"))
    
    if not state_files:
        return None
    
    # Return the most recently modified state file
    return max(state_files, key=lambda f: f.stat().st_mtime)

def load_state(state_file: Path) -> Dict[str, Any]:
    """Load state from dill file."""
    try:
        with open(state_file, 'rb') as f:
            return dill.load(f)
    except Exception as e:
        print(f"Error loading state file {state_file}: {e}")
        return {}

def display_state_summary(state: Dict[str, Any]) -> None:
    """Display a summary of the loaded state."""
    if not state:
        print("State is empty")
        return
    
    print(f"\n=== LOADED STATE ({len(state)} variables) ===")
    for key, value in state.items():
        value_type = type(value).__name__
        if isinstance(value, Image.Image):
            print(f"  {key}: {value_type} ({value.size}, {value.mode})")
        elif isinstance(value, (str, int, float, bool)):
            preview = str(value)[:50]
            if len(str(value)) > 50:
                preview += "..."
            print(f"  {key}: {value_type} = {preview}")
        else:
            print(f"  {key}: {value_type}")

def start_repl(state: Dict[str, Any]) -> None:
    """Start an interactive REPL with the state loaded as global variables."""
    # Inject state variables into the global namespace
    globals().update(state)
    
    print("\n=== INTERACTIVE REPL ===")
    print("State variables have been loaded into the global namespace.")
    print("Type 'help()' for Python help, or 'exit()' to quit.")
    print("Available functions:")
    print("  - show_image(var_name): Display an image variable")
    print("  - list_vars(): List all loaded variables")
    print("  - save_image(var_name, filename): Save an image to file")
    
    # Helper functions for the REPL
    def show_image(var_name: str) -> None:
        """Display an image variable."""
        if var_name in globals() and isinstance(globals()[var_name], Image.Image):
            globals()[var_name].show()
        else:
            print(f"Variable '{var_name}' is not an image or doesn't exist")
    
    def list_vars() -> None:
        """List all loaded variables."""
        print("\nLoaded variables:")
        for key, value in state.items():
            print(f"  {key}: {type(value).__name__}")
    
    def save_image(var_name: str, filename: str) -> None:
        """Save an image variable to file."""
        if var_name in globals() and isinstance(globals()[var_name], Image.Image):
            globals()[var_name].save(filename)
            print(f"Saved {var_name} to {filename}")
        else:
            print(f"Variable '{var_name}' is not an image or doesn't exist")
    
    # Add helper functions to globals
    globals()['show_image'] = show_image
    globals()['list_vars'] = list_vars
    globals()['save_image'] = save_image
    
    # Start the interactive console
    console = code.InteractiveConsole(globals())
    console.interact()

def main():
    """Main function to find state file and start REPL."""
    print("Looking for the most recent state file...")
    
    state_file = find_most_recent_state_file()
    if not state_file:
        print("No state.pkl files found in temporary directories.")
        print("Run a vision agent task first to generate state files.")
        sys.exit(1)
    
    print(f"Found state file: {state_file}")
    
    # Convert timestamp to readable local datetime
    mtime = state_file.stat().st_mtime
    local_datetime = datetime.fromtimestamp(mtime)
    print(f"Last modified: {local_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load the state
    state = load_state(state_file)
    
    # Display summary
    display_state_summary(state)
    
    # Start REPL
    start_repl(state)

if __name__ == "__main__":
    main()