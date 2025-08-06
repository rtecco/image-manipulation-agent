"""Shared utilities for Streamlit report generation."""

import tempfile
import dill
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from PIL import Image


def find_state_files() -> List[Path]:
    """Find all final_state_*.pkl files in temp directories."""
    temp_dir = Path(tempfile.gettempdir())
    return list(temp_dir.glob("**/final_state_*.pkl"))


def load_state(state_file: Path) -> Dict[str, Any]:
    """Load state from dill file."""
    try:
        with open(state_file, 'rb') as f:
            return dill.load(f)
    except Exception:
        return {}


def get_state_info(state_file: Path) -> Dict[str, Any]:
    """Get basic info about a state file."""
    state = load_state(state_file)
    
    # Use internal timestamp if available, otherwise file mtime
    if "_timestamp" in state:
        timestamp_str = state["_timestamp"]
        modified = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
    else:
        modified = datetime.fromtimestamp(state_file.stat().st_mtime)
    
    return {
        "path": state_file,
        "modified": modified,
        "task": state.get("_task", state.get("task", "Unknown")),
        "code_executions": len(state.get("code_results", [])),
        "iterations": state.get("iteration", 0),
        "max_iterations": state.get("max_iterations", 0),
        "success": state.get("iteration", 0) >= state.get("max_iterations", 0) and state.get("max_iterations", 0) > 0,
        "completed": state.get("_completed", False),
        "state": state
    }


def get_images_from_state(state: Dict[str, Any]) -> Dict[int, Image.Image]:
    """Extract all image_clue_n from state."""
    images = {}
    for key, value in state.items():
        if key.startswith("image_clue_") and isinstance(value, Image.Image):
            try:
                n = int(key.split("_")[-1])
                images[n] = value
            except ValueError:
                continue
    return images


def extract_plan(state: Dict[str, Any]) -> str:
    """Extract the plan from the first LLM response."""
    messages = state.get("messages", [])
    
    # Look for the first AI message containing a plan
    for msg in messages:
        if hasattr(msg, 'content'):
            content = str(msg.content)
            # Check if this looks like a plan (contains steps or substantial content)
            if len(content) > 100 and ('step' in content.lower() or 'STEPS=' in content):
                return content
    
    return "No plan found"