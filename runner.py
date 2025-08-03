import subprocess
import sys
from typing import Dict, Any, Tuple, Optional
import pickle
import tempfile
import os
from pathlib import Path
from PIL import Image


class ProgramRunner:
    """Executes Python code snippets with persistence across iterations."""
    
    def __init__(self) -> None:
        """Initialize the program runner."""
        self.state: Dict[str, Any] = {}
        self.temp_dir = tempfile.mkdtemp()
        self.state_file = Path(self.temp_dir) / "state.pkl"
        
        # Load execution template
        template_path = Path(__file__).parent / "execution_template.py.tmpl"
        self.execution_template = template_path.read_text()
        
    def load_image(self, image: Image.Image, index: int = 0) -> None:
        """Load an image as a global variable image_clue_i."""
        variable_name = f"image_clue_{index}"
        self.state[variable_name] = image
        
    def execute_code(self, code: str) -> Tuple[str, str, bool]:
        """
        Execute Python code in isolated subprocess with state persistence.
        
        Returns:
            Tuple of (stdout, stderr, success)
        """
        # Save current state
        with open(self.state_file, 'wb') as f:
            pickle.dump(self.state, f)
            
        # Create execution script from template
        execution_script = self.execution_template.format(
            state_file=self.state_file,
            user_code=self._indent_code(code)
        )
        
        # Execute in subprocess
        try:
            result = subprocess.run(
                [sys.executable, '-c', execution_script],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.temp_dir
            )
            
            # Load updated state
            if self.state_file.exists():
                try:
                    with open(self.state_file, 'rb') as f:
                        self.state.update(pickle.load(f))
                except:
                    pass
                    
            return result.stdout, result.stderr, result.returncode == 0
            
        except subprocess.TimeoutExpired:
            return "", "Code execution timed out", False
        except Exception as e:
            return "", f"Execution error: {e}", False
    
    def _indent_code(self, code: str) -> str:
        """Indent code for inclusion in script."""
        return '\n'.join('    ' + line for line in code.split('\n'))
    
    def get_state(self) -> Dict[str, Any]:
        """Get current execution state."""
        return self.state.copy()
    
    def clear_state(self) -> None:
        """Clear execution state."""
        self.state.clear()