import subprocess
import sys
from typing import Dict, Any, Tuple
import pickle
import tempfile
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
        
    def load_image_clue(self, image: Image.Image, index: int = 0) -> None:
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

        print(execution_script)

        # Execute in subprocess
        try:
            result = subprocess.run(
                [sys.executable, '-c', execution_script],
                capture_output=True,
                text=True,
                timeout=90,
                cwd=self.temp_dir
            )
            
            # Load updated state
            if self.state_file.exists():
                with open(self.state_file, 'rb') as f:
                    self.state.update(pickle.load(f))                    
                    
            return result.stdout, result.stderr, result.returncode == 0
            
        except subprocess.TimeoutExpired:
            return "", "Code execution timed out", False
        except Exception as e:
            return "", f"Execution error: {e}", False
    
    def _indent_code(self, code: str) -> str:
        """Indent code for inclusion in script."""
        return '\n'.join('    ' + line for line in code.split('\n'))
    
    def get_state(self) -> Dict[str, Any]:
        return self.state.copy()
    
    def clear_state(self) -> None:
        self.state.clear()

    def print_state(self, show_images: bool = False) -> None:
        if self.state:
            print("\n=== FINAL STATE ===")
            import pprint
            pp = pprint.PrettyPrinter(indent=2, width=80, depth=3)
            for key, value in self.state.items():
                print(f"\n{key}: {type(value).__name__}")
                
                # Special handling for PIL Images
                if isinstance(value, Image.Image):
                    if not show_images:
                        continue

                    print(f"  Size: {value.size}, Mode: {value.mode}")
                    try:
                        # Create a copy with the key as title for display
                        display_image = value.copy()
                        import matplotlib.pyplot as plt
                        plt.figure()
                        plt.imshow(display_image)
                        plt.title(f"State Variable: {key}")
                        plt.axis('off')
                        plt.show()
                    except:
                        print(f"  Could not display image: {key}")
                else:
                    try:
                        pp.pprint(value)
                    except:
                        # Fallback for objects that can't be pretty printed
                        print(f"  {repr(value)[:200]}{'...' if len(repr(value)) > 200 else ''}")
        else:
            print("\n=== FINAL STATE ===")
            print("(empty)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Python code with state persistence")
    parser.add_argument("program_file", help="Path to Python program file to execute")
    parser.add_argument("--image0", help="Path to image file 0")
    parser.add_argument("--image1", help="Path to image file 1")
    parser.add_argument("--image2", help="Path to image file 2")
    parser.add_argument("--image3", help="Path to image file 3")
    parser.add_argument("--image4", help="Path to image file 4")
    args = parser.parse_args()
    
    # Read program from file
    program_path = Path(args.program_file)
    if not program_path.exists():
        print(f"Error: Program file {args.program_file} not found", file=sys.stderr)
        sys.exit(1)
    
    program_code = program_path.read_text()
    
    runner = ProgramRunner()
    
    # Load images if provided
    for i in range(5):
        image_arg = getattr(args, f'image{i}')
        if image_arg:
            try:
                image = Image.open(image_arg)
                runner.load_image_clue(image, i)
                print(f"Loaded image {i} from {image_arg}")
            except Exception as e:
                print(f"Error loading image {i} from {image_arg}: {e}", file=sys.stderr)
    
    stdout, stderr, success = runner.execute_code(program_code)
    
    print("=== EXECUTION RESULTS ===")
    print(f"Success: {success}")
    
    if stdout:
        print("\n=== STDOUT ===")
        print(stdout)
    
    if stderr:
        print("\n=== STDERR ===")
        print(stderr)
    
    runner.print_state()