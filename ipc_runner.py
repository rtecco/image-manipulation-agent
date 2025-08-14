import subprocess
import sys
import pickle
import io
import tempfile
import base64
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
from PIL import Image

class IPCProgramRunner:
    """Executes Python code snippets via persistent subprocess with IPC communication."""
    
    def __init__(self) -> None:
        """Initialize the IPC program runner."""
        self.process: Optional[subprocess.Popen] = None
        self.temp_dir = tempfile.mkdtemp()
        self._start_server()
        
    def _start_server(self) -> None:
        """Start the execution server subprocess."""
        server_path = Path(__file__).parent / "execution_server.py"
        
        self.process = subprocess.Popen(
            [sys.executable, '-u', str(server_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=self.temp_dir
        )
        
        # Test connection
        response = self._send_command({'command': 'ping'})
        if not response.get('success'):
            raise RuntimeError(f"Failed to start execution server: {response}")
    
    def _send_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Send a command to the execution server and get response."""
        if not self.process or self.process.poll() is not None:
            raise RuntimeError("Execution server is not running")
        
        try:
            # Serialize and encode command
            command_data = pickle.dumps(command)
            encoded_data = base64.b64encode(command_data).decode('ascii')
            
            # Send command with newline
            self.process.stdin.write(encoded_data + '\n')
            self.process.stdin.flush()
            
            # Read response
            response_line = self.process.stdout.readline()
            if not response_line:
                raise RuntimeError("No response from execution server")
            
            # Decode and deserialize response
            encoded_response = response_line.strip()
            response_data = base64.b64decode(encoded_response)
            return pickle.loads(response_data)
            
        except Exception as e:
            raise RuntimeError(f"IPC communication failed: {e}")
    
    def load_image_clue(self, image: Image.Image, index: int = 0) -> None:
        """Load an image as a global variable image_clue_i."""
        variable_name = f"image_clue_{index}"
        
        # Convert image to bytes
        buf = io.BytesIO()
        image.save(buf, format='JPEG')
        image_data = buf.getvalue()
        
        # Send to server
        response = self._send_command({
            'command': 'load_image',
            'image_data': image_data,  # Send raw bytes with pickle
            'variable_name': variable_name
        })
        
        if not response.get('success'):
            raise RuntimeError(f"Failed to load image: {response.get('error')}")
    
    def execute_code(self, code: str) -> Tuple[str, str, bool]:
        """
        Execute Python code in the persistent subprocess.
        
        Returns:
            Tuple of (stdout, stderr, success)
        """
        response = self._send_command({
            'command': 'execute',
            'code': code
        })
        
        return (
            response.get('stdout', ''),
            response.get('stderr', ''),
            response.get('success', False)
        )
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current execution state from the server."""
        response = self._send_command({'command': 'list_variables'})
        
        if response.get('success'):
            return response.get('variables', {})
        else:
            return {}
    
    def clear_state(self) -> None:
        """Clear the execution state."""
        response = self._send_command({'command': 'clear_state'})
        
        if not response.get('success'):
            raise RuntimeError(f"Failed to clear state: {response.get('error')}")
    
    def get_result_image(self, n: int) -> Image.Image | None:
        """Return the image_clue_n, or None if it doesn't exist."""
        variable_name = f"image_clue_{n}"
        
        response = self._send_command({
            'command': 'get_variable',
            'variable_name': variable_name
        })
        
        if response.get('success') and response.get('type') == 'image':
            # Convert bytes back to PIL Image
            image_data = response['data']  # Raw bytes from pickle
            return Image.open(io.BytesIO(image_data))
        
        return None
    
    def print_state(self, show_images: bool = False) -> None:
        """Print the current execution state."""
        state = self.get_state()
        
        if state:
            print("\n=== FINAL STATE ===")
            for key, info in state.items():
                print(f"\n{key}: {info['type']}")
                
                # Special handling for PIL Images
                if info['type'] == 'Image' and show_images:
                    image = self.get_result_image(int(key.split('_')[-1]) if 'image_clue_' in key else 0)
                    if image:
                        print(f"  Size: {image.size}, Mode: {image.mode}")
                        try:
                            import matplotlib.pyplot as plt
                            plt.figure()
                            plt.imshow(image)
                            plt.title(f"State Variable: {key}")
                            plt.axis('off')
                            plt.show()
                        except:
                            print(f"  Could not display image: {key}")
                else:
                    print(f"  {info['repr']}")
        else:
            print("\n=== FINAL STATE ===")
            print("(empty)")
    
    def __del__(self):
        """Clean up the subprocess when the runner is destroyed."""
        if self.process:
            try:
                self._send_command({'command': 'shutdown'})
                self.process.wait(timeout=5)
            except:
                self.process.terminate()
                self.process.wait(timeout=5)
            finally:
                if self.process.poll() is None:
                    self.process.kill()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Python code with state persistence via IPC")
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
    
    runner = IPCProgramRunner()
    
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