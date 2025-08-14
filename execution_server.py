#!/usr/bin/env python3
import pickle
import sys
import traceback
import io
import contextlib
import base64
from typing import Dict, Any, Optional
from PIL import Image
import numpy as np
import pandas as pd
# Optional imports - available if needed but not imported globally
# import wand
# import sklearn
# import skimage

class ExecutionServer:
    """Long-running Python execution server with IPC communication."""
    
    def __init__(self):
        self.namespace = {}
        self.base_imports = {
            'Image': Image,
            'np': np,
            'pd': pd,
            'io': io
        }
        self._setup_base_imports()
    
    def _setup_base_imports(self):
        """Set up base imports in the execution namespace."""
        self.namespace.update(self.base_imports)
    
    def execute_code(self, code: str) -> Dict[str, Any]:
        """
        Execute Python code in the persistent namespace.
        
        Returns:
            Dict with keys: success, stdout, stderr
        """
        # Capture stdout/stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            # Redirect output
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            
            # Execute code in persistent namespace
            exec(code, self.namespace)
            
            return {
                'success': True,
                'stdout': stdout_capture.getvalue(),
                'stderr': stderr_capture.getvalue(),
                'error': None
            }
            
        except Exception as e:
            # Capture full traceback
            error_traceback = traceback.format_exc()
            
            return {
                'success': False,
                'stdout': stdout_capture.getvalue(),
                'stderr': stderr_capture.getvalue() + error_traceback,
                'error': str(e)
            }
            
        finally:
            # Restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    
    def load_image(self, image_data: bytes, variable_name: str) -> Dict[str, Any]:
        """Load image data into the namespace."""
        try:
            image = Image.open(io.BytesIO(image_data))
            self.namespace[variable_name] = image
            
            return {
                'success': True,
                'message': f'Loaded image as {variable_name}',
                'error': None
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Failed to load image: {str(e)}',
                'error': str(e)
            }
    
    def get_variable(self, var_name: str) -> Dict[str, Any]:
        """Get a variable from the namespace."""
        try:
            if var_name in self.namespace:
                value = self.namespace[var_name]
                # Special handling for PIL Images
                if isinstance(value, Image.Image):
                    # Convert to bytes for transmission
                    buf = io.BytesIO()
                    value.save(buf, format='JPEG')
                    return {
                        'success': True,
                        'type': 'image',
                        'data': buf.getvalue(),  # Raw bytes with pickle
                        'size': value.size,
                        'mode': value.mode
                    }
                else:
                    return {
                        'success': True,
                        'type': type(value).__name__,
                        'data': str(value),
                        'repr': repr(value)
                    }
            else:
                return {
                    'success': False,
                    'error': f'Variable {var_name} not found'
                }
        except Exception as e:
            return {
                'success': False,
                'error': f'Error retrieving {var_name}: {str(e)}'
            }
    
    def list_variables(self) -> Dict[str, Any]:
        """List all variables in the namespace."""
        try:
            variables = {}
            for name, value in self.namespace.items():
                if not name.startswith('_') and not callable(value):
                    variables[name] = {
                        'type': type(value).__name__,
                        'repr': repr(value)[:100] + ('...' if len(repr(value)) > 100 else '')
                    }
            
            return {
                'success': True,
                'variables': variables
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Error listing variables: {str(e)}'
            }
    
    def clear_state(self) -> Dict[str, Any]:
        """Clear the execution namespace (except base imports)."""
        try:
            # Keep only base imports and built-ins
            keys_to_remove = [
                k for k in self.namespace.keys() 
                if not k.startswith('__') and k not in self.base_imports
            ]
            
            for key in keys_to_remove:
                del self.namespace[key]
            
            return {
                'success': True,
                'message': f'Cleared {len(keys_to_remove)} variables'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Error clearing state: {str(e)}'
            }
    
    def _read_message(self) -> Dict[str, Any]:
        """Read a base64-encoded pickled message from stdin."""
        try:
            line = sys.stdin.readline()
            if not line:
                raise EOFError("Could not read message")
            
            # Decode base64 and unpickle
            encoded_data = line.strip()
            if not encoded_data:
                raise ValueError("Empty message")
            message_data = base64.b64decode(encoded_data)
            return pickle.loads(message_data)
        except Exception as e:
            # Debug: write error to stderr
            print(f"Debug: Error reading message: {e}", file=sys.stderr)
            raise
    
    def run(self):
        """Main server loop - processes pickled messages from stdin."""
        while True:
            try:
                # Read message from stdin
                try:
                    message = self._read_message()
                except EOFError:
                    break  # EOF
                except Exception as e:
                    self._send_response({
                        'success': False,
                        'error': f'Invalid message: {str(e)}'
                    })
                    continue
                
                # Process message
                response = self._process_message(message)
                self._send_response(response)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                self._send_response({
                    'success': False,
                    'error': f'Server error: {str(e)}'
                })
    
    def _process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single IPC message."""
        command = message.get('command')
        
        if command == 'execute':
            code = message.get('code', '')
            return self.execute_code(code)
        
        elif command == 'load_image':
            image_data = message.get('image_data', b'')
            var_name = message.get('variable_name', 'image_clue_0')
            return self.load_image(image_data, var_name)
        
        elif command == 'get_variable':
            var_name = message.get('variable_name', '')
            return self.get_variable(var_name)
        
        elif command == 'list_variables':
            return self.list_variables()
        
        elif command == 'clear_state':
            return self.clear_state()
        
        elif command == 'ping':
            return {'success': True, 'message': 'pong'}
        
        elif command == 'shutdown':
            sys.exit(0)
        
        else:
            return {
                'success': False,
                'error': f'Unknown command: {command}'
            }
    
    def _send_response(self, response: Dict[str, Any]):
        """Send base64-encoded pickled response to stdout."""
        try:
            # Serialize and encode response
            response_data = pickle.dumps(response)
            encoded_data = base64.b64encode(response_data).decode('ascii')
            
            # Send with newline
            print(encoded_data, flush=True)
            
        except Exception as e:
            # Fallback response if serialization fails
            try:
                fallback = {
                    'success': False,
                    'error': f'Response serialization failed: {str(e)}'
                }
                fallback_data = pickle.dumps(fallback)
                encoded_fallback = base64.b64encode(fallback_data).decode('ascii')
                print(encoded_fallback, flush=True)
            except:
                pass  # Give up if even fallback fails


if __name__ == '__main__':
    server = ExecutionServer()
    server.run()