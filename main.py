from typing import Optional
import argparse
import sys
from pathlib import Path
from PIL import Image

from agent import VisionAgent
from ipc_runner import IPCProgramRunner

def main() -> None:
    parser = argparse.ArgumentParser(description="Vision Agent for visual task processing")
    parser.add_argument("--task", required=True, help="Path to task description file")
    parser.add_argument("--image", required=True, help="Path to source image")
    parser.add_argument("--config",required=True, help="Path to experiment config file")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                        help="Set logging level")
    
    args = parser.parse_args()
    
    # Load task from file
    task_file = Path(args.task)
    if not task_file.exists():
        print(f"Error: Task file {args.task} not found", file=sys.stderr)
        sys.exit(1)
    
    task = task_file.read_text().strip()
    
    # Load image
    try:
        image = Image.open(args.image)
    except Exception as e:
        print(f"Error loading image {args.image}: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Validate config file exists
    config_file = Path(args.config)
    if not config_file.exists():
        print(f"Error: Config file {args.config} not found", file=sys.stderr)
        sys.exit(1)
    
    runner = IPCProgramRunner()
    agent = VisionAgent(
        runner=runner,
        config_path=args.config,
        log_level=args.log_level
    )
    
    result = agent.run(task=task, image=image)
    print(f"Final result: {result}")


if __name__ == "__main__":
    main()