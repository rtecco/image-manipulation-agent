from typing import Optional
import argparse
import sys
from pathlib import Path
from PIL import Image

from agent import VisionAgent
from runner import ProgramRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Vision Agent for visual task processing")
    parser.add_argument("--task", required=True, help="Path to task description file")
    parser.add_argument("--image", required=True, help="Path to source image")
    parser.add_argument("--max-iterations", type=int, default=10, help="Maximum iterations")
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="Model to use")
    parser.add_argument("--checkpoint-dir", help="Directory for checkpointing")
    
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
    
    # Initialize components
    runner = ProgramRunner()
    agent = VisionAgent(
        runner=runner,
        model=args.model,
        max_iterations=args.max_iterations,
        checkpoint_dir=args.checkpoint_dir
    )
    
    try:
        # Run the agent
        result = agent.run(task=task, image=image)
        print(f"Final result: {result}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()