#!/usr/bin/env python3

from agent import VisionAgent
from runner import ProgramRunner

if __name__ == "__main__":
    runner = ProgramRunner()
    agent = VisionAgent(runner=runner)
    
    agent.visualize_graph()