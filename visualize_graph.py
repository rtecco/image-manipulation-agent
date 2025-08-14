#!/usr/bin/env python3

from agent import VisionAgent
from ipc_runner import IPCProgramRunner

if __name__ == "__main__":
    runner = IPCProgramRunner()
    agent = VisionAgent(runner=runner)
    
    agent.visualize_graph()