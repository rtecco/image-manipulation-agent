"""LangGraph agent for visual task processing with iterative code generation."""

import re
from typing import Dict, Any, Optional, List, TypedDict
from PIL import Image
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph.state import StateGraph, END, CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver

from runner import ProgramRunner


class AgentState(TypedDict):
    """State for the vision agent."""
    messages: List[BaseMessage]
    task: str
    image: Image.Image
    iteration: int
    max_iterations: int
    final_answer: Optional[str]
    code_results: List[Dict[str, Any]]


class VisionAgent:
    """LangGraph agent for visual task processing."""
    
    def __init__(
        self,
        runner: ProgramRunner,
        model: str = "claude-sonnet-4-20250514",
        max_iterations: int = 10,
        checkpoint_dir: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> None:
        """Initialize the vision agent."""
        self.runner = runner
        self.max_iterations = max_iterations
        
        # Initialize model
        if "claude" in model.lower():
            self.llm = ChatAnthropic(model_name=model, timeout=None, stop=None)
        elif "gemini" in model.lower():
            self.llm = ChatGoogleGenerativeAI(model=model)
        else:
            raise ValueError(f"Unsupported model: {model}")
        
        # Default system prompt
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        
        # Setup checkpointing
        if checkpoint_dir:
            # TODO: Implement file-based checkpointing
            self.checkpointer: Optional[BaseCheckpointSaver] = MemorySaver()
        else:
            self.checkpointer = MemorySaver()
            
        # Build graph
        self.graph = self._build_graph()
    
    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for code generation."""
        prompt_path = Path(__file__).parent / "system-prompt.md"
        return prompt_path.read_text().strip()

    def _build_graph(self) -> CompiledStateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        workflow.add_node("generate_code", self._generate_code)
        workflow.add_node("execute_code", self._execute_code)
        workflow.add_node("check_completion", self._check_completion)
        
        workflow.set_entry_point("generate_code")
        
        workflow.add_edge("generate_code", "execute_code")
        workflow.add_edge("execute_code", "check_completion")
        
        workflow.add_conditional_edges(
            "check_completion",
            self._should_continue,
            {
                "continue": "generate_code",
                "end": END
            }
        )
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    def _generate_code(self, state: AgentState) -> AgentState:
        """Generate Python code for the current iteration."""
        # Create context from message history
        context = self._build_context(state)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", context)
        ])
        
        response = self.llm.invoke(prompt.format_messages())
        
        state["messages"].append(response)
        state["iteration"] += 1
        
        return state
    
    def _execute_code(self, state: AgentState) -> AgentState:
        """Execute the generated code."""
        last_message = state["messages"][-1]
        
        # Extract code from the message
        code = self._extract_code(last_message.content)
        
        if code:
            # Execute the code
            stdout, stderr, success = self.runner.execute_code(code)
            
            # Store results
            result = {
                "code": code,
                "stdout": stdout,
                "stderr": stderr,
                "success": success,
                "iteration": state["iteration"]
            }
            state["code_results"].append(result)
            
            # Add execution results to messages
            if stdout or stderr:
                execution_msg = f"Execution results:\nStdout: {stdout}\nStderr: {stderr}"
                state["messages"].append(HumanMessage(content=execution_msg))
        
        return state
    
    def _check_completion(self, state: AgentState) -> AgentState:
        """Check if the task is completed."""
        last_message = state["messages"][-1]
        
        # Check for final answer
        final_answer = self._extract_answer(last_message.content)
        if final_answer:
            state["final_answer"] = final_answer
            
        return state
    
    def _should_continue(self, state: AgentState) -> str:
        """Determine if the agent should continue."""
        if state["final_answer"]:
            return "end"
        if state["iteration"] >= state["max_iterations"]:
            return "end"
        return "continue"
    
    def _build_context(self, state: AgentState) -> str:
        """Build context from message history and current state."""
        context_parts = [
            f"Task: {state['task']}",
            f"Iteration: {state['iteration']}/{state['max_iterations']}",
            "Available image variables: image_clue_0"
        ]
        
        # Add previous execution results
        if state["code_results"]:
            context_parts.append("\nPrevious execution results:")
            for result in state["code_results"][-3:]:  # Last 3 results
                context_parts.append(f"Code: {result['code'][:100]}...")
                if result['stdout']:
                    context_parts.append(f"Output: {result['stdout'][:200]}")
                if result['stderr']:
                    context_parts.append(f"Error: {result['stderr'][:200]}")
        
        return "\n".join(context_parts)
    
    def _extract_code(self, content: str) -> Optional[str]:
        """Extract code from <code></code> tags."""
        pattern = r'<code>(.*?)</code>'
        match = re.search(pattern, content, re.DOTALL)
        return match.group(1).strip() if match else None
    
    def _extract_answer(self, content: str) -> Optional[str]:
        """Extract final answer from <answer></answer> tags."""
        pattern = r'<answer>(.*?)</answer>'
        match = re.search(pattern, content, re.DOTALL)
        return match.group(1).strip() if match else None
    
    def run(self, task: str, image: Image.Image) -> str:
        """Run the agent on a task with an image."""
        # Load image into runner
        self.runner.load_image(image, 0)
        
        # Initialize state
        initial_state: AgentState = {
            "messages": [HumanMessage(content=task)],
            "task": task,
            "image": image,
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "final_answer": None,
            "code_results": []
        }
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        # Return final answer or last result
        if final_state["final_answer"]:
            return final_state["final_answer"]
        elif final_state["code_results"]:
            last_result = final_state["code_results"][-1]
            return last_result.get("stdout", "No output generated")
        else:
            return "No solution found"
    
    def visualize_graph(self, output_path: Optional[str] = None) -> None:
        """Generate a visualization of the workflow graph."""
        try:
            # Get the drawable graph
            graph_image = self.graph.get_graph().draw_mermaid_png()
            
            if output_path:
                with open(output_path, 'wb') as f:
                    f.write(graph_image)
                print(f"Graph saved to {output_path}")
            else:
                # Display inline if possible
                from PIL import Image
                import io
                img = Image.open(io.BytesIO(graph_image))
                img.show()
                
        except ImportError:
            print("Graph visualization requires optional dependencies. Install with: uv pip install -e .[viz]")
        except Exception as e:
            print(f"Error generating graph visualization: {e}")