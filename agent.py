import re
import base64
import pprint
import io
from typing import Dict, Any, Optional, List, TypedDict
from PIL import Image
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage

from langgraph.graph.state import StateGraph, END, CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver

from runner import ProgramRunner

class EncodedImage(TypedDict):
    base64: str
    width: int
    height: int

class AgentState(TypedDict):
    messages: List[BaseMessage]
    task: str
    encoded_image: EncodedImage
    code_prompt_template: str
    plan_prompt_template: str
    iteration: int
    max_iterations: int
    final_answer: Optional[str]
    code_results: List[Dict[str, Any]]

def encode_image(image):
    """
    Convert a PIL.Image object or image file path to base64-encoded string, and get resolution info.
    
    Args:
        image: Can be a PIL.Image object or image file path.
    Returns:
        dict with keys:
        - 'base64': base64-encoded string
        - 'width': width in pixels
        - 'height': height in pixels
    """
    img_obj = None
    
    if isinstance(image, str):
        # Handle file path
        img_obj = Image.open(image)
        with open(image, "rb") as image_file:
            base64_str = base64.b64encode(image_file.read()).decode('utf-8')
    else:
        # Handle PIL.Image object
        img_obj = image
        buffered = io.BytesIO()
        image.save(buffered, format='jpeg')
        base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    width, height = img_obj.size
    
    return EncodedImage({
        'base64': base64_str,
        'width': width,
        'height': height
    })

def get_prompt_file(fn: str) -> str:
    path = Path(__file__).parent / fn
    return path.read_text().strip()

class VisionAgent:
    """LangGraph agent for visual task processing."""
    
    def __init__(
        self,
        runner: ProgramRunner,
        model: str = "claude-3-5-sonnet-20240620",
        temperature: float = 0.8,
        checkpoint_dir: Optional[str] = None,
    ) -> None:
        """Initialize the vision agent."""
        self.runner = runner
        
        # Initialize model
        if "claude" in model.lower():
            self.llm = ChatAnthropic(
                model_name=model, 
                timeout=None, 
                stop=None,
                temperature=temperature)
        elif "gemini" in model.lower():
            self.llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)
        else:
            raise ValueError(f"Unsupported model: {model}")

        self.plan_prompt_template = get_prompt_file("plan_prompt.md")
        self.code_prompt_template = get_prompt_file("code_prompt.md")
        
        # Setup checkpointing
        if checkpoint_dir:
            # TODO: Implement file-based checkpointing
            self.checkpointer: Optional[BaseCheckpointSaver] = MemorySaver()
        else:
            self.checkpointer = MemorySaver()

        # Build graph
        self.graph = self._build_graph()

    def _build_graph(self) -> CompiledStateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)

        workflow.add_node("generate_plan", self._generate_plan)        
        workflow.add_node("generate_code", self._generate_code)
        workflow.add_node("redo_generate_code", self._redo_generate_code)
        workflow.add_node("execute_code", self._execute_code)
        workflow.add_node("next_iteration", self._next_iteration)

        workflow.set_entry_point("generate_plan")
        
        workflow.add_edge("generate_plan", "generate_code")
        workflow.add_edge("generate_code", "execute_code")
        workflow.add_edge("redo_generate_code", "execute_code")
        workflow.add_edge("next_iteration", "generate_code")

        workflow.add_conditional_edges(
            "execute_code",
            self._route_next,
            {
                "redo_generate_code": "redo_generate_code",
                "next_iteration": "next_iteration",
                "end": END,
            }
        )
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    def _generate_plan(self, state: AgentState) -> Dict[str, Any]:
        plan_prompt = state["plan_prompt_template"].format(task=state["task"],width=state["encoded_image"]["width"],height=state["encoded_image"]["height"])
        plan_msg = HumanMessage(
            content = [
                 {"type": "text", "text": plan_prompt},
                 {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": state["encoded_image"]["base64"]}},
             ]
        )
        
        state["messages"].append(plan_msg)

        response = self.llm.invoke(state["messages"])
        
        updates = {
            "messages": state["messages"] + [response]
        }

        # Get the number of plan steps output by the LLM.
        steps_match = re.search(r'STEPS=(\d+)', response.content)

        if steps_match:
            max_steps = int(steps_match.group(1))
            updates["max_iterations"] = max_steps
            print(f"Updated max_iterations to {max_steps} based on plan")
        else:
            raise RuntimeError("Plan didn't output the number of steps")
        
        return updates
    
    def _generate_code(self, state: AgentState) -> Dict[str, Any]:        
        code_prompt = state["code_prompt_template"].format(i=state["iteration"])
        msg = HumanMessage(content=code_prompt)

        state["messages"].append(msg)

        response = self.llm.invoke(state["messages"])
                
        updates = {
            "messages": state["messages"] + [response],
        }

        return updates
    
    def _redo_generate_code(self, state: AgentState) -> Dict[str, Any]:        
        raise RuntimeError()

    def _execute_code(self, state: AgentState) -> Dict[str, Any]:
        """Execute the generated code."""

        last_message = state["messages"][-1]
        code = self._extract_code(last_message.content)
        
        updates = {}
        
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
            updates["code_results"] = state["code_results"] + [result]
            
            # Add execution results to messages
            if stdout or stderr:
                execution_msg = f"Execution results:\nStdout: {stdout}\nStderr: {stderr}"
                updates["messages"] = state["messages"] + [HumanMessage(content=execution_msg)]
        
        return updates

    def _next_iteration(self, state: AgentState) -> Dict[str, Any]:
        last_result = state["code_results"][-1]
        results = "Iteration {i} generated this code: {code}\nIt printed the following: {stdout}".format(i=state["iteration"],code=last_result["code"],stdout=last_result["stdout"])

        msg = HumanMessage(content=results)

        state["messages"].append(msg)

        return {
            "message": state["messages"],
            "iteration": state["iteration"] + 1
        }
    
    def _route_next(self, state: AgentState) -> str:

        # Time's up
        if state["iteration"] >= state["max_iterations"]:
            return "end"
        
        # Re-generate bad code
        if state["code_results"]:
            last_result = state["code_results"][-1]
            if not last_result["success"]:
                return "redo_generate_code"

        # Next iteration
        return "next_iteration"
    
    def _extract_code(self, content: str) -> Optional[str]:
        """Extract code from <code></code> tags."""
        pattern = r'<code>(.*?)</code>'
        match = re.search(pattern, content, re.DOTALL)
        return match.group(1).strip() if match else None
    
    def run(self, task: str, image: Image.Image) -> str:
        """Run the agent on a task with an image."""

        # Load original image into runner
        self.runner.load_image_clue(image, 0)
        
        encoded_image = encode_image(image)

        # Initialize state
        initial_state: AgentState = {
            "messages": [],
            "task": task,
            "encoded_image": encoded_image,
            "iteration": 1,
            "max_iterations": -1,
            "final_answer": None,
            "code_results": [],
            "code_prompt_template": self.code_prompt_template,
            "plan_prompt_template": self.plan_prompt_template
        }
        
        # Run the graph with thread configuration
        config = {"configurable": {"thread_id": "main-thread"}}
        final_state = self.graph.invoke(initial_state, config=config)

        result = self.runner.get_result_image(final_state["max_iterations"])

        if not result:
            raise RuntimeError("no final result")

        result.show()

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