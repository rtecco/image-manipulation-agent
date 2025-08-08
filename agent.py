import re
import base64
import io
import yaml
from typing import Dict, Any, Optional, List, TypedDict
from PIL import Image
from pathlib import Path
from datetime import datetime

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage

from token_rate_limiter import TokenAwareRateLimiter, TokenAwareChatAnthropic

from langgraph.graph.state import StateGraph, END, CompiledStateGraph

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
        config_path: str
    ) -> None:
        """Initialize the vision agent."""
        self.runner = runner
        
        # Load config
        config = self._load_config(config_path)
        
        # Initialize model
        model = config["model"]["name"]
        temperature = config["model"]["temperature"]
        max_retries = config["model"]["max_retries"]
        tokens_per_minute = config["rate_limiting"]["tokens_per_minute"]
        max_burst_tokens=config["rate_limiting"]["max_burst_tokens"]

        if "claude" in model.lower():
            # Initialize token-aware rate limiter
            token_rate_limiter = TokenAwareRateLimiter(
                tokens_per_minute=tokens_per_minute,
                max_burst_tokens=max_burst_tokens,
            )
            
            self.llm = TokenAwareChatAnthropic(
                token_rate_limiter=token_rate_limiter,
                model_name=model,
                max_retries=max_retries,
                timeout=None, 
                stop=None,
                temperature=temperature)
        elif "gemini" in model.lower():
            self.llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)
        else:
            raise ValueError(f"Unsupported model: {model}")

        # Load prompts
        prompts = config["prompts"]
        self.plan_prompt_template = self._load_prompt(prompts["plan_prompt"], prompts.get("plan_prompt_file"))
        self.code_prompt_template = self._load_prompt(prompts["code_prompt"], prompts.get("code_prompt_file"))
        
        # Store execution config
        self.execution_config = config["execution"]
        
        # Build graph
        self.graph = self._build_graph()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_prompt(self, inline_prompt: Optional[str], prompt_file: Optional[str]) -> str:
        """Load prompt from inline text or file."""
        if inline_prompt:
            return inline_prompt.strip()
        elif prompt_file:
            return Path(prompt_file).read_text().strip()
        else:
            raise ValueError("Must specify either inline prompt or prompt file")

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
        workflow.add_edge("redo_generate_code", "generate_code")
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
        
        return workflow.compile()
    
    def _generate_plan(self, state: AgentState) -> Dict[str, Any]:
        plan_prompt = state["plan_prompt_template"].format(task=state["task"],width=state["encoded_image"]["width"],height=state["encoded_image"]["height"])
        
        encoded_source_image_msg = HumanMessage(content=[{
            "type": "image", 
            "source": {
                "type": "base64", 
                "media_type": "image/jpeg", 
                "data": state["encoded_image"]["base64"]
            }
        }])

        plan_msg = HumanMessage(content=plan_prompt)
        state["messages"].append(plan_msg)

        response = self.llm.invoke(state["messages"] + [encoded_source_image_msg])

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
        
        # Get the runner image from the previous iteration
        runner_image = self.runner.get_result_image(state["iteration"] - 1)
        
        if not runner_image:
            raise RuntimeError(f"Expected image_clue_{state['iteration'] - 1} from previous iteration but not found")
        
        # Inclue the last image in this LLM invocation only not the conversation
        encoded_runner_image = encode_image(runner_image)
        encoded_runner_image_msg = HumanMessage(content=[{
                "type": "image", 
                "source": {
                    "type": "base64", 
                    "media_type": "image/jpeg", 
                    "data": encoded_runner_image["base64"]
                }
            }])

        msg = HumanMessage(content=code_prompt)

        state["messages"].append(msg)

        response = self.llm.invoke(state["messages"] + [encoded_runner_image_msg])
                
        updates = {
            "messages": state["messages"] + [response],
        }

        return updates
    
    def _redo_generate_code(self, state: AgentState) -> Dict[str, Any]:
        last_result = state["code_results"][-1]
        results = "Iteration {i} generated this code: {code}\nIt failed with the following error {stderr}. Please fix the error. It is still iteration {i}.".format(i=state["iteration"],code=last_result["code"],stderr=last_result["stderr"])

        msg = HumanMessage(content=results)
        print(msg)
        state["messages"].append(msg)

        return {
            "messages": state["messages"]
        }
    
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
        
        return updates

    def _next_iteration(self, state: AgentState) -> Dict[str, Any]:
        last_result = state["code_results"][-1]
        results = "Iteration {i} generated this code: {code}\nIt ran successfully printing the following: {stdout}".format(i=state["iteration"],code=last_result["code"],stdout=last_result["stdout"])

        msg = HumanMessage(content=results)

        state["messages"].append(msg)

        return {
            "messages": state["messages"],
            "iteration": state["iteration"] + 1
        }
    
    def _route_next(self, state: AgentState) -> str:
        
        # Re-generate bad code
        if state["code_results"]:
            last_result = state["code_results"][-1]
            if not last_result["success"]:
                return "redo_generate_code"

        # Time's up
        if state["iteration"] >= state["max_iterations"]:
            return "end"
        
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
        
        # Run the graph with increased recursion limit
        config = {"recursion_limit": self.execution_config["recursion_limit"]}
        final_state = None
        error_message = None
        
        try:
            final_state = self.graph.invoke(initial_state, config=config)
        except Exception as e:
            error_message = str(e)
            print(f"❌ Graph execution failed: {error_message}")
            # Use the initial state as fallback for reporting
            final_state = initial_state
            final_state["_execution_error"] = error_message

        # Create combined state with both agent and runner data
        combined_state = self._create_combined_state(final_state)
        
        # Add error info to combined state if execution failed
        if error_message:
            combined_state["_execution_error"] = error_message
            combined_state["_completed"] = False
        
        # Save combined state for dashboard
        state_path = self._save_combined_state(combined_state)
        print(f"💾 State saved: {state_path}")

        # Generate Streamlit report using combined state
        report_path = self.generate_streamlit_report(combined_state)
        print(f"📊 Report generated: {report_path}")
        
        if error_message:
            return f"Task failed: {error_message}. Report: {report_path}"
        else:
            return f"Task completed. Report: {report_path}"

    def _create_combined_state(self, final_state: AgentState) -> Dict[str, Any]:
        """Create combined state with both agent and runner data."""        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Combine agent state and runner state
        combined_state = {
            **final_state,  # All agent state (messages, code_results, etc.)
            **self.runner.get_state(),  # All runner state (image_clue_n, etc.)
            "_timestamp": timestamp,
            "_task": final_state["task"],
            "_completed": True
        }
        
        return combined_state

    def _save_combined_state(self, combined_state: Dict[str, Any]) -> Path:
        """Save combined state to file for dashboard access."""
        import tempfile
        import dill
        
        timestamp = combined_state["_timestamp"]
        
        # Save to temp directory with consistent naming
        state_path = Path(tempfile.gettempdir()) / f"final_state_{timestamp}.pkl"
        with open(state_path, 'wb') as f:
            dill.dump(combined_state, f)
        
        return state_path

    def generate_streamlit_report(self, combined_state: Dict[str, Any]) -> str:
        """Generate a standalone Streamlit report using combined state."""
        import base64
        import io
        
        # Use existing timestamp from combined state
        timestamp = combined_state["_timestamp"]
        report_path = f"report_{timestamp}.py"
        
        # Extract data from combined state (includes both agent and runner state)
        task = combined_state.get("_task", combined_state.get("task", "Unknown Task"))
        images = {}
        
        # Get all image_clue_n from the combined state
        for key, value in combined_state.items():
            if key.startswith("image_clue_") and hasattr(value, 'save'):
                try:
                    n = int(key.split("_")[-1])
                    # Convert image to base64 for embedding
                    buffer = io.BytesIO()
                    value.save(buffer, format='PNG')
                    img_b64 = base64.b64encode(buffer.getvalue()).decode()
                    images[n] = img_b64
                except (ValueError, AttributeError):
                    continue
        
        # Generate enhanced Streamlit script
        script = f'''"""Generated Streamlit report for Vision Agent run - {timestamp}"""
import streamlit as st
import base64
from io import BytesIO
from PIL import Image

st.set_page_config(page_title="Vision Agent Report", layout="wide")

st.title("🤖 Vision Agent Report")
st.markdown("---")

# Task info
st.subheader("📋 Task")
st.info("{task}")
st.write("**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")

# Show execution status
execution_error = {repr(combined_state.get("_execution_error"))}
if execution_error:
    st.error(f"⚠️ **Execution Failed:** {{execution_error}}")
else:
    st.success("✅ **Execution Completed Successfully**")

# Tabs for different views
tab1, tab2, tab3 = st.tabs(["🖼️ Images", "💻 Code", "📊 Summary"])

with tab1:
    images = {repr(images)}
    if images:
        st.markdown(f"**{{len(images)}} images generated:**")
        
        # Image selector
        image_keys = sorted(images.keys())
        selected_img = st.selectbox("Select image:", image_keys, 
                                  format_func=lambda x: f"Step {{x}}")
        
        # Display selected image
        img_data = base64.b64decode(images[selected_img])
        img = Image.open(BytesIO(img_data))
        st.image(img, caption=f"Step {{selected_img}}", use_container_width=True)
        
        # Thumbnail gallery
        if len(images) > 1:
            st.markdown("**All steps:**")
            cols = st.columns(min(len(images), 4))
            for i, n in enumerate(image_keys):
                with cols[i % 4]:
                    img_data = base64.b64decode(images[n])
                    img = Image.open(BytesIO(img_data))
                    st.image(img, caption=f"Step {{n}}", use_container_width=True)
    else:
        st.info("No images generated in this run")

with tab2:
    code_results = {repr(combined_state.get("code_results", []))}
    if code_results:
        st.markdown("**Code execution history:**")
        
        for i, result in enumerate(code_results, 1):
            status_emoji = "✅" if result.get('success') else "❌"
            
            with st.expander(f"{{status_emoji}} Iteration {{i}}", 
                           expanded=(i == len(code_results))):
                st.markdown("**Code:**")
                st.code(result.get("code", ""), language="python")
                
                if result.get("stdout") or result.get("stderr"):
                    col1, col2 = st.columns(2)
                    with col1:
                        if result.get("stdout"):
                            st.markdown("**Output:**")
                            st.success(result["stdout"])
                    with col2:
                        if result.get("stderr"):
                            st.markdown("**Error:**")
                            st.error(result["stderr"])
    else:
        st.info("No code execution results")

with tab3:
    st.markdown("**Run Statistics:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Iterations", len(code_results))
    with col2:
        success_count = sum(1 for r in code_results if r.get('success'))
        st.metric("Successful", success_count)
    with col3:
        st.metric("Images Created", len(images))
    
    if code_results:
        success_rate = success_count / len(code_results) * 100
        st.markdown(f"**Success Rate:** {{success_rate:.1f}}%")
'''
        
        with open(report_path, 'w') as f:
            f.write(script)
        
        return report_path

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