### High-Level Goal

Create an agent in LangGraph that takes a prompt including a visual task and a source image. The agent will iteratively generate and refine Python programs to solve the visual task. The agent can generate intermediate images for further refinement in the next iteration.

The generated code should be simple, efficient and follow Python best practices.

### Project Structure
- Keep the directory structure flat. Source and test code together.
- Test code should have the `_test.py` suffix.
- Create and update a pyproject.toml file as needed

### Tooling
- uv
- pytest
- mypy
- ruff
- langchain for Claude and Gemini

## Code Conventions
- Generate type hints for Python code
- Don't generate module comments

### Components
1. A main driver. Sets up the other components.
2. Program runner. Takes a snippet of Python code and evaluates in a sandbox that persists in-between iterations.
3. LangGraph agent

#### Main Driver
- Located in `main.py` with a __main__ section.
- Setup the other components.
- Kicks off the main agent loop.

### Program Runner
- Located in `runner.py`
- In each turn, the LLM autonomously generates a block of Python code (`code_block_i`), which is dynamically tailored to the task. This code is expected to be wrapped in `<code></code>` tags for reliable parsing
- This generated `code_block_i` is then executed within an isolated Python runtime.
- Process Isolation: Each code snippet is executed in a subprocess dynamically spawned by the main process
- Cross-Turn Persistence: The runtime environment is designed to retain variables and state across turns. This allows the LLM to reuse or modify intermediate Python code execution results from previous turns, for example, cropping an image first, then applying filters, and finally computing geometric features.
- Input images or video frames are pre-loaded as global variables named `image_clue_i` (where `i` is the image index) using `PIL.Image.open()`. The model can directly use these variables without needing to load them again. Image resolution is also provided to assist operations like cropping.
- The execution of the code produces multimodal results (`mm_clue_i`), which can be textual or visual.
- Textual results are generated using print() statements
- Visual results, such as image visualizations, are displayed using `plt.show()` (from Matplotlib), and there is no need to save them.

### LangGraph Agent
- Located in `agent.py`
- The agent loop should execute at most for a configurable `n` number of times.
- `i` is the index of the current iteration
- Support checkpointing.
- Create a function that converts the message history from the checkpoint state into context.
- Do not use create_react_agent
- The multimodal LLM the agent uses should be configurable.
- The agent takes a configurable system prompt.
- The agent takes a task-specific prompt and source image.

### Agent Steps
1. Initial Input and Context: The MLLM receives an input, which includes the user's query and any pre-loaded images or video frames, referenced as variables like `image_clue_i`. The MLLM is guided by a carefully constructed system prompt that encourages code generation for problem-solving

2. Code Generation: In a given turn (the i-th turn), the MLLM autonomously generates a block of Python code, referred to as `code_block_i`. This code is dynamically tailored to the task at hand. The system prompt specifies how to structure the code, including wrapping it in `<code></code>` tags

3. Code Execution: The generated Python code (`code_block_i`) is executed within an isolated Python runtime. Each code snippet is executed in a subprocess to ensure stability and prevent side effects from impacting the overall session. PyVision leverages Python's rich ecosystem of scientific and vision libraries (e.g., OpenCV, Pillow, NumPy, Pandas, Scikit-learn, Scikit-image) as building blocks for these dynamically generated tools.

4. Output Generation and Feedback: The execution of the code produces multimodal results, denoted as `mm_clue_i`. These results can be textual (e.g., from `print()` statements) or visual (e.g., from `plt.show()` for image visualizations). This output is then fed back into the MLLM's context.

5. Iterative Refinement: With the new output appended to its context, the MLLM uses this information to update and refine its reasoning for the next turn. The runtime environment maintains variables and state across turns, allowing the model to reuse or modify previous results. Communication between the runtime and the MLLM is handled through structured variable passing, rather than direct file system dependencies

6. Loop Continuation and Final Answer: This agentic loop continues, with the MLLM iteratively generating, executing, and refining its approach over multiple turns, until it automatically decides that the problem is solved and outputs a final answer. The final answer is expected to be enclosed in an `<answer></answer>` tag. There's also a maximum number of steps `n` before the agent gives up.