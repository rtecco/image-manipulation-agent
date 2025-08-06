"""Enhanced Streamlit dashboard for browsing Vision Agent runs."""

import streamlit as st
from report_utils import find_state_files, get_state_info, get_images_from_state, extract_plan

st.set_page_config(page_title="Vision Agent Dashboard", layout="wide", initial_sidebar_state="expanded")

# Header
st.title("🤖 Vision Agent Dashboard")
st.markdown("---")

# Sidebar for controls
with st.sidebar:
    st.header("Controls")
    
    # Refresh button
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()
    
    # Auto-refresh toggle
    auto_refresh = st.checkbox("Auto-refresh (30s)")
    
    # Filter options
    st.header("Filters")
    show_failed = st.checkbox("Show failed runs", value=True)
    show_successful = st.checkbox("Show successful runs", value=True)

# Find and load all runs
state_files = find_state_files()

if not state_files:
    st.warning("No vision agent runs found. Run a task first!")
    st.info("Runs will appear here after you execute vision agent tasks.")
    st.stop()

# Get info for all runs and apply filters
runs = [get_state_info(f) for f in state_files]
runs.sort(key=lambda r: r["modified"], reverse=True)

# Apply filters
filtered_runs = []
for run in runs:
    if run["success"] and show_successful:
        filtered_runs.append(run)
    elif not run["success"] and show_failed:
        filtered_runs.append(run)

st.markdown(f"**{len(filtered_runs)}** runs found (of {len(runs)} total)")

# Main layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📋 Recent Runs")
    
    if not filtered_runs:
        st.info("No runs match current filters")
        st.stop()
    
    # Run selection with better UI
    selected_idx = None
    
    for i, run in enumerate(filtered_runs):
        status_emoji = "✅" if run["success"] else "❌"
        time_str = run['modified'].strftime('%m/%d %H:%M')
        
        # Create a nice card-like button
        button_text = f"{status_emoji} **{time_str}**\n\n{run['task'][:50]}..."
        
        if st.button(
            button_text,
            key=f"run_{i}",
            use_container_width=True,
            help=f"Task: {run['task']}\nCode Executions: {run['code_executions']}\nIterations: {run['iterations']}/{run['max_iterations']}\nCompleted: {run['completed']}"
        ):
            selected_idx = i
        
        # Add some spacing
        st.markdown("")

with col2:
    if selected_idx is not None:
        run = filtered_runs[selected_idx]
        state = run["state"]
        
        # Header with status
        status_color = "green" if run["success"] else "red"
        st.markdown(f"""
        ### 🔍 Run Analysis
        **Date:** {run['modified'].strftime('%Y-%m-%d %H:%M:%S')}  
        **Status:** :{status_color}[{'✅ Success' if run['success'] else '❌ Failed'}]  
        **Code Executions:** {run['code_executions']} | **Iterations:** {run['iterations']}/{run['max_iterations']}
        """)
        
        # Task description
        st.markdown("**Task Description:**")
        st.info(run['task'])
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["🖼️ Images", "💻 Code", "🔍 Messages"])
        
        with tab1:
            images = get_images_from_state(state)
            if images:
                st.markdown(f"**{len(images)} images generated during execution:**")
                
                # Display all images vertically
                for n in sorted(images.keys()):
                    st.image(images[n], caption=f"Step {n}", use_container_width=True)
            else:
                st.info("No images found in this run")
        
        with tab2:
            if "code_results" in state and state["code_results"]:
                st.markdown("**Code execution history:**")
                
                # Count attempts per iteration
                iteration_attempts = {}
                for result in state["code_results"]:
                    iteration_num = result.get('iteration', 0)
                    iteration_attempts[iteration_num] = iteration_attempts.get(iteration_num, 0) + 1
                
                # Reset counter for display
                iteration_attempt_count = {}
                
                for i, result in enumerate(state["code_results"], 1):
                    status_emoji = "✅" if result['success'] else "❌"
                    iteration_num = result.get('iteration', 0)
                    
                    # Track attempt number for this iteration
                    iteration_attempt_count[iteration_num] = iteration_attempt_count.get(iteration_num, 0) + 1
                    attempt_num = iteration_attempt_count[iteration_num]
                    
                    with st.expander(f"{status_emoji} Attempt {attempt_num} for Iteration {iteration_num}", 
                                   expanded=(i == len(state["code_results"]))):  # Expand last iteration
                        
                        # Code
                        st.markdown("**Code:**")
                        st.code(result["code"], language="python")
                        
                        # Results in columns
                        if result.get("stdout") or result.get("stderr"):
                            stdout_col, stderr_col = st.columns(2)
                            
                            with stdout_col:
                                if result.get("stdout"):
                                    st.markdown("**Output:**")
                                    st.success(result["stdout"])
                            
                            with stderr_col:
                                if result.get("stderr"):
                                    st.markdown("**Errors:**")
                                    st.error(result["stderr"])
            else:
                st.info("No code execution results found")
        
        with tab3:
            if "messages" in state and state["messages"]:
                st.markdown(f"**{len(state['messages'])} messages in conversation:**")
                
                for i, msg in enumerate(state["messages"]):
                    msg_type = "🤖 AI" if hasattr(msg, '__class__') and "AI" in str(type(msg)) else "👤 Human"
                    
                    with st.expander(f"{msg_type} Message {i+1}", expanded=(i < 2)):  # Expand first 2
                        if hasattr(msg, 'content'):
                            content = str(msg.content)
                            
                            # Try to parse and pretty print JSON content
                            import json
                            try:
                                # Check if content looks like JSON
                                if content.strip().startswith(('[', '{')):
                                    parsed_json = json.loads(content)
                                    st.markdown("**Pretty JSON:**")
                                    st.json(parsed_json)
                                else:
                                    # Regular text content
                                    if len(content) > 1000:
                                        st.text_area("Content", content, height=200)
                                    else:
                                        st.code(content, language="text")
                            except json.JSONDecodeError:
                                # Not valid JSON, display as regular text
                                if len(content) > 1000:
                                    st.text_area("Content", content, height=200)
                                else:
                                    st.code(content, language="text")
                        else:
                            st.write(str(msg))
            else:
                st.info("No messages found in this run")
    
    else:
        # Welcome message when nothing is selected
        st.markdown("""
        ### 👋 Welcome to Vision Agent Dashboard
        
        Select a run from the left panel to view detailed analysis including:
        
        - 📋 **Generated Plans** - See how the AI planned the task
        - 🖼️ **Image Results** - View all generated images step by step  
        - 💻 **Code Execution** - Inspect generated code and outputs
        - 🔍 **Full Messages** - Browse the complete conversation
        
        Use the sidebar filters to show/hide successful or failed runs.
        """)

# Auto-refresh functionality
if auto_refresh:
    import time
    time.sleep(30)
    st.rerun()