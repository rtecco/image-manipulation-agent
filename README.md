### Installation (macOS)

`brew install mise`

`mise use uv`

`uv sync`

### Running Agent

`source ./venv/bin/activate[.fish]`

`ANTHROPIC_API_KEY=<yr key> python main.py --image <path to source image> --config <path to experiment YAML>`

### Running the Streamlit Dashboard

To see the output of experiment runs

`streamlit run dashboard.py`
