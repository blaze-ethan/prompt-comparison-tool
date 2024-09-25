import streamlit as st
from openai import OpenAI
from anthropic import Anthropic
import os
import csv
import io
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize session state for responses
if 'responses' not in st.session_state:
    st.session_state.responses = []

# Function to get API keys
def get_api_keys():
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not openai_key:
        openai_key = st.text_input("Enter your OpenAI API key", type="password")
    if not anthropic_key:
        anthropic_key = st.text_input("Enter your Anthropic API key", type="password")
    
    return openai_key, anthropic_key

# Set up API clients
openai_key, anthropic_key = get_api_keys()
if openai_key and anthropic_key:
    openai_client = OpenAI(api_key=openai_key)
    anthropic_client = Anthropic(api_key=anthropic_key)
else:
    st.error("Please provide both OpenAI and Anthropic API keys to continue.")
    st.stop()

def generate_response(system_prompt, user_prompt, model, params):
    try:
        if model.startswith("gpt"):
            response = openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=params['temperature'],
                max_tokens=params['max_tokens'],
                top_p=params['top_p']
            )
            return response.choices[0].message.content
        elif model.startswith("claude"):
            response = anthropic_client.messages.create(
                model=model,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                temperature=params['temperature'],
                max_tokens=params['max_tokens']
            )
            return response.content[0].text
    except Exception as e:
        return f"Error: {str(e)}"

def export_to_csv(data):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Prompt Pair', 'Model', 'System Prompt', 'User Prompt', 'Parameters', 'Response'])
    for i, ((system_prompt, user_prompt, model, params), response) in enumerate(data, 1):
        writer.writerow([
            f'Prompt Pair {i}',
            model,
            system_prompt,
            user_prompt,
            ', '.join([f'{k}: {v}' for k, v in params.items()]),
            response
        ])
    return output.getvalue()

st.title("Multi-Model Prompt Comparison Tool")

# Input for number of prompt pairs
num_pairs = st.number_input("Number of prompt pairs to compare", min_value=1, max_value=5, value=2)

# Define available models
models = [
    "gpt-4o-mini", "gpt-4o", "gpt-4", "gpt-3.5-turbo",
    "claude-3-5-sonnet-20240620", "claude-3-opus-20240229",
    "claude-3-sonnet-20240229", "claude-3-haiku-20240307"
]

# Create a list to store all prompt pairs and their selected models
prompt_pairs = []

for i in range(num_pairs):
    st.subheader(f"Prompt Pair {i+1}")
    
    # Model selector for each prompt pair
    model = st.selectbox(
        f"Select model for Prompt Pair {i+1}",
        models,
        key=f"model_select_{i}"
    )
    
    # Model-specific parameters
    params = {}
    if model.startswith("gpt"):
        params['temperature'] = st.slider(f"Temperature (Pair {i+1})", 0.0, 2.0, 1.0, 0.1, key=f"temp_{i}")
        params['max_tokens'] = st.slider(f"Maximum Tokens (Pair {i+1})", 1, 4096, 2048, 1, key=f"max_tokens_{i}")
        params['top_p'] = st.slider(f"Top P (Pair {i+1})", 0.0, 1.0, 1.0, 0.1, key=f"top_p_{i}")
    elif model.startswith("claude"):
        params['temperature'] = st.slider(f"Temperature (Pair {i+1})", 0.0, 1.0, 0.0, 0.1, key=f"temp_{i}")
        params['max_tokens'] = st.slider(f"Max tokens to sample (Pair {i+1})", 1, 8192, 4096, 1, key=f"max_tokens_{i}")
    
    system_prompt = st.text_area(f"System Prompt {i+1}", key=f"system_input_{i}")
    user_prompt = st.text_area(f"User Prompt {i+1}", key=f"user_input_{i}")
    prompt_pairs.append((system_prompt, user_prompt, model, params))

# Generate All Responses button
if st.button("Generate All Responses"):
    st.session_state.responses = []
    for system_prompt, user_prompt, model, params in prompt_pairs:
        response = generate_response(system_prompt, user_prompt, model, params)
        st.session_state.responses.append(response)

# Display responses if they exist
if st.session_state.responses:
    st.subheader("Comparison of Responses")
    cols = st.columns(len(st.session_state.responses))
    for i, (response, (system_prompt, user_prompt, model, params)) in enumerate(zip(st.session_state.responses, prompt_pairs)):
        with cols[i]:
            st.markdown(f"**Prompt Pair {i+1} (Model: {model})**")
            with st.expander("System Prompt", expanded=True):
                st.text_area("", system_prompt, height=100, disabled=True, key=f"system_display_{i}")
            with st.expander("User Prompt", expanded=True):
                st.text_area("", user_prompt, height=50, disabled=True, key=f"user_display_{i}")
            with st.expander("Parameters", expanded=True):
                for key, value in params.items():
                    st.write(f"{key}: {value}")
            with st.expander("Response", expanded=True):
                st.markdown(response)
            st.markdown("---")  # Add a separator between pairs

    # Export to CSV option
    csv_data = export_to_csv(zip(prompt_pairs, st.session_state.responses))
    st.download_button(
        label="Download results as CSV",
        data=csv_data,
        file_name="prompt_comparison_results.csv",
        mime="text/csv",
    )

# Add layout
st.markdown("""
<style>
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stExpander > div:first-child {
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)