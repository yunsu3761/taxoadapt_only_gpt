"""
Web Interface for TaxoAdapt using Streamlit
Run with: streamlit run web_interface.py
"""
import streamlit as st
import os
import json
import csv
from pathlib import Path
from config_manager import DimensionConfig, load_biology_preset
import subprocess
import sys
from dotenv import load_dotenv, set_key, find_dotenv

# Get base directory
BASE_DIR = Path(__file__).parent.resolve()

# Load .env file
load_dotenv()

# Page config
st.set_page_config(
    page_title="TaxoAdapt Interface",
    page_icon="🔬",
    layout="wide"
)

# Initialize session state
if 'config' not in st.session_state:
    st.session_state.config = None
if 'dimensions' not in st.session_state:
    st.session_state.dimensions = {}
if 'running' not in st.session_state:
    st.session_state.running = False
if 'custom_dimensions' not in st.session_state:
    st.session_state.custom_dimensions = {}

def main():
    st.title("🔬 TaxoAdapt - Taxonomy Generation Framework")
    st.markdown("---")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # ============ 1. API Keys ============
        st.subheader("1. API Key")
        
        # Get default values from .env file
        default_openai_key = os.getenv('OPENAI_API_KEY', '')
        
        # Show current .env file status
        env_path = os.path.join(BASE_DIR, '.env')
        if os.path.exists(env_path):
            st.success("✅ .env file found")
        else:
            st.warning("⚠️ No .env file found. Key will be used temporarily for this session.")
        
        openai_api_key = st.text_input(
            "OpenAI API Key:",
            value=default_openai_key if default_openai_key else "",
            type="password",
            help="Enter your OpenAI API key for GPT models. This will be loaded from .env if available.",
            key="openai_api_key_input"
        )
        
        # Placeholder for compatibility (not used)
        huggingface_token = None
        
        # Button to save keys to .env file
        col_save1, col_save2 = st.columns([1, 1])
        with col_save1:
            if st.button("💾 Save to .env", help="Save API key to .env file"):
                try:
                    # Ensure .env exists
                    if not os.path.exists(env_path):
                        with open(env_path, 'w') as f:
                            f.write("# API Keys\n")
                    
                    # Update .env file
                    if openai_api_key:
                        set_key(env_path, 'OPENAI_API_KEY', openai_api_key)
                    
                    # Reload environment
                    load_dotenv(override=True)
                    st.success("✅ API key saved to .env file!")
                except Exception as e:
                    st.error(f"❌ Error saving to .env: {str(e)}")
        
        with col_save2:
            if st.button("🔄 Reload from .env", help="Reload API key from .env file"):
                load_dotenv(override=True)
                st.rerun()
        
        st.markdown("---")
        
        # ============ 2. Preset selection ============
        st.subheader("2. Select Preset")
        preset = st.radio(
            "Choose a preset configuration:",
            ["NLP (Default)", "Biology", "Load from file", "Custom"],
            key="preset_selection"
        )
        
        if preset == "NLP (Default)":
            if st.button("Load NLP Preset"):
                st.session_state.config = DimensionConfig()
                st.session_state.dimensions = st.session_state.config.dimensions.copy()
                st.success("✅ NLP preset loaded!")
                st.rerun()
        
        elif preset == "Biology":
            if st.button("Load Biology Preset"):
                st.session_state.config = load_biology_preset()
                st.session_state.dimensions = st.session_state.config.dimensions.copy()
                st.success("✅ Biology preset loaded!")
                st.rerun()
        
        elif preset == "Load from file":
            config_file = st.file_uploader("Upload YAML config", type=['yaml', 'yml'])
            if config_file is not None:
                # Save temporarily
                temp_path = f"/tmp/{config_file.name}"
                with open(temp_path, 'wb') as f:
                    f.write(config_file.getvalue())
                st.session_state.config = DimensionConfig(config_path=temp_path)
                st.session_state.dimensions = st.session_state.config.dimensions.copy()
                st.success(f"✅ Config loaded from {config_file.name}")
                st.rerun()
        
        elif preset == "Custom":
            if st.button("Create Empty Config"):
                st.session_state.config = DimensionConfig(config_path=None)
                st.session_state.config.dimensions = {}
                st.session_state.dimensions = {}
                st.success("✅ Empty config created!")
                st.rerun()
        
        st.markdown("---")
        
        # ============ 3. Dataset selection ============
        st.subheader("3. Select Dataset")
        dataset_source = st.radio(
            "Dataset source:",
            ["Use preset dataset", "Upload custom data"],
            key="dataset_source"
        )
        
        uploaded_data = None
        if dataset_source == "Use preset dataset":
            dataset_options = [
                'emnlp_2024',
                'emnlp_2022',
                'cvpr_2024',
                'cvpr_2020',
                'iclr_2024',
                'iclr_2021',
                'icra_2024',
                'icra_2020',
                'posco',
            ]
            dataset = st.selectbox("Dataset:", dataset_options, key="dataset_selection")
        else:
            dataset = st.text_input("Custom dataset name:", value="custom_dataset")
            st.write("**Upload your data:**")
            
            upload_format = st.radio(
                "Data format:",
                ["Excel (xlsx)", "JSON Lines (Title & Abstract)", "CSV", "TXT (one paper per line)"],
                key="upload_format"
            )
            
            uploaded_data = st.file_uploader(
                "Upload file",
                type=['xlsx', 'xls', 'jsonl', 'json', 'csv', 'txt'],
                help="Upload papers with title and abstract"
            )
            
            if upload_format == "Excel (xlsx)":
                st.info('Expected columns: "Title" (or "title") and "Abstract" (or "abstract")')
            elif upload_format == "JSON Lines (Title & Abstract)":
                st.info('Expected format: Each line is a JSON with "title" and "abstract" keys')
                st.code('{"title": "Paper Title", "abstract": "Paper abstract..."}\n{"title": "Another Paper", "abstract": "..."}', language="json")
            elif upload_format == "CSV":
                st.info('Expected columns: "title" and "abstract"')
            else:
                st.info('Each line should be: {"Title": "...", "Abstract": "..."}')
        
        st.markdown("---")
        
        # ============ 4. Initial Taxonomy Upload (Optional) ============
        st.subheader("4. Initial Taxonomy (Optional)")
        st.write("Upload initial taxonomy files if you have pre-defined taxonomy structure.")
        
        initial_taxonomy_files = st.file_uploader(
            "Upload initial_taxo_*.txt files",
            type=['txt'],
            accept_multiple_files=True,
            help="Upload pre-defined initial taxonomy files (e.g., initial_taxo_tasks.txt)"
        )
        
        if initial_taxonomy_files:
            st.success(f"✅ {len(initial_taxonomy_files)} taxonomy file(s) uploaded")
            for f in initial_taxonomy_files:
                st.write(f"  - {f.name}")
        
        st.markdown("---")
        
        # ============ 5. Execution parameters ============
        st.subheader("5. Execution Parameters")
        topic = st.text_input("Topic:", "natural language processing")
        max_depth = st.number_input("Max Depth:", min_value=1, max_value=10, value=2)
        init_levels = st.number_input("Init Levels:", min_value=1, max_value=5, value=1)
        max_density = st.number_input("Max Density:", min_value=1, max_value=200, value=40)
        test_samples = st.number_input("Test Samples (0=All):", min_value=0, max_value=10000, value=0)
        llm_type = "gpt"  # Fixed to GPT only
    
    # Main area
    if st.session_state.config is None:
        st.info("👈 Please select a preset from the sidebar to get started.")
        return
    
    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Dimensions", "📝 Definition & Dimension Editor", "▶️ Run", "💾 Save Config"])
    
    # ============ TAB 1: Dimensions ============
    with tab1:
        st.header("Dimension Configuration")
        
        if st.session_state.dimensions:
            # Display current dimensions
            st.subheader("Current Dimensions")
            
            for dim_name, dim_config in st.session_state.dimensions.items():
                with st.expander(f"📌 {dim_name.upper()}", expanded=False):
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        st.text_area(
                            "Definition:",
                            dim_config['definition'],
                            key=f"def_{dim_name}",
                            height=100
                        )
                        st.text_area(
                            "Node Definition:",
                            dim_config['node_definition'],
                            key=f"node_def_{dim_name}",
                            height=100
                        )
                    
                    with col2:
                        st.write("")
                        st.write("")
                        if st.button("🗑️ Remove", key=f"remove_{dim_name}"):
                            del st.session_state.dimensions[dim_name]
                            st.session_state.config.dimensions = st.session_state.dimensions
                            st.success(f"Removed {dim_name}")
                            st.rerun()
                        
                        if st.button("💾 Update", key=f"update_{dim_name}"):
                            st.session_state.dimensions[dim_name]['definition'] = st.session_state[f"def_{dim_name}"]
                            st.session_state.dimensions[dim_name]['node_definition'] = st.session_state[f"node_def_{dim_name}"]
                            st.session_state.config.dimensions = st.session_state.dimensions
                            st.success(f"Updated {dim_name}")
        
        # Add new dimension
        st.markdown("---")
        st.subheader("Add New Dimension")
        
        with st.form("add_dimension_form"):
            new_dim_name = st.text_input("Dimension Name (e.g., 'challenges'):")
            new_dim_def = st.text_area("Definition:", height=100)
            new_node_def = st.text_area("Node Definition:", height=100)
            
            if st.form_submit_button("➕ Add Dimension"):
                if new_dim_name and new_dim_def and new_node_def:
                    st.session_state.dimensions[new_dim_name] = {
                        'definition': new_dim_def,
                        'node_definition': new_node_def
                    }
                    st.session_state.config.dimensions = st.session_state.dimensions
                    st.success(f"✅ Added dimension: {new_dim_name}")
                    st.rerun()
                else:
                    st.error("Please fill in all fields")
    
    # ============ TAB 2: Definition & Dimension Editor for prompts.py ============
    with tab2:
        st.header("📝 Definition & Dimension Editor")
        st.markdown("Edit the definitions that will be used in `prompts.py` for taxonomy generation.")
        
        st.info("These definitions control how the LLM generates and classifies taxonomy nodes. " 
                "Modify them to customize the taxonomy generation for your specific domain.")
        
        # Load current prompts.py definitions for display
        current_dim_defs = {}
        current_node_dim_defs = {}
        
        if st.session_state.dimensions:
            for dim_name, dim_config in st.session_state.dimensions.items():
                current_dim_defs[dim_name] = dim_config.get('definition', '')
                current_node_dim_defs[dim_name] = dim_config.get('node_definition', '')
        
        st.subheader("Dimension Definitions")
        st.markdown("*These define what each dimension means (used in paper classification)*")
        
        updated_dim_defs = {}
        for dim_name in st.session_state.dimensions.keys():
            updated_dim_defs[dim_name] = st.text_area(
                f"Definition for '{dim_name}':",
                value=current_dim_defs.get(dim_name, ''),
                key=f"prompt_def_{dim_name}",
                height=120,
                help=f"This definition tells the LLM what '{dim_name}' means when classifying papers."
            )
        
        st.markdown("---")
        st.subheader("Node Dimension Definitions")
        st.markdown("*These define what types of nodes should be created under each dimension*")
        
        updated_node_defs = {}
        for dim_name in st.session_state.dimensions.keys():
            updated_node_defs[dim_name] = st.text_area(
                f"Node Definition for '{dim_name}':",
                value=current_node_dim_defs.get(dim_name, ''),
                key=f"prompt_node_def_{dim_name}",
                height=120,
                help=f"This tells the LLM what types of nodes to generate under '{dim_name}'."
            )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Definitions to Config", type="primary"):
                for dim_name in st.session_state.dimensions.keys():
                    st.session_state.dimensions[dim_name]['definition'] = updated_dim_defs[dim_name]
                    st.session_state.dimensions[dim_name]['node_definition'] = updated_node_defs[dim_name]
                st.session_state.config.dimensions = st.session_state.dimensions
                
                # Immediately update prompts.py
                try:
                    update_prompts_file_dynamic(st.session_state.config)
                    st.success("✅ Definitions saved to configuration and prompts.py updated!")
                except Exception as e:
                    st.error(f"❌ Error updating prompts.py: {e}")
                    st.success("✅ Definitions saved to configuration (but prompts.py update failed)")
        
        with col2:
            if st.button("🔄 Reset to Preset"):
                if st.session_state.get('preset_selection') == "NLP (Default)":
                    st.session_state.config = DimensionConfig()
                elif st.session_state.get('preset_selection') == "Biology":
                    st.session_state.config = load_biology_preset()
                st.session_state.dimensions = st.session_state.config.dimensions.copy()
                st.success("✅ Reset to preset!")
                st.rerun()
        
        # Preview generated prompts.py content
        with st.expander("🔍 Preview prompts.py Changes", expanded=False):
            st.markdown("**dimension_definitions:**")
            preview_dim_defs = "dimension_definitions = {\n"
            for dim, definition in updated_dim_defs.items():
                preview_dim_defs += f"    '{dim}': \"\"\"{definition}\"\"\",\n"
            preview_dim_defs += "}"
            st.code(preview_dim_defs, language="python")
            
            st.markdown("**node_dimension_definitions:**")
            preview_node_defs = "node_dimension_definitions = {\n"
            for dim, definition in updated_node_defs.items():
                preview_node_defs += f"    '{dim}': \"\"\"{definition}\"\"\",\n"
            preview_node_defs += "}"
            st.code(preview_node_defs, language="python")
    
    # ============ TAB 3: Run ============
    with tab3:
        st.header("Run TaxoAdapt")
        
        # Show current configuration
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Topic", topic)
            st.metric("Dataset", dataset)
            st.metric("Max Depth", max_depth)
            st.metric("Test Samples", test_samples if test_samples > 0 else "All")
        
        with col2:
            st.metric("Dimensions", len(st.session_state.dimensions))
            st.metric("Max Density", max_density)
            st.metric("LLM Type", "GPT (OpenAI)")
            st.metric("Initial Taxonomy Files", len(initial_taxonomy_files) if initial_taxonomy_files else 0)
        
        st.markdown("**Dimensions:**")
        st.write(", ".join(st.session_state.dimensions.keys()))
        
        # API Key Status
        st.markdown("---")
        st.markdown("**API Key Status:**")
        if openai_api_key:
            st.success("✅ OpenAI API Key provided")
        else:
            if os.environ.get('OPENAI_API_KEY'):
                st.info("ℹ️ Using OPENAI_API_KEY from environment")
            else:
                st.warning("⚠️ No OpenAI API Key (required)")
        
        st.markdown("---")
        
        # Validation checks
        errors = []
        if not st.session_state.dimensions:
            errors.append("⚠️ Please configure at least one dimension before running.")
        if dataset_source == "Upload custom data" and uploaded_data is None:
            errors.append("⚠️ Please upload your custom dataset file.")
        if not openai_api_key and not os.environ.get('OPENAI_API_KEY'):
            errors.append("⚠️ OpenAI API Key is required.")
        
        for err in errors:
            st.warning(err)
        
        # Run button
        if not errors:
            if st.button("🚀 Run TaxoAdapt", type="primary", disabled=st.session_state.running):
                run_taxoadapt(
                    config=st.session_state.config,
                    dataset=dataset,
                    topic=topic,
                    max_depth=max_depth,
                    init_levels=init_levels,
                    max_density=max_density,
                    llm=llm_type,
                    openai_api_key=openai_api_key,
                    huggingface_token=huggingface_token,
                    uploaded_data=uploaded_data,
                    upload_format=st.session_state.get('upload_format', None) if dataset_source == "Upload custom data" else None,
                    initial_taxonomy_files=initial_taxonomy_files,
                    test_samples=test_samples if test_samples > 0 else None
                )
    
    # ============ TAB 4: Save Config ============
    with tab4:
        st.header("Save Configuration")
        
        st.write("Save your current dimension configuration for future use.")
        
        save_path = st.text_input(
            "Save path:",
            value="configs/my_config.yaml",
            help="Path relative to project root"
        )
        
        if st.button("💾 Save Configuration"):
            try:
                full_path = os.path.join(BASE_DIR, save_path)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                st.session_state.config.save_config(full_path)
                st.success(f"✅ Configuration saved to {save_path}")
                
                # Show download button
                with open(full_path, 'r') as f:
                    st.download_button(
                        label="📥 Download Config File",
                        data=f.read(),
                        file_name=os.path.basename(save_path),
                        mime="text/yaml"
                    )
            except Exception as e:
                st.error(f"❌ Error saving config: {e}")


def run_taxoadapt(config, dataset, topic, max_depth, init_levels, max_density, llm, 
                   openai_api_key=None, huggingface_token=None, uploaded_data=None, 
                   upload_format=None, initial_taxonomy_files=None, test_samples=None):
    """Run TaxoAdapt (main2.py) with the given configuration"""
    st.session_state.running = True
    
    try:
        # Update prompts.py with user-defined dimensions
        with st.spinner("🔄 Updating prompts.py with custom definitions..."):
            update_prompts_file_dynamic(config)
        st.success("✅ prompts.py updated with custom definitions")
        
        # Prepare data directory
        data_dir = BASE_DIR / "datasets" / dataset.lower().replace(' ', '_')
        os.makedirs(data_dir, exist_ok=True)
        
        # Process uploaded data if provided
        if uploaded_data is not None:
            with st.spinner("📤 Processing uploaded data..."):
                num_papers = process_uploaded_data(uploaded_data, dataset, upload_format)
            st.success(f"✅ Custom data processed ({num_papers} papers)")
        
        # Process initial taxonomy files if provided
        if initial_taxonomy_files:
            with st.spinner("📤 Processing initial taxonomy files..."):
                for taxonomy_file in initial_taxonomy_files:
                    file_path = data_dir / taxonomy_file.name
                    with open(file_path, 'wb') as f:
                        f.write(taxonomy_file.getvalue())
                    st.info(f"  Saved: {taxonomy_file.name}")
            st.success(f"✅ {len(initial_taxonomy_files)} initial taxonomy file(s) saved")
        
        # Set environment variables for API keys
        env = os.environ.copy()
        if openai_api_key:
            env['OPENAI_API_KEY'] = openai_api_key
        if huggingface_token:
            env['HUGGINGFACE_TOKEN'] = huggingface_token
            env['HF_TOKEN'] = huggingface_token
        
        # Prepare command - Use main2.py
        cmd = [
            sys.executable,
            str(BASE_DIR / "main2.py"),
            "--topic", topic,
            "--dataset", dataset,
            "--llm", llm,
            "--max_depth", str(max_depth),
            "--init_levels", str(init_levels),
            "--max_density", str(max_density)
        ]
        
        # Add test_samples if specified
        if test_samples:
            cmd.extend(["--test_samples", str(test_samples)])
        
        st.info(f"🚀 Running TaxoAdapt with {len(config.dimensions)} dimensions...")
        st.code(" ".join(cmd), language="bash")
        
        # Create output display
        with st.spinner("⏳ Running... This may take a while."):
            # Run the command
            result = subprocess.run(
                cmd,
                cwd=str(BASE_DIR),
                capture_output=True,
                text=True,
                env=env
            )
            
            # Display output
            with st.expander("📄 Execution Log", expanded=True):
                if result.stdout:
                    st.text_area("STDOUT:", result.stdout, height=300)
                if result.stderr:
                    st.text_area("STDERR:", result.stderr, height=200)
            
            if result.returncode == 0:
                st.success("✅ TaxoAdapt completed successfully!")
                
                # Show output location
                st.info(f"📁 Results saved to: {data_dir}")
                
                # List output files
                if data_dir.exists():
                    files = [f for f in os.listdir(data_dir) if f.startswith('final_taxo_')]
                    if files:
                        st.write("**Generated taxonomies:**")
                        for f in files:
                            st.write(f"- {f}")
                            
                            # Offer download
                            file_path = data_dir / f
                            with open(file_path, 'r', encoding='utf-8') as fp:
                                st.download_button(
                                    label=f"📥 Download {f}",
                                    data=fp.read(),
                                    file_name=f,
                                    mime="text/plain" if f.endswith('.txt') else "application/json",
                                    key=f"download_{f}"
                                )
            else:
                st.error(f"❌ TaxoAdapt failed with exit code {result.returncode}")
    
    except Exception as e:
        st.error(f"❌ Error: {e}")
        import traceback
        st.code(traceback.format_exc())
    
    finally:
        st.session_state.running = False


def update_prompts_file_dynamic(config: DimensionConfig):
    """
    Dynamically update prompts.py with user-defined dimension definitions.
    This replaces dimension_definitions, node_dimension_definitions, 
    TypeClsSchema, type_cls_system_instruction, and type_cls_main_prompt.
    """
    import re
    
    prompts_path = BASE_DIR / 'prompts.py'
    with open(prompts_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Get dimensions from config
    dims = config.dimensions
    dim_list = list(dims.keys())
    
    # ============ 1. Update dimension_definitions ============
    dim_def_str = "dimension_definitions = {\n"
    for dim_name, dim_config in dims.items():
        definition = dim_config.get('definition', '').replace('"""', '\\"\\"\\"')
        dim_def_str += f"    '{dim_name}': \"\"\"{definition}\"\"\",\n"
    dim_def_str += "    }"
    
    # Replace the active dimension_definitions (not commented)
    pattern = r'^dimension_definitions = \{[^}]*\}'
    content = re.sub(pattern, dim_def_str, content, count=1, flags=re.MULTILINE)
    
    # ============ 2. Update node_dimension_definitions ============
    node_dim_def_str = "node_dimension_definitions = {\n"
    for dim_name, dim_config in dims.items():
        node_def = dim_config.get('node_definition', '').replace('"""', '\\"\\"\\"')
        node_dim_def_str += f"    '{dim_name}': \"\"\"{node_def}\"\"\",\n"
    node_dim_def_str += "}"
    
    # Find and replace node_dimension_definitions
    pattern = r'^node_dimension_definitions = \{[^}]*\}'
    content = re.sub(pattern, node_dim_def_str, content, count=1, flags=re.MULTILINE)
    
    # ============ 3. Update TypeClsSchema ============
    schema_fields = "\n".join([f"  {dim_name}: bool" for dim_name in dim_list])
    schema_code = f"""class TypeClsSchema(BaseModel):
{schema_fields}"""
    
    pattern = r'class TypeClsSchema\(BaseModel\):.*?(?=\n\nclass |\n\ndef |\n# )'
    content = re.sub(pattern, schema_code + "\n", content, flags=re.DOTALL, count=1)
    
    # ============ 4. Update type_cls_system_instruction ============
    instruction_parts = ["You are a helpful multi-label classification assistant which helps me classify steel technology papers based on low-carbon steel technology dimensions. Papers may belong to multiple dimensions.\n\nSteel Technology Dimensions (dimension:definition):\n"]
    for i, (dim_name, dim_config) in enumerate(dims.items(), 1):
        definition = dim_config.get('definition', '').replace('"""', '"')
        dim_title = dim_name.replace('_', ' ').title()
        instruction_parts.append(f"{i}. {dim_title}: {definition}")
    
    new_instruction = "\n".join(instruction_parts)
    
    pattern = r'type_cls_system_instruction = """[^"]*?"""'
    content = re.sub(pattern, f'type_cls_system_instruction = """{new_instruction}"""', content, count=1, flags=re.DOTALL)
    
    # ============ 5. Update type_cls_main_prompt function ============
    json_fields = []
    for dim_name in dim_list:
        json_fields.append(f'  "{dim_name}": true,')
    
    json_output = "{\n" + "\n".join(json_fields[:-1]) + json_fields[-1].replace(',', '') + "\n}"
    
    new_type_cls_main_prompt = f'''def type_cls_main_prompt(paper):
   out = f"""Given the following paper title and abstract, can you classify this steel technology paper according to the relevant dimensions. 

"Title": "{{paper.title}}"
"Abstract": "{{paper.abstract}}"

Your output should be in the following JSON format:
{json_output}
"""
   return out'''
    
    # Replace the function
    pattern = r'def type_cls_main_prompt\(paper\):.*?return out'
    content = re.sub(pattern, new_type_cls_main_prompt, content, flags=re.DOTALL, count=1)
    
    # Write back
    with open(prompts_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Updated {prompts_path} with {len(dim_list)} dimensions: {dim_list}")


def process_uploaded_data(uploaded_file, dataset_name, upload_format):
    """Process uploaded custom data and save in the required format"""
    data_dir = BASE_DIR / "datasets" / dataset_name.lower().replace(' ', '_')
    os.makedirs(data_dir, exist_ok=True)
    
    internal_file = data_dir / 'internal.txt'
    papers = []
    
    # Handle Excel files separately (binary format)
    if upload_format == "Excel (xlsx)":
        import pandas as pd
        df = pd.read_excel(uploaded_file)
        # Normalize column names (case-insensitive)
        df.columns = [col.strip() for col in df.columns]
        
        # Try to find title and abstract columns
        title_col = None
        abstract_col = None
        for col in df.columns:
            if col.lower() == 'title':
                title_col = col
            elif col.lower() == 'abstract':
                abstract_col = col
        
        if title_col and abstract_col:
            for _, row in df.iterrows():
                title = row.get(title_col, '')
                abstract = row.get(abstract_col, '')
                if pd.notna(title) and pd.notna(abstract):
                    papers.append({
                        "Title": str(title),
                        "Abstract": str(abstract)
                    })
        else:
            st.error(f"Could not find 'title' and 'abstract' columns. Found columns: {list(df.columns)}")
            return 0
    else:
        # Read uploaded file as text
        content = uploaded_file.getvalue().decode('utf-8')
    
        if upload_format == "JSON Lines (Title & Abstract)":
            # Parse JSON Lines
            for line in content.strip().split('\n'):
                if line.strip():
                    try:
                        paper = json.loads(line)
                        if 'title' in paper and 'abstract' in paper:
                            papers.append({
                                "Title": paper['title'],
                                "Abstract": paper['abstract']
                            })
                    except json.JSONDecodeError:
                        st.warning(f"Skipping invalid JSON line: {line[:50]}...")
    
        elif upload_format == "CSV":
            # Parse CSV
            import io
            csv_reader = csv.DictReader(io.StringIO(content))
            for row in csv_reader:
                if 'title' in row and 'abstract' in row:
                    papers.append({
                        "Title": row['title'],
                        "Abstract": row['abstract']
                    })
                elif 'Title' in row and 'Abstract' in row:
                    papers.append({
                        "Title": row['Title'],
                        "Abstract": row['Abstract']
                    })
    
        else:  # TXT format
            # Each line is expected to be a JSON string
            for line in content.strip().split('\n'):
                if line.strip():
                    try:
                        paper = json.loads(line)
                        if 'Title' in paper and 'Abstract' in paper:
                            papers.append(paper)
                        elif 'title' in paper and 'abstract' in paper:
                            papers.append({
                                "Title": paper['title'],
                                "Abstract": paper['abstract']
                            })
                    except json.JSONDecodeError:
                        st.warning(f"Skipping invalid line: {line[:50]}...")
    
    # Write to internal.txt in the required format
    with open(internal_file, 'w', encoding='utf-8') as f:
        for paper in papers:
            formatted_dict = json.dumps(paper, ensure_ascii=False)
            f.write(f'{formatted_dict}\n')
    
    st.info(f"📊 Processed {len(papers)} papers and saved to {internal_file}")
    return len(papers)


if __name__ == "__main__":
    main()
