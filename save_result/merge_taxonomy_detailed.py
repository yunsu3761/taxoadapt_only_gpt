#!/usr/bin/env python
"""
Merge taxonomy classification results with original dataset (detailed version)
- Includes descriptions from initial_taxo files
- Creates separate rows for each classification (long format)
- Also creates wide format with level-based columns
"""
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
import re
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from prompts import dimension_definitions

def parse_initial_taxonomy_txt(file_path):
    """Parse initial_taxo_*.txt file and extract label-description mapping"""
    label_info = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by node separators
    nodes = re.split(r'-{3,}', content)
    
    for node_text in nodes:
        if not node_text.strip():
            continue
        
        label = None
        description = None
        level = None
        
        for line in node_text.split('\n'):
            line = line.strip()
            if line.startswith('Label:'):
                label = line.split('Label:')[1].strip()
            elif line.startswith('Description:'):
                description = line.split('Description:')[1].strip()
            elif line.startswith('Level:'):
                try:
                    level = int(line.split('Level:')[1].strip())
                except:
                    pass
        
        if label and description:
            label_info[label] = {
                'description': description,
                'level': level
            }
    
    return label_info

def load_all_initial_descriptions(data_dir):
    """Load descriptions from all initial taxonomy files"""
    # Load dimensions dynamically from prompts.py
    dimensions = list(dimension_definitions.keys())
    all_descriptions = {}
    
    for dim in dimensions:
        txt_file = data_dir / f'initial_taxo_{dim}.txt'
        if txt_file.exists():
            print(f"Loading descriptions from {txt_file.name}...")
            dim_descriptions = parse_initial_taxonomy_txt(txt_file)
            all_descriptions[dim] = dim_descriptions
            print(f"  - Loaded {len(dim_descriptions)} node descriptions")
        else:
            print(f"Warning: {txt_file.name} not found")
            all_descriptions[dim] = {}
    
    return all_descriptions

def extract_paper_classifications(taxonomy_node, dimension, path="", parent_labels=None, classifications=None):
    """Recursively extract paper IDs and their classifications with full path info"""
    if classifications is None:
        classifications = defaultdict(list)
    if parent_labels is None:
        parent_labels = []
    
    current_label = taxonomy_node.get('label', '')
    current_level = taxonomy_node.get('level', 0)
    current_description = taxonomy_node.get('description', '')
    current_source = taxonomy_node.get('source', 'Initial')  # Initial, width, depth
    
    # Build full path
    current_path = f"{path}/{current_label}" if path else current_label
    current_labels = parent_labels + [current_label]
    
    # Get papers at this node
    paper_ids = taxonomy_node.get('paper_ids', [])
    for paper_id in paper_ids:
        classifications[paper_id].append({
            'dimension': dimension,
            'label': current_label,
            'description': current_description,
            'path': current_path,
            'level': current_level,
            'source': current_source,
            'full_path_labels': current_labels.copy()
        })
    
    # Recursively process children
    children = taxonomy_node.get('children', {})
    if isinstance(children, dict):
        for child_label, child_node in children.items():
            extract_paper_classifications(child_node, dimension, current_path, current_labels, classifications)
    elif isinstance(children, list):
        for child_node in children:
            extract_paper_classifications(child_node, dimension, current_path, current_labels, classifications)
    
    return classifications

def load_taxonomy_files(data_dir, initial_descriptions):
    """Load all taxonomy JSON files and enrich with initial descriptions"""
    # Load dimensions dynamically from prompts.py
    dimensions = list(dimension_definitions.keys())
    all_classifications = defaultdict(list)
    
    for dim in dimensions:
        json_file = data_dir / f'final_taxo_{dim}.json'
        if json_file.exists():
            print(f"Loading {json_file.name}...")
            with open(json_file, 'r', encoding='utf-8') as f:
                taxonomy = json.load(f)
            
            # Extract classifications
            dim_classifications = extract_paper_classifications(taxonomy, dim)
            
            # Enrich with initial descriptions if missing
            for paper_id, labels in dim_classifications.items():
                for label_info in labels:
                    if not label_info['description']:
                        label = label_info['label']
                        if label in initial_descriptions.get(dim, {}):
                            label_info['description'] = initial_descriptions[dim][label]['description']
            
            # Merge into all_classifications
            for paper_id, labels in dim_classifications.items():
                all_classifications[paper_id].extend(labels)
            
            print(f"  - Found classifications for {len(dim_classifications)} papers")
        else:
            print(f"Warning: {json_file.name} not found")
    
    return all_classifications

def create_long_format(df_original, classifications):
    """Create long format: one row per paper-classification pair"""
    rows = []
    
    for idx, row in df_original.iterrows():
        paper_id = idx
        base_info = row.to_dict()
        
        if paper_id in classifications:
            # Create a row for each classification
            for cls in classifications[paper_id]:
                row_data = base_info.copy()
                row_data.update({
                    'classification_dimension': cls['dimension'],
                    'classification_label': cls['label'],
                    'classification_description': cls['description'],
                    'classification_path': cls['path'],
                    'classification_level': cls['level'],
                    'classification_source': cls['source']
                })
                rows.append(row_data)
        else:
            # Unclassified paper
            row_data = base_info.copy()
            row_data.update({
                'classification_dimension': '',
                'classification_label': '',
                'classification_description': '',
                'classification_path': '',
                'classification_level': None,
                'classification_source': ''
            })
            rows.append(row_data)
    
    return pd.DataFrame(rows)

def create_wide_format(df_original, classifications):
    """Create wide format: separate columns for each dimension and level"""
    df_wide = df_original.copy()
    
    # Initialize columns for each dimension (load dynamically)
    dimensions = list(dimension_definitions.keys())
    max_levels = 5  # Assume max 5 levels
    
    for dim in dimensions:
        df_wide[f'{dim}_classified'] = False
        for level in range(max_levels):
            df_wide[f'{dim}_level{level}_label'] = ''
            df_wide[f'{dim}_level{level}_description'] = ''
            df_wide[f'{dim}_level{level}_source'] = ''
    
    # Fill classifications
    for idx, row in df_original.iterrows():
        paper_id = idx
        
        if paper_id in classifications:
            # Group by dimension
            dim_data = defaultdict(list)
            for cls in classifications[paper_id]:
                dim_data[cls['dimension']].append(cls)
            
            # Fill each dimension
            for dim, cls_list in dim_data.items():
                df_wide.at[idx, f'{dim}_classified'] = True
                
                # Sort by level and take the deepest path
                cls_list_sorted = sorted(cls_list, key=lambda x: x['level'], reverse=True)
                deepest = cls_list_sorted[0]
                
                # Fill level-by-level from path
                labels = deepest['full_path_labels']
                for level, label in enumerate(labels):
                    if level < max_levels:
                        df_wide.at[idx, f'{dim}_level{level}_label'] = label
                        
                        # Try to find description and source
                        desc = ''
                        source = 'Initial'
                        for cls in cls_list:
                            if cls['label'] == label:
                                desc = cls['description']
                                source = cls['source']
                                break
                        df_wide.at[idx, f'{dim}_level{level}_description'] = desc
                        df_wide.at[idx, f'{dim}_level{level}_source'] = source
    
    return df_wide

def main():
    # Paths
    data_dir = Path('/root/taxo/taxoadapt/datasets/posco')
    excel_path = data_dir / 'posco_dataset_summary.xlsx'
    output_long = data_dir / 'posco_dataset_taxonomy_long.xlsx'
    output_long_csv = data_dir / 'posco_dataset_taxonomy_long.csv'
    output_wide = data_dir / 'posco_dataset_taxonomy_wide.xlsx'
    output_wide_csv = data_dir / 'posco_dataset_taxonomy_wide.csv'
    
    print("=" * 60)
    print("Detailed Taxonomy Merge Tool")
    print("=" * 60)
    
    # Load initial descriptions
    print("\n[1/4] Loading initial taxonomy descriptions...")
    initial_descriptions = load_all_initial_descriptions(data_dir)
    
    # Load taxonomy classifications
    print("\n[2/4] Loading final taxonomy classifications...")
    classifications = load_taxonomy_files(data_dir, initial_descriptions)
    print(f"\nTotal papers with classifications: {len(classifications)}")
    
    # Load original data (only 20 rows for testing)
    print(f"\n[3/4] Loading original data (20 rows)...")
    df_original = pd.read_excel(excel_path, nrows=20)
    print(f"  - Loaded {len(df_original)} papers (testing)")
    
    # Create both formats
    print(f"\n[4/4] Creating output formats...")
    
    # Long format
    print("  - Creating long format (one row per classification)...")
    df_long = create_long_format(df_original, classifications)
    df_long.to_excel(output_long, index=False, engine='openpyxl')
    df_long.to_csv(output_long_csv, index=False, encoding='utf-8')
    print(f"    ✓ Saved: {output_long.name} ({len(df_long)} rows)")
    
    # Wide format
    print("  - Creating wide format (level-based columns)...")
    df_wide = create_wide_format(df_original, classifications)
    df_wide.to_excel(output_wide, index=False, engine='openpyxl')
    df_wide.to_csv(output_wide_csv, index=False, encoding='utf-8')
    print(f"    ✓ Saved: {output_wide.name} ({len(df_wide)} rows)")
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Statistics")
    print("=" * 60)
    print(f"Original papers: {len(df_original)}")
    print(f"Papers with classifications: {len(classifications)}")
    print(f"Total classification entries (long format): {len(df_long)}")
    print()
    
    # Dimension statistics
    dim_stats = df_long.groupby('classification_dimension').size()
    for dim, count in dim_stats.items():
        if dim:  # Skip empty dimension
            print(f"{dim:20s}: {count:5d} classification entries")
    
    print("\n" + "=" * 60)
    print("✓ Merge completed!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  Long format (one row per classification):")
    print(f"    - {output_long}")
    print(f"    - {output_long_csv}")
    print(f"  Wide format (level-based columns):")
    print(f"    - {output_wide}")
    print(f"    - {output_wide_csv}")

if __name__ == "__main__":
    main()
