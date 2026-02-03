#!/usr/bin/env python
"""
Merge taxonomy classification results with original dataset
"""
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from prompts import dimension_definitions

def extract_paper_classifications(taxonomy_node, dimension, path="", classifications=None):
    """Recursively extract paper IDs and their classifications from taxonomy tree"""
    if classifications is None:
        classifications = defaultdict(list)
    
    current_label = taxonomy_node.get('label', '')
    current_path = f"{path}/{current_label}" if path else current_label
    
    # Get papers at this node
    paper_ids = taxonomy_node.get('paper_ids', [])
    for paper_id in paper_ids:
        classifications[paper_id].append({
            'dimension': dimension,
            'path': current_path,
            'label': current_label,
            'level': taxonomy_node.get('level', 0)
        })
    
    # Recursively process children
    children = taxonomy_node.get('children', {})
    if isinstance(children, dict):
        for child_label, child_node in children.items():
            extract_paper_classifications(child_node, dimension, current_path, classifications)
    elif isinstance(children, list):
        for child_node in children:
            extract_paper_classifications(child_node, dimension, current_path, classifications)
    
    return classifications

def load_taxonomy_files(data_dir):
    """Load all taxonomy JSON files"""
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
            
            # Merge into all_classifications
            for paper_id, labels in dim_classifications.items():
                all_classifications[paper_id].extend(labels)
            
            print(f"  - Found classifications for {len(dim_classifications)} papers")
        else:
            print(f"Warning: {json_file.name} not found")
    
    return all_classifications

def merge_with_original_data(excel_path, classifications):
    """Merge classifications with original Excel data"""
    print(f"\nLoading original data from {excel_path.name}...")
    df = pd.read_excel(excel_path)
    print(f"  - Loaded {len(df)} papers")
    
    # Add classification columns
    df['classified'] = False
    df['raw_material_optimization'] = ''
    df['reduction_efficiency_enhancement'] = ''
    df['fuel_and_gas_substitution'] = ''
    df['blast_furnace_structural_and_energy_efficiency_improvement'] = ''
    df['all_labels'] = ''
    
    # Merge classifications
    classified_count = 0
    for idx, row in df.iterrows():
        paper_id = idx  # Assuming paper_id = row index
        
        if paper_id in classifications:
            df.at[idx, 'classified'] = True
            classified_count += 1
            
            # Group by dimension
            dim_paths = defaultdict(list)
            all_labels = []
            
            for cls in classifications[paper_id]:
                dim = cls['dimension']
                path = cls['path']
                label = cls['label']
                
                dim_paths[dim].append(path)
                all_labels.append(f"{dim}:{label}")
            
            # Fill dimension columns with paths
            for dim, paths in dim_paths.items():
                df.at[idx, dim] = ' | '.join(paths)
            
            # Fill all_labels column
            df.at[idx, 'all_labels'] = ' | '.join(all_labels)
    
    print(f"  - Merged classifications for {classified_count} papers")
    print(f"  - Unclassified: {len(df) - classified_count} papers")
    
    return df

def main():
    # Paths
    data_dir = Path('/root/taxo/taxoadapt/datasets/posco')
    excel_path = data_dir / 'posco_dataset_summary.xlsx'
    output_path = data_dir / 'posco_dataset_with_taxonomy.xlsx'
    output_csv_path = data_dir / 'posco_dataset_with_taxonomy.csv'
    
    print("=" * 60)
    print("Taxonomy Merge Tool")
    print("=" * 60)
    
    # Load taxonomy classifications
    classifications = load_taxonomy_files(data_dir)
    print(f"\nTotal papers with classifications: {len(classifications)}")
    
    # Load and merge with original data
    df_merged = merge_with_original_data(excel_path, classifications)
    
    # Save results
    print(f"\nSaving results...")
    df_merged.to_excel(output_path, index=False, engine='openpyxl')
    print(f"  ✓ Excel saved: {output_path.name}")
    
    df_merged.to_csv(output_csv_path, index=False, encoding='utf-8')
    print(f"  ✓ CSV saved: {output_csv_path.name}")
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Statistics")
    print("=" * 60)
    print(f"Total papers: {len(df_merged)}")
    print(f"Classified papers: {df_merged['classified'].sum()}")
    print(f"Unclassified papers: {(~df_merged['classified']).sum()}")
    print()
    
    for dim in ['raw_material_optimization', 'reduction_efficiency_enhancement', 'fuel_and_gas_substitution', 'blast_furnace_structural_and_energy_efficiency_improvement']:
        count = (df_merged[dim] != '').sum()
        print(f"{dim:60s}: {count:5d} papers")
    
    print("\n" + "=" * 60)
    print("✓ Merge completed!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - {output_path}")
    print(f"  - {output_csv_path}")

if __name__ == "__main__":
    main()
