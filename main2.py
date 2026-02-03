import os
import json
from collections import deque
from contextlib import redirect_stdout
import argparse
from tqdm import tqdm
import pandas as pd
import multiprocessing
import torch.multiprocessing

# Fix CUDA multiprocessing issue with vLLM
# Must be set before any CUDA operations
if __name__ == "__main__":
    # Set environment variable for vLLM's internal multiprocessing
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    
    # Also set Python's multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

from model_definitions import initializeLLM, promptLLM, constructPrompt
from prompts import multi_dim_prompt, NodeListSchema, type_cls_system_instruction, type_cls_main_prompt, TypeClsSchema
from taxonomy import Node, DAG
from datasets import load_dataset
from expansion import expandNodeWidth, expandNodeDepth
from paper import Paper
from utils import clean_json_string

def construct_dataset(args):
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    split = 'train'
    
    if args.dataset == 'posco':
        # Read Excel file from posco folder
        excel_path = '/root/taxoadapt_only_gpt/datasets/posco/posco_papers.xlsx'
        df = pd.read_excel(excel_path)
        ds = [{'title': row.get('Title', row.get('title', '')), 
               'abstract': row.get('Abstract', row.get('abstract', ''))} 
              for _, row in df.iterrows()]
    elif args.dataset == 'emnlp_2024':
        ds = load_dataset("EMNLP/EMNLP2024-papers")
        ds = ds[split]
    elif args.dataset == 'emnlp_2022':
        ds = load_dataset("TimSchopf/nlp_taxonomy_data")
        split = 'test'
        ds = ds[split]
    elif args.dataset == 'cvpr_2024':
        ds = load_dataset("DeepNLP/CVPR-2024-Accepted-Papers")
        ds = ds[split]
    elif args.dataset == 'cvpr_2020':
        ds = load_dataset("DeepNLP/CVPR-2020-Accepted-Papers")
        ds = ds[split]
    elif args.dataset == 'iclr_2024':
        ds = load_dataset("DeepNLP/ICLR-2024-Accepted-Papers")
        ds = ds[split]
    elif args.dataset == 'iclr_2021':
        ds = load_dataset("DeepNLP/ICLR-2021-Accepted-Papers")
        ds = ds[split]
    elif args.dataset == 'icra_2024':
        ds = load_dataset("DeepNLP/ICRA-2024-Accepted-Papers")
        ds = ds[split]
    else:
        ds = load_dataset("DeepNLP/ICRA-2020-Accepted-Papers")
        ds = ds[split]
    
    
    internal_collection = {}

    with open(os.path.join(args.data_dir, 'internal.txt'), 'w', encoding='utf-8') as i:
        internal_count = 0
        id = 0
        for p in tqdm(ds):
            if ('title' not in p) and ('abstract' not in p):
                continue
            
            # Limit samples for testing if specified
            if args.test_samples is not None and internal_count >= args.test_samples:
                print(f"Limiting to {args.test_samples} samples for testing")
                break
            
            temp_dict = {"Title": p['title'], "Abstract": p['abstract']}
            formatted_dict = json.dumps(temp_dict, ensure_ascii=False)
            i.write(f'{formatted_dict}\n')
            internal_collection[id] = Paper(id, p['title'], p['abstract'], label_opts=args.dimensions, internal=True)
            internal_count += 1
            id += 1
        print("Total # of Papers: ", internal_count)
    
    return internal_collection, internal_count

def initialize_DAG(args, use_txt=True):
    """Initialize DAG either from txt files or using LLM"""
    
    if use_txt:
        # Load from txt files
        roots = {}
        id2node = {}
        label2node = {}
        
        for dim in args.dimensions:
            txt_file = f'{args.data_dir}/initial_taxo_{dim}.txt'
            
            if not os.path.exists(txt_file):
                raise FileNotFoundError(f"Initial taxonomy file not found: {txt_file}")
            
            parsed_data = parse_initial_taxonomy_txt(txt_file)
            
            # Add id to each node
            node_counter = {'count': len(id2node)}
            def add_ids(node_dict, node_counter):
                node_dict['id'] = node_counter['count']
                node_counter['count'] += 1
                if 'children' in node_dict:
                    for child in node_dict['children'].values():
                        add_ids(child, node_counter)
            
            add_ids(parsed_data, node_counter)
            
            # Convert to Node objects
            root = Node.from_dict(parsed_data, id2node, label2node)
            roots[dim] = root
            mod_topic = args.topic.replace(' ', '_').lower()
            label2node[mod_topic + f"_{dim}"] = root
        
        return roots, id2node, label2node
    
    else:
        # Generate using LLM
        ## we want to make this a directed acyclic graph (DAG) so maintain a list of the nodes
        roots = {}
        id2node = {}
        label2node = {}
        idx = 0

        for dim in args.dimensions:
            mod_topic = args.topic.replace(' ', '_').lower()
            mod_full_topic = args.topic.replace(' ', '_').lower() + f"_{dim}"
            root = Node(
                    id=idx,
                    label=mod_topic,
                    dimension=dim
                )
            roots[dim] = root
            id2node[idx] = root
            label2node[mod_full_topic] = root
            idx += 1

        queue = deque([node for id, node in id2node.items()])

        # if taking long, you can probably parallelize this between the different taxonomies (expand by level)
        while queue:
            curr_node = queue.popleft()
            label = curr_node.label
            dim = curr_node.dimension
            # expand
            system_instruction, main_prompt, json_output_format = multi_dim_prompt(curr_node)
            prompts = [constructPrompt(args, system_instruction, main_prompt + "\n\n" + json_output_format)]
            outputs = promptLLM(args=args, prompts=prompts, schema=NodeListSchema, max_new_tokens=3000, json_mode=True, temperature=0.01, top_p=1.0)[0]
            outputs = json.loads(clean_json_string(outputs)) if "```" in outputs else json.loads(outputs.strip())
            outputs = outputs['root_topic'] if 'root_topic' in outputs else outputs[label]

            # add all children
            for key, value in outputs.items():
                mod_key = key.replace(' ', '_').lower()
                mod_full_key = mod_key + f"_{dim}"
                if mod_full_key not in label2node:
                    child_node = Node(
                            id=len(id2node),
                            label=mod_key,
                            dimension=dim,
                            description=value['description'],
                            parents=[curr_node]
                        )
                    curr_node.add_child(mod_key, child_node)
                    id2node[child_node.id] = child_node
                    label2node[mod_full_key] = child_node
                    if child_node.level < args.init_levels:
                        queue.append(child_node)
                elif label2node[mod_full_key] in label2node[label + f"_{dim}"].get_ancestors():
                    continue
                else:
                    child_node = label2node[mod_full_key]
                    curr_node.add_child(mod_key, child_node)
                    child_node.add_parent(curr_node)

        return roots, id2node, label2node


def parse_initial_taxonomy_txt(file_path):
    """Parse initial_taxo_*.txt file and return a dictionary structure"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    def parse_node(lines, start_idx):
        """Recursively parse a node and its children"""
        node = {}
        children = {}
        i = start_idx
        
        # Determine the base indent of this node
        base_indent = None
        while i < len(lines) and base_indent is None:
            stripped = lines[i].strip()
            if stripped and stripped.startswith('Label:'):
                base_indent = len(lines[i]) - len(lines[i].lstrip(' '))
                break
            i += 1
        
        if base_indent is None:
            return node, i
        
        # Parse current node's properties
        while i < len(lines):
            line = lines[i]
            indent = len(line) - len(line.lstrip(' '))
            stripped = line.strip()
            
            # Skip empty lines and separators
            if not stripped or stripped.startswith('---'):
                i += 1
                continue
            
            # Stop if we encounter a node at same or lower indent level (after we've started parsing)
            if stripped.startswith('Label:') and indent == base_indent and 'label' in node:
                break
            elif stripped.startswith('Label:') and indent < base_indent:
                break
            
            # Parse node properties at base indent level
            if indent == base_indent:
                if stripped.startswith('Label:'):
                    node['label'] = stripped.split('Label:')[1].strip()
                elif stripped.startswith('Dimension:'):
                    node['dimension'] = stripped.split('Dimension:')[1].strip()
                elif stripped.startswith('Description:'):
                    desc = stripped.split('Description:')[1].strip()
                    node['description'] = desc if desc else ''
                elif stripped.startswith('Level:'):
                    node['level'] = int(stripped.split('Level:')[1].strip())
                elif stripped.startswith('Source:'):
                    node['source'] = stripped.split('Source:')[1].strip()
                elif stripped == 'Children:':
                    # Found children section, parse all children
                    i += 1
                    while i < len(lines):
                        next_line = lines[i]
                        next_indent = len(next_line) - len(next_line.lstrip(' '))
                        next_stripped = next_line.strip()
                        
                        # Skip separators and empty lines
                        if not next_stripped or next_stripped.startswith('---'):
                            i += 1
                            continue
                        
                        # Child starts with Label: at deeper indent
                        if next_stripped.startswith('Label:') and next_indent > base_indent:
                            child_node, i = parse_node(lines, i)
                            if child_node and 'label' in child_node:
                                children[child_node['label']] = child_node
                            continue
                        
                        # If we're back to parent indent or less, stop parsing children
                        if next_indent <= base_indent:
                            break
                        
                        i += 1
                    continue
            
            i += 1
        
        if children:
            node['children'] = children
        
        return node, i
    
    root_node, _ = parse_node(lines, 0)
    return root_node


def main(args):

    print("######## STEP 1: LOAD IN DATASET ########")

    internal_collection, internal_count = construct_dataset(args)
    
    print(f'Internal: {internal_count}')

    print("######## STEP 2: INITIALIZE DAG ########")
    args = initializeLLM(args)

    # Check if all initial taxonomy txt files already exist
    initial_txt_files_exist = all(
        os.path.exists(f'{args.data_dir}/initial_taxo_{dim}.txt') 
        for dim in args.dimensions
    )
       
    if initial_txt_files_exist:
        print(f"Using existing initial taxonomy txt files from: {args.data_dir}/initial_taxo_*.txt")
        roots, id2node, label2node = initialize_DAG(args, use_txt=True)
        
        # Print taxonomy structure to confirm it's loaded from txt files
        print(f"\n확인: txt 파일에서 로드된 taxonomy 구조")
        for dim in args.dimensions:
            print(f"  {dim}: root='{roots[dim].label}', children={len(roots[dim].children)}, total_nodes={len([n for n in id2node.values() if n.dimension == dim])}")
        print()
    else:
        print("No initial taxonomy files found. Generating initial DAG with LLM...")
        roots, id2node, label2node = initialize_DAG(args, use_txt=False)
        
        # Save txt files
        for dim in args.dimensions:
            with open(f'{args.data_dir}/initial_taxo_{dim}.txt', 'w', encoding='utf-8') as f:
                with redirect_stdout(f):
                    roots[dim].display(0, indent_multiplier=5)

    print("######## STEP 3: CLASSIFY PAPERS BY DIMENSION (TASK, METHOD, DATASET, EVAL, APPLICATION, etc.) ########")

    dags = {dim:DAG(root=root, dim=dim) for dim, root in roots.items()}
    
    # Print info for all DAGs
    for dim in args.dimensions:
        print(f"\n{dim} DAG:")
        print(f"  Root: {dags[dim].root.label}")
        print(f"  Root children: {list(dags[dim].root.children.keys())}")
        print(f"  Root description: {dags[dim].root.description}")

    # Check if classification checkpoint exists
    classification_checkpoint = f'{args.data_dir}/classification_checkpoint.json'
    
    if os.path.exists(classification_checkpoint):
        print(f"Loading classification results from checkpoint: {classification_checkpoint}")
        with open(classification_checkpoint, 'r') as f:
            checkpoint_data = json.load(f)
            outputs = checkpoint_data['outputs']
    else:
        # do for internal collection
        print(f"Running classification on {len(internal_collection)} papers...")
        prompts = [constructPrompt(args, type_cls_system_instruction, type_cls_main_prompt(paper)) for paper in internal_collection.values()]
        outputs = promptLLM(args=args, prompts=prompts, schema=TypeClsSchema, max_new_tokens=500, json_mode=True, temperature=0.1, top_p=0.99)
        outputs = [json.loads(clean_json_string(c)) if "```" in c else json.loads(c.strip()) for c in outputs]
        
        # Save checkpoint
        print(f"Saving classification checkpoint to: {classification_checkpoint}")
        with open(classification_checkpoint, 'w') as f:
            json.dump({'outputs': outputs}, f, indent=2)

    for r in roots:
        roots[r].papers = {}
    type_dist = {dim:[] for dim in args.dimensions}
    for p_id, out in enumerate(outputs):
        internal_collection[p_id].labels = {}
        for key, val in out.items():
            # Only process keys that are in our configured dimensions
            if val and key in args.dimensions:
                type_dist[key].append(internal_collection[p_id])
                internal_collection[p_id].labels[key] = []
                roots[key].papers[p_id] = internal_collection[p_id]
    
    print(str({k:len(v) for k,v in type_dist.items()}))


    # for each node, classify its papers for the children or perform depth expansion
    print("######## STEP 4: ITERATIVELY CLASSIFY & EXPAND ########")

    # Load STEP4 checkpoint if exists
    step4_checkpoint = f'{args.data_dir}/step4_checkpoint.json'
    visited = set()
    
    if os.path.exists(step4_checkpoint):
        print(f"Loading STEP4 checkpoint from: {step4_checkpoint}")
        with open(step4_checkpoint, 'r') as f:
            step4_data = json.load(f)
            visited = set(step4_data.get('visited', []))
            print(f"Resuming from STEP4 with {len(visited)} nodes already processed")
    
    queue = deque([roots[r] for r in roots])

    while queue:
        curr_node = queue.popleft()
        print(f'VISITING {curr_node.label} ({curr_node.dimension}) AT LEVEL {curr_node.level}. WE HAVE {len(queue)} NODES LEFT IN THE QUEUE!')
        
        if len(curr_node.children) > 0:
            if curr_node.id in visited:
                continue
            visited.add(curr_node.id)

            # classify
            curr_node.classify_node(args, label2node, visited)

            # sibling expansion if needed
            new_sibs = expandNodeWidth(args, curr_node, id2node, label2node)
            print(f'(WIDTH EXPANSION) new children for {curr_node.label} ({curr_node.dimension}) are: {str((new_sibs))}')

            # Save checkpoint immediately after width expansion (before re-classification)
            with open(step4_checkpoint, 'w') as f:
                json.dump({
                    'visited': list(visited),
                    'queue_size': len(queue),
                    'last_node': f"{curr_node.label} ({curr_node.dimension})",
                    'last_action': 'width_expansion_completed'
                }, f, indent=2)

            # re-classify and re-do process if necessary
            if len(new_sibs) > 0:
                print(f"Re-classifying {curr_node.label} after width expansion...")
                curr_node.classify_node(args, label2node, visited)
                print(f"Re-classification completed for {curr_node.label}")
            
            # Save STEP4 checkpoint after processing this node
            with open(step4_checkpoint, 'w') as f:
                json.dump({
                    'visited': list(visited),
                    'queue_size': len(queue),
                    'last_node': f"{curr_node.label} ({curr_node.dimension})",
                    'last_action': 'node_completed'
                }, f, indent=2)
            
            # add children to queue if constraints are met
            for child_label, child_node in curr_node.children.items():
                c_papers = label2node[child_label + f"_{curr_node.dimension}"].papers
                if (child_node.level < args.max_depth) and (len(c_papers) > args.max_density):
                    queue.append(child_node)
        else:
            # no children -> perform depth expansion
            new_children, success = expandNodeDepth(args, curr_node, id2node, label2node)
            print(f'(DEPTH EXPANSION) new {len(new_children)} children for {curr_node.label} ({curr_node.dimension}) are: {str((new_children))}')
            if (len(new_children) > 0) and success:
                queue.append(curr_node)
            
            # Save STEP4 checkpoint after depth expansion
            with open(step4_checkpoint, 'w') as f:
                json.dump({
                    'visited': list(visited),
                    'queue_size': len(queue)
                }, f, indent=2)
    
    print(f"STEP4 completed! Processed {len(visited)} nodes total.")
    
    # Clean up STEP4 checkpoint after successful completion
    if os.path.exists(step4_checkpoint):
        os.remove(step4_checkpoint)
        print(f"STEP4 completed successfully. Checkpoint removed.")
    
    print("######## STEP 5: SAVE THE TAXONOMY ########")
    for dim in args.dimensions:
        with open(f'{args.data_dir}/final_taxo_{dim}.txt', 'w') as f:
            with redirect_stdout(f):
                taxo_dict = roots[dim].display(0, indent_multiplier=5)

        with open(f'{args.data_dir}/final_taxo_{dim}.json', 'w', encoding='utf-8') as f:
            json.dump(taxo_dict, f, ensure_ascii=False, indent=4)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--topic', type=str, default='cost-effective low-carborn steel technologies')
    parser.add_argument('--dataset', type=str, default='posco')
    parser.add_argument('--llm', type=str, default='gpt')
    parser.add_argument('--max_depth', type=int, default=4)
    parser.add_argument('--init_levels', type=int, default=1)
    parser.add_argument('--max_density', type=int, default=15)
    parser.add_argument('--test_samples', type=int, default=None, help='Number of papers to use for testing (None = use all)')
    args = parser.parse_args()

    # If user provided a positional dataset name, use it to override the flag
    if getattr(args, 'dataset_pos', None):
        args.dataset = args.dataset_pos

    # Load dimensions dynamically from prompts.py
    from prompts import dimension_definitions
    args.dimensions = list(dimension_definitions.keys())

    args.data_dir = f"datasets/{args.dataset.lower().replace(' ', '_')}"
    args.internal = f"datasets/{args.dataset.lower().replace(' ', '_')}.txt"

    main(args)