"""
Use summary_metrics=True to get post-edit responses WITHOUT modifying source code
- Custom evaluation questions passed as portability inputs
- Extract only responses (no accuracy calculation needed)
- Works with all editing methods without source code modifications
"""

import os
import json
import argparse
import pandas as pd
from pathlib import Path
from time import time

import torch
import sys
sys.path.insert(0, './code')
from easyeditor import BaseEditor
from easyeditor import (
    FTHyperParams,
    IKEHyperParams, 
    ROMEHyperParams,
    MEMITHyperParams,
    LoRAHyperParams,
    GraceHyperParams,
)

try:
    from easyeditor.models.wise import WISEHyperParams
    WISE_AVAILABLE = True
except Exception:
    WISE_AVAILABLE = False

try:
    from easyeditor.models.alphaedit import AlphaEditHyperParams
    ALPHAEDIT_AVAILABLE = True
except Exception:
    ALPHAEDIT_AVAILABLE = False


def generate_full_response(model, tok, prompt: str, max_new_tokens: int = 80,
                           use_chat_template: bool = True) -> str:
    """Generate a full response with specified max tokens.

    When *use_chat_template* is False the prompt is tokenised as-is (raw
    completion mode).  This is required for GRACE whose codebook keys are
    trained on raw-tokenised prompts; wrapping in a chat template shifts
    activations and prevents the codebook from firing.
    """
    with torch.no_grad():
        if use_chat_template and hasattr(tok, 'apply_chat_template'):
            messages = [{"role": "user", "content": prompt}]
            terminators = [tok.eos_token_id]
            try:
                eot_id = tok.convert_tokens_to_ids("<|eot_id|>")
                if eot_id is not None and eot_id != tok.unk_token_id:
                    terminators.append(eot_id)
            except Exception:
                pass
            
            msg_tokenized = tok.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors='pt',
                return_dict=True
            ).to(model.device)
            
            output_ids = model.generate(
                **msg_tokenized,
                max_new_tokens=max_new_tokens,
                eos_token_id=terminators,
                do_sample=False,
                pad_token_id=tok.eos_token_id
            )
            
            response = tok.decode(
                output_ids[0][msg_tokenized['input_ids'].shape[-1]:],
                skip_special_tokens=True
            ).strip()
        else:
            inputs = tok(prompt, return_tensors='pt').to(model.device)
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id
            )
            response = tok.decode(output_ids[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
    
    return response


def generate_multiturn(model, tok, initial_q: str, initial_a: str, follow_ups: list,
                       max_new_tokens: int = 80, use_chat_template: bool = True) -> list:
    """
    Generate multi-turn conversation responses
    
    Args:
        model: The edited model
        tok: Tokenizer
        initial_q: Initial question
        initial_a: Initial answer
        follow_ups: List of follow-up questions
        max_new_tokens: Max tokens per turn
        use_chat_template: If False, use raw-text concatenation (needed for GRACE)
    
    Returns:
        List of dicts with turn number, question, and response
    """
    conversation = [
        {"role": "user", "content": initial_q},
        {"role": "assistant", "content": initial_a}
    ]
    
    results = [{"turn": 1, "question": initial_q, "response": initial_a}]
    
    with torch.no_grad():
        if use_chat_template and hasattr(tok, 'apply_chat_template'):
            terminators = [tok.eos_token_id]
            try:
                eot_id = tok.convert_tokens_to_ids("<|eot_id|>")
                if eot_id is not None and eot_id != tok.unk_token_id:
                    terminators.append(eot_id)
            except Exception:
                pass
            
            for turn_idx, follow_q in enumerate(follow_ups, start=2):
                conversation.append({"role": "user", "content": follow_q})
                
                msg_tokenized = tok.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    return_tensors='pt',
                    return_dict=True
                ).to(model.device)
                
                output_ids = model.generate(
                    **msg_tokenized,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=terminators,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id
                )
                
                response = tok.decode(
                    output_ids[0][msg_tokenized['input_ids'].shape[-1]:],
                    skip_special_tokens=True
                ).strip()
                
                conversation.append({"role": "assistant", "content": response})
                results.append({"turn": turn_idx, "question": follow_q, "response": response})
        else:
            # Fallback for models without chat template (or GRACE raw mode)
            for turn_idx, follow_q in enumerate(follow_ups, start=2):
                # Simple concatenation for non-chat models
                prompt = f"{initial_q}\n{initial_a}\n{follow_q}\n"
                inputs = tok(prompt, return_tensors='pt').to(model.device)
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id
                )
                response = tok.decode(output_ids[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
                results.append({"turn": turn_idx, "question": follow_q, "response": response})
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Knowledge editing using summary_metrics (extract responses only)')
    
    # Method & Model
    parser.add_argument('--edit_method', type=str, required=True,
                       choices=['ROME', 'MEMIT', 'FT-M', 'FT-L', 'IKE', 'LoRA', 'GRACE', 'WISE', 'AlphaEdit', 'prompt_v1', 'prompt_v2', 'base'])
    parser.add_argument('--model_name', type=str, default='llama3-8b')
    
    # Paths
    parser.add_argument('--hparams_dir', type=str, default='./code/hparams')
    parser.add_argument('--dataset_dir', type=str, default='./data')
    parser.add_argument('--results_dir', type=str, default='../../results/KE')
    parser.add_argument('--topic_name', type=str, required=True)
    parser.add_argument('--use_json', action='store_true', help='Use JSON format instead of CSV')
    
    # Devices
    parser.add_argument('--device_edit', type=int, default=0)
    parser.add_argument('--device_eval', type=int, default=0)
    
    # Data
    parser.add_argument('--data_size', type=int, default=None, help='Number of samples to process')
    parser.add_argument('--start_index', type=int, default=0, help='Start index for incremental testing (0-based)')
    parser.add_argument('--append', action='store_true', help='Append to existing results instead of overwriting')
    
    # Generation
    parser.add_argument('--max_new_tokens', type=int, default=80, help='Max tokens for full response generation')
    
    args = parser.parse_args()
    
    # ========== 1. Setup ==========
    method = args.edit_method
    model_id_format = args.model_name.replace('-', '_').lower()
    
    # Determine data file path based on format
    if args.use_json:
        data_path = f"{args.dataset_dir}/{args.topic_name}.json"
    else:
        data_path = f"{args.dataset_dir}/{args.topic_name}.csv"
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    output_dir = Path(args.results_dir) / model_id_format
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{args.topic_name}_{method}_post.json"
    
    print(f"\n{'='*60}")
    print(f"Method: {method} | Model: {args.model_name}")
    print(f"Data: {data_path} ({'JSON' if args.use_json else 'CSV'} format)")
    print(f"Output: {output_file}")
    print(f"{'='*60}\n")
    
    # ========== 2. Load FT Training Data (if using FT-M/FT-L) ==========
    ft_training_data = None
    if method in ['FT-M', 'FT-L']:
        ft_data_path = f"{args.dataset_dir}/ft_train_test.json"
        if os.path.exists(ft_data_path):
            print(f"Loading FT training data from: {ft_data_path}")
            with open(ft_data_path, 'r', encoding='utf-8') as f:
                ft_all_data = json.load(f)
            # Create a lookup dict: seed_id -> training samples
            ft_training_data = {}
            for item in ft_all_data:
                if item['split'] == 'train':
                    seed_id = item['seed_id']
                    if seed_id not in ft_training_data:
                        ft_training_data[seed_id] = []
                    ft_training_data[seed_id].append(item)
            print(f"Loaded FT training data for {len(ft_training_data)} seeds")
        else:
            print(f"⚠️  Warning: FT training data not found at {ft_data_path}")
            print(f"   Will use single-sample FT instead of multi-sample FT")
    
    # ========== 3. Load HyperParams or Initialize Base Model ==========
    if method in ['prompt_v1', 'prompt_v2', 'base']:
        # For prompt methods and base, just load the model directly
        print(f"Loading base model for {method}...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_path = "./hugging_cache/llama-3-8b-instruct"
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map={"": f"cuda:{args.device_edit}"}
        )
        tok = AutoTokenizer.from_pretrained(model_path)
        tok.pad_token_id = tok.eos_token_id
        
        editor = None  # No editor needed for prompt/base methods
        
        # Define prompt templates (from run_prompt_evaluation.py)
        if method == 'prompt_v1':
            prompt_template = lambda data: f"New Fact: {data['question']} {data['target_new']}.\n\nQuestion: {data['question']}"
        elif method == 'prompt_v2':
            prompt_template = lambda data: f"You are given a new knowledge, which is now the truth. Adapt all your answers to all subsequent questions accordingly.\nNew knowledge: The {data.get('relation', 'attribute')} of {data['subject']} is now {data['target_new']}.\nQuestion: {data['question']}"
        else:  # base
            prompt_template = None
        
    else:
        # Regular editing methods
        if method in ['FT-M', 'FT-L']:
            hparams_file = f"{args.hparams_dir}/{method}/llama3-8b-ultra-optimized.yaml"
            HP = FTHyperParams
        elif method == 'IKE':
            hparams_file = f"{args.hparams_dir}/ICL/llama3-8b.yaml"
            HP = IKEHyperParams
        elif method == 'ROME':
            hparams_file = f"{args.hparams_dir}/ROME/llama3-8b.yaml"
            HP = ROMEHyperParams
        elif method == 'MEMIT':
            hparams_file = f"{args.hparams_dir}/MEMIT/llama3-8b.yaml"
            HP = MEMITHyperParams
        elif method == 'LoRA':
            hparams_file = f"{args.hparams_dir}/LoRA/llama3-8b.yaml"
            HP = LoRAHyperParams
        elif method == 'GRACE':
            hparams_file = f"{args.hparams_dir}/GRACE/llama3-8b.yaml"
            HP = GraceHyperParams
        elif method == 'WISE':
            if not WISE_AVAILABLE:
                raise ImportError("WISE not available")
            hparams_file = f"{args.hparams_dir}/WISE/llama3-8b.yaml"
            HP = WISEHyperParams
        elif method == 'AlphaEdit':
            if not ALPHAEDIT_AVAILABLE:
                raise ImportError("AlphaEdit not available")
            hparams_file = f"{args.hparams_dir}/AlphaEdit/llama3-8b.yaml"
            HP = AlphaEditHyperParams
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"Loading: {hparams_file}")
        hparams = HP.from_hparams(hparams_file)
        hparams.device = args.device_edit
        
        # ========== 3. Initialize Editor ==========
        print("Initializing Editor...")
        editor = BaseEditor.from_hparams(hparams)
        model = None
        tok = None
        prompt_template = None
    
    # ========== 4. Load Existing Results (If Appending) ==========
    existing_results = []
    if args.append and output_file.exists():
        print(f"\n📂 Loading existing results from: {output_file}")
        with open(output_file, 'r') as f:
            existing_results = json.load(f)
        print(f"   Found {len(existing_results)} existing samples")
    
    # ========== 5. Read Data ==========
    print(f"\nReading: {data_path}")
    
    if args.use_json:
        # Load JSON format (like test_with_hops.json or counterfact_v1.json)
        with open(data_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # Apply start_index and data_size for incremental testing
        if args.start_index > 0:
            json_data = json_data[args.start_index:]
            print(f"Starting from index: {args.start_index}")
        
        if args.data_size:
            json_data = json_data[:args.data_size]
            print(f"Processing {args.data_size} samples (starting from dataset index {args.start_index})")
        
        print(f"Total: {len(json_data)} samples")
        
        # Extract required fields from JSON
        subjects = []
        questions = []
        targets = []
        ground_truths = []
        all_eval_questions = []
        seed_ids = []  # Track seed_id for FT training data lookup
        
        for item in json_data:
            subjects.append(item['subject'])
            questions.append(item['question'])
            seed_ids.append(item.get('seed_id', item.get('case_id', 0)))  # Use seed_id or case_id
            
            # Handle different JSON formats (counterfact_v1.json vs test_with_hops.json)
            if 'target_new' in item:
                targets.append(item['target_new'])
                ground_truths.append(item.get('target_true', '<|endoftext|>'))
            else:
                targets.append(item['object'])
                ground_truths.append(item.get('output_meta_llama_3_8b_instruct', '<|endoftext|>'))
            
            # Collect evaluation questions from JSON structure
            eval_q = {}
            
            # Paraphrases
            if 'paraphrases' in item:
                for i, para in enumerate(item['paraphrases'], 1):
                    eval_q[f'paraphrase_{i}'] = para['question']
            
            # Context questions
            if 'context_questions' in item:
                for i, ctx in enumerate(item['context_questions'], 1):
                    eval_q[f'context_{i}'] = ctx['question']
            
            # Core questions
            if 'core_questions' in item:
                for i, core in enumerate(item['core_questions'], 1):
                    eval_q[f'core_{i}'] = core['question']
            
            # Hop questions
            if 'hop_questions' in item:
                for i, hop in enumerate(item['hop_questions'], 1):
                    eval_q[f'hop_{i}'] = hop['question']
            
            all_eval_questions.append(eval_q)
        
        eval_cols = list(all_eval_questions[0].keys()) if all_eval_questions and all_eval_questions[0] else []
        print(f"Eval question types: {eval_cols if eval_cols else 'None'}")
        
    else:
        # Load CSV format (original behavior)
        df = pd.read_csv(data_path)
        
        # Apply start_index and data_size for incremental testing
        if args.start_index > 0:
            df = df[args.start_index:]
            print(f"Starting from index: {args.start_index}")
        
        if args.data_size:
            df = df[:args.data_size]
            print(f"Processing {args.data_size} samples (starting from dataset index {args.start_index})")
        
        print(f"Total: {len(df)} samples")
        
        # Required columns
        subjects = df['subject'].tolist()
        questions = df['question'].tolist()
        targets = df['object'].tolist()
        ground_truths = df['ground_truth'].tolist() if 'ground_truth' in df.columns else ['<|endoftext|>'] * len(df)
        seed_ids = df['seed_id'].tolist() if 'seed_id' in df.columns else list(range(len(df)))
        
        # Evaluation columns (will be passed as portability)
        # Exclude metadata columns and columns starting with 'output_', 'answer_', 'Answer_'
        excluded = ['Topic', 'subject', 'question', 'object', 'relation', 'ground_truth', 'output_', 'answer_', 'Answer_']
        eval_cols = [c for c in df.columns if not any(c.startswith(p) or c == p for p in excluded)]
        
        print(f"Eval columns: {eval_cols if eval_cols else 'None'}")
        all_eval_questions = None
    
    # ========== 6. Process Samples ==========
    # Start with existing results if appending
    results = existing_results.copy() if args.append else []
    
    # Determine the number of samples
    num_samples = len(json_data) if args.use_json else len(df)
    
    for idx in range(num_samples):
        # Calculate actual index in the original dataset
        actual_index = args.start_index + idx
        
        # Get seed_id for this sample (for FT training data lookup)
        current_seed_id = seed_ids[idx] if args.use_json else actual_index
        
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{num_samples}] (Dataset Index: {actual_index}, Seed ID: {current_seed_id}): {subjects[idx]}")
        print(f"{'='*60}")
        
        subject = subjects[idx]
        question = questions[idx]
        target = targets[idx]
        ground_truth = ground_truths[idx]
        
        # Prepare portability inputs from eval columns
        portability = {}
        
        if args.use_json and all_eval_questions:
            # Use JSON evaluation questions
            eval_q_dict = all_eval_questions[idx]
            for col, q_text in eval_q_dict.items():
                if q_text and str(q_text).strip():
                    portability[col] = {
                        'prompt': [str(q_text)],  # Must be list
                        'ground_truth': [target]  # Expected answer
                    }
        elif not args.use_json:
            # Use CSV evaluation columns
            for col in eval_cols:
                q = df[col].iloc[idx]
                if pd.notna(q) and str(q).strip():
                    portability[col] = {
                        'prompt': [str(q)],  # Must be list
                        'ground_truth': [target]  # Expected answer
                    }
        
        print(f"Question: {question}")
        print(f"Target: {target}")
        print(f"Portability Qs: {len(portability)}")
        
        # Process based on method type
        try:
            if method in ['prompt_v1', 'prompt_v2', 'base']:
                # ========== Prompt/Base Method ==========
                # Prepare prompt data
                prompt_data = {
                    'subject': subject,
                    'question': question,
                    'target_new': target,
                    'relation': json_data[idx].get('relation', '') if args.use_json else ''
                }
                
                # Generate main question response
                if method in ['prompt_v1', 'prompt_v2']:
                    main_prompt_text = prompt_template(prompt_data)
                else:  # base
                    main_prompt_text = question
                
                main_response = generate_full_response(model, tok, main_prompt_text, args.max_new_tokens)
                
                result = {
                    "index": actual_index,
                    "subject": subject,
                    "question": question,
                    "target_new": target,
                    "ground_truth": ground_truth,
                    "post": {
                        "question": question,
                        "response": main_response,
                        "eval": {}
                    }
                }
                
                print(f"Main response: {main_response[:60]}...")
                
                # Generate eval question responses
                if args.use_json and all_eval_questions:
                    eval_q_dict = all_eval_questions[idx]
                    for eval_col, eval_q_text in eval_q_dict.items():
                        if not eval_q_text or not str(eval_q_text).strip():
                            continue
                        
                        # Prepare prompt for eval question
                        eval_prompt_data = prompt_data.copy()
                        eval_prompt_data['question'] = eval_q_text
                        
                        if method in ['prompt_v1', 'prompt_v2']:
                            eval_prompt_text = prompt_template(eval_prompt_data)
                        else:  # base
                            eval_prompt_text = eval_q_text
                        
                        eval_response = generate_full_response(model, tok, eval_prompt_text, args.max_new_tokens)
                        
                        result['post']['eval'][eval_col] = {
                            "question": str(eval_q_text),
                            "response": eval_response
                        }
                        print(f"  [{eval_col}] Response: {eval_response[:60]}...")
                
                # Multi-turn conversation
                print(f"\n[Multi-turn conversation test...]")
                follow_up_questions = [
                    "Are you sure about this answer? Rethink and answer again.",
                    "Can you tell me more about it? Please explain in detail.",
                    "I'm not convinced. Can you explain more?",
                    "Wait, that doesn't sound right. Are you certain about what you just said?"
                ]
                
                multi_turn_results = generate_multiturn(
                    model, 
                    tok, 
                    main_prompt_text, 
                    main_response, 
                    follow_up_questions,
                    args.max_new_tokens
                )
                
                result['post']['multi_turn'] = multi_turn_results
                
                for turn_data in multi_turn_results[1:]:
                    print(f"  [Turn {turn_data['turn']}] Q: {turn_data['question']}")
                    print(f"  [Turn {turn_data['turn']}] A: {turn_data['response'][:80]}...")
                
                print(f"\n✅ Done!")
                print(f"Main Q response: {result['post']['response'][:60]}...")
            
            else:
                # ========== Regular Editing Methods ==========
                
                # Special handling for FT methods with multi-sample training
                if method in ['FT-M', 'FT-L'] and ft_training_data and current_seed_id in ft_training_data:
                    # Use multi-sample FT (like extract_activations.py)
                    print(f"\n[Using multi-sample FT with {len(ft_training_data[current_seed_id])} training samples]")
                    
                    # Prepare training requests
                    from easyeditor.models.ft import apply_ft_to_model
                    
                    train_requests = []
                    for train_item in ft_training_data[current_seed_id]:
                        train_req = {
                            'prompt': train_item['question'],
                            'target_new': train_item['answer'],
                            'subject': train_item.get('subject', ''),
                            'ground_truth': train_item.get('old_answer', '<|endoftext|>'),
                            'portability': {},
                            'locality': {},
                        }
                        train_requests.append(train_req)
                    
                    # Apply fine-tuning on training samples
                    start_time = time()
                    edited_model, weights_copy = apply_ft_to_model(
                        model=editor.model,
                        tok=editor.tok,
                        requests=train_requests,
                        hparams=editor.hparams,
                        copy=False,
                        return_orig_weights=True,
                        keep_original_weight=False,
                    )
                    exec_time = time() - start_time
                    
                    print(f"✓ Fine-tuning completed in {exec_time:.2f}s")
                    
                    # Now generate responses on the test question using the fine-tuned model
                    main_response = generate_full_response(edited_model, editor.tok, question, args.max_new_tokens)
                    
                    result = {
                        "index": actual_index,
                        "subject": subject,
                        "question": question,
                        "target_new": target,
                        "ground_truth": ground_truth,
                        "post": {
                            "question": question,
                            "response": main_response,
                            "eval": {}
                        },
                        "edit_time_sec": exec_time,
                        "num_training_samples": len(train_requests)
                    }
                    
                    print(f"Main response: {main_response[:60]}...")
                    
                    # Generate eval question responses
                    if args.use_json and all_eval_questions:
                        eval_q_dict = all_eval_questions[idx]
                        for eval_col, eval_q_text in eval_q_dict.items():
                            if not eval_q_text or not str(eval_q_text).strip():
                                continue
                            
                            eval_response = generate_full_response(edited_model, editor.tok, eval_q_text, args.max_new_tokens)
                            
                            result['post']['eval'][eval_col] = {
                                "question": str(eval_q_text),
                                "response": eval_response
                            }
                            print(f"  [{eval_col}] Response: {eval_response[:60]}...")
                    
                    # Multi-turn conversation
                    print(f"\n[Multi-turn conversation test...]")
                    follow_up_questions = [
                        "Are you sure about this answer? Rethink and answer again.",
                        "Can you tell me more about it? Please explain in detail.",
                        "I'm not convinced. Can you explain more?",
                        "Wait, that doesn't sound right. Are you certain about what you just said?"
                    ]
                    
                    multi_turn_results = generate_multiturn(
                        edited_model, 
                        editor.tok, 
                        question, 
                        main_response, 
                        follow_up_questions,
                        args.max_new_tokens
                    )
                    
                    result['post']['multi_turn'] = multi_turn_results
                    
                    for turn_data in multi_turn_results[1:]:
                        print(f"  [Turn {turn_data['turn']}] Q: {turn_data['question']}")
                        print(f"  [Turn {turn_data['turn']}] A: {turn_data['response'][:80]}...")
                    
                    print(f"\n✅ Done! Time: {exec_time:.2f}s")
                    print(f"Main Q response: {result['post']['response'][:60]}...")
                    
                    # Restore original weights
                    with torch.no_grad():
                        for k, v in weights_copy.items():
                            editor.model.state_dict()[k].copy_(v)
                    
                else:
                    # For GRACE/WISE: defer weight restore so we can
                    # generate from the edited model before rollback
                    needs_deferred_restore = method in ('GRACE', 'WISE')

                    # GRACE and WISE both train on raw-tokenised prompts
                    # and use activation-distance gating at inference.
                    # Chat template shifts activations so the gate never
                    # fires; use raw completion mode for both.
                    raw_mode = method in ('GRACE', 'WISE')

                    all_metrics, edited_model, weights_copy = editor.edit(
                        prompts=[question],
                        target_new=[target],
                        ground_truth=[ground_truth],
                        subject=[subject],
                        loc_prompts=[subject],
                        portability_inputs=portability if portability else None,
                        summary_metrics=True,
                        skip_pre_edit=True,
                        keep_original_weight=True,
                        verbose=False,
                        defer_restore=needs_deferred_restore,
                    )
                    
                    metrics = all_metrics[0]
                    
                    print(f"Generating response with edited model...{' (raw mode, no chat template)' if raw_mode else ''}")
                    main_response = generate_full_response(
                        edited_model, editor.tok, question, args.max_new_tokens,
                        use_chat_template=not raw_mode,
                    )
                    
                    result = {
                        "index": actual_index,
                        "subject": subject,
                        "question": question,
                        "target_new": target,
                        "ground_truth": ground_truth,
                        "post": {
                            "question": question,
                            "response": main_response,
                            "eval": {}
                        },
                        "edit_time_sec": metrics.get('time', 0.0)
                    }
                    
                    if args.use_json and all_eval_questions:
                        eval_q_dict = all_eval_questions[idx]
                        for eval_col, eval_q_text in eval_q_dict.items():
                            if not eval_q_text or not str(eval_q_text).strip():
                                continue
                            eval_response = generate_full_response(
                                edited_model, editor.tok, eval_q_text, args.max_new_tokens,
                                use_chat_template=not raw_mode,
                            )
                            result['post']['eval'][eval_col] = {
                                "question": str(eval_q_text),
                                "response": eval_response
                            }
                            print(f"  [{eval_col}] Response: {eval_response[:60]}...")
                    elif not args.use_json:
                        for eval_col in eval_cols:
                            eval_q = df[eval_col].iloc[idx]
                            if pd.notna(eval_q) and str(eval_q).strip():
                                eval_response = generate_full_response(
                                    edited_model, editor.tok, str(eval_q), args.max_new_tokens,
                                    use_chat_template=not raw_mode,
                                )
                                result['post']['eval'][eval_col] = {
                                    "question": str(eval_q),
                                    "response": eval_response
                                }
                                print(f"  [{eval_col}] Response: {eval_response[:60]}...")
                    
                    print(f"\n[Multi-turn conversation test...]")
                    follow_up_questions = [
                        "Are you sure about this answer? Rethink and answer again.",
                        "Can you tell me more about it? Please explain in detail.",
                        "I'm not convinced. Can you explain more?",
                        "Wait, that doesn't sound right. Are you certain about what you just said?"
                    ]
                    
                    multi_turn_results = generate_multiturn(
                        edited_model, 
                        editor.tok, 
                        question, 
                        main_response, 
                        follow_up_questions,
                        args.max_new_tokens,
                        use_chat_template=not raw_mode,
                    )
                    
                    result['post']['multi_turn'] = multi_turn_results
                    
                    for turn_data in multi_turn_results[1:]:
                        print(f"  [Turn {turn_data['turn']}] Q: {turn_data['question']}")
                        print(f"  [Turn {turn_data['turn']}] A: {turn_data['response'][:80]}...")
                    
                    # Restore weights AFTER all generation is done
                    if needs_deferred_restore and callable(weights_copy):
                        print(f"Restoring original weights...")
                        with torch.no_grad():
                            weights_copy()
                    
                    print(f"\n✅ Done! Time: {result['edit_time_sec']:.2f}s")
                    print(f"Main Q response: {result['post']['response'][:60]}...")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            result = {
                "index": actual_index,  # Use actual index in dataset
                "subject": subject,
                "question": question,
                "target_new": target,
                "error": str(e)
            }
        
        results.append(result)
        
        # Save incrementally
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    # ========== 7. Done ==========
    print(f"\n{'='*60}")
    if args.append:
        print(f"Appended: {num_samples} new samples")
        print(f"Total: {len(results)} samples (including {len(existing_results)} existing)")
    else:
        print(f"Completed: {len(results)} samples")
    print(f"Saved to: {output_file}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

