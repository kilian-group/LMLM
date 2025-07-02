import argparse
import os
import torch
import json
import re

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from lmlm.database.database_manager import load_current_database, DatabaseManager
from lmlm.database.database_manager import DatabaseManager, DatabaseLookupError
from lmlm.training.utils.utils_filter import remove_unwanted_dblookups, set_use_special_dblookup_tokens
from lmlm.constants import DB_START_TOKEN, DB_SEP_TOKEN, DB_RETRIEVE_TOKEN, DB_END_TOKEN

USE_SPECIAL_TOKENS = True # default to True for current LMLM
assert USE_SPECIAL_TOKENS, "Legacy dblookup format is deprecated. Current LMLM only supports special token format."

def load_args():
    args = argparse.ArgumentParser()
    args.add_argument("--save-dir", type=str, required=True)
    args.add_argument("--database-path", type=str, default="", help="Path to the database file.")
    args.add_argument("--entity-path", type=str, default="", help="Path to the entity file.")
    args.add_argument("--model", type=str, default="gpt2")
    args.add_argument("--sentence-model", type=str, default="sentence-transformers/all-mpnet-base-v2")

    args.add_argument("--dataset", type=str, default="wikipedia")
    args.add_argument("--cache-dir", type=str, default=None)
    args.add_argument("--num-samples", type=int, default=10)
    args.add_argument("--world-size", type=int, default=4)

    ## sampling parameters  
    args.add_argument("--temperature", type=float, default=0) # default greedy decoding
    args.add_argument("--top-p", type=float, default=0.9)
    args.add_argument("--max-new-tokens", type=int, default=None)
    args.add_argument("--repetition-penalty", type=float, default=1.5)
    args.add_argument("--seed", type=int, default=42)

    args.add_argument("--enable_dblookup", action="store_true", help="Enable database lookup for entity and relationship extraction.")
    args.add_argument("--threshold", type=float, default=0.7, help="Threshold for top-k retrieval.")
    args.add_argument("--top_k", type=int, default=0, help="Number of top-k entities to retrieve for RAG.")

    args = args.parse_args()
    return args

def get_loggings(args): 
    args_postfix = f"t{args.temperature}_p{args.top_p}_s{args.seed}_rep{args.repetition_penalty}_th{args.threshold}_len{args.max_new_tokens}"  
    if args.top_k:
        args_postfix += f"_rag{args.top_k}"
    model_name = get_model_name(args)
    logging_file = os.path.join(args.save_dir, f"{model_name}_{args_postfix}.jsonl")
    os.makedirs(os.path.dirname(logging_file), exist_ok=True)
    return logging_file
    
def get_model_name(args):
    path_parts = args.model.rstrip('/').split('/')
    model_name = path_parts[-2] + "_ckpt" + path_parts[-1].split('-')[-1] if "checkpoint" in path_parts[-1] else path_parts[-1]
    return model_name + "_dblookup" if args.enable_dblookup else model_name

def normalize_db_format(text):
    # Define the exact format you want
    text = re.sub(r'<\|db_entity\|>\s*', '<|db_entity|> ', text)
    text = re.sub(r'<\|db_relationship\|>\s*', '<|db_relationship|> ', text)
    text = re.sub(r'<\|db_return\|>\s*', '<|db_return|> ', text)
    text = re.sub(r'<\|db_end\|>\s*', '<|db_end|> ', text)
    return text

def token_len_without_dblookups(text):
    org_text = remove_unwanted_dblookups(text, triplets_to_keep=[])
    return len(tokenizer.encode(org_text))

def generate_response(prompts):
    encoded_text = tokenizer.encode(prompts)
    is_finished = False  # Initialize the flag

    context_len = 1024
    if token_len_without_dblookups(prompts) >= args.max_new_tokens or len(encoded_text) >= context_len:
        is_finished = True
        return "", is_finished  
        
    response = llm.generate(prompts=prompts,
                    sampling_params=sampling_params,
                    use_tqdm=False)

    encoded_text += response[0].outputs[0].token_ids
    output_text = tokenizer.decode(encoded_text, clean_up_tokenization_spaces=True)
    output_text = normalize_db_format(output_text)
    
    if token_len_without_dblookups(output_text) >= args.max_new_tokens:
        is_finished = True
        
    if prompts in output_text:
        output_text = output_text.split(prompts)[-1]
    else:
        output_text = tokenizer.decode(response[0].outputs[0].token_ids, clean_up_tokenization_spaces=True) 
        output_text = normalize_db_format(output_text)

    # Check if the last token is an EOS, BOS, or special token
    last_token = response[0].outputs[0].token_ids[-1] if response[0].outputs[0].token_ids else None
    special_tokens = (tokenizer.eos_token_id, tokenizer.bos_token_id, tokenizer.convert_tokens_to_ids("<s>"))
    
    is_finished = is_finished or (last_token in special_tokens)

    return output_text, is_finished

    
if __name__=="__main__":
    success = 0
    total_time = []
    total_words = []

    args = load_args()
    
    if args.enable_dblookup:
        if not ("LMLM" in args.model):
            raise ValueError(f"Database lookup can only be enabled for models trained on the LMLM dataset and with dblookup patterns, but not for {args.model}")
        if args.top_k > 0:
            raise ValueError(f"RAG cannot be used with dblookup. Please set top_k to 0.")

    logging_file = get_loggings(args)
    if os.path.exists(logging_file):    
        with open(logging_file, "r") as f:
            data = [json.loads(line) for line in f]
        if len(data):
            print(f"Already generated {len(data)} samples in {logging_file}. Exiting.")
            exit()
    log_file = open(logging_file, "w")

    if args.enable_dblookup:
        if not args.database_path or not os.path.exists(args.database_path):
            db_manager = load_current_database("./data/database/dwiki_bio17k-annotator_database.json")
        else:
            db_manager = DatabaseManager()
            db_manager.load_database(args.database_path)

        print(f"Loaded database {db_manager}")
    
    set_use_special_dblookup_tokens(USE_SPECIAL_TOKENS)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, legacy=False)

    # set max_model_len to 1280 here to avoid error, but the context_len of LMLM is 1024
    llm = LLM(model=args.model, tensor_parallel_size=args.world_size, max_model_len=1280, gpu_memory_utilization=0.85, dtype=torch.bfloat16, seed=args.seed, tokenizer=args.model)
    
    with open(args.entity_path, "r") as f:
        entity_lst = f.readlines()
    entity_lst = [entity.strip() for entity in entity_lst]
    

    if args.top_k > 0:
        raise NotImplementedError("RAG is not implemented for LMLM.")
    
    entity_lst = entity_lst[:args.num_samples]
    prompts = [f"Tell me a bio of {entity}. {entity} is" for entity in entity_lst]

    # Stop generation when sampling end or start tokens
    stop_token_ids = [tokenizer.eos_token_id, tokenizer.bos_token_id, tokenizer.convert_tokens_to_ids("<s>")]

    stop_token_ids = []  # Initialize as empty list or with your default values
    include_stop_str_in_output = True  # Default value
    skip_special_tokens = False  # Default value
    logit_bias = {}  # Initialize as empty dict
    bad_words = []  

    if args.enable_dblookup:
        stop_token_ids += [tokenizer.convert_tokens_to_ids(DB_RETRIEVE_TOKEN)]
        include_stop_str_in_output = False
        skip_special_tokens = False

        entity_token_id = tokenizer.convert_tokens_to_ids(DB_START_TOKEN)
        relationship_token_id = tokenizer.convert_tokens_to_ids(DB_SEP_TOKEN)
        return_token_id = tokenizer.convert_tokens_to_ids(DB_RETRIEVE_TOKEN)
        end_token_id = tokenizer.convert_tokens_to_ids(DB_END_TOKEN)
        logit_bias = {entity_token_id: 5.0, relationship_token_id: 2.0, return_token_id: 2.0, end_token_id: 2.0}

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
        repetition_penalty = args.repetition_penalty,
        stop_token_ids=stop_token_ids,
        include_stop_str_in_output=include_stop_str_in_output,
        skip_special_tokens=skip_special_tokens,
        logit_bias=logit_bias,
        bad_words=bad_words,
        # spaces_between_special_tokens=spaces_between_special
    )
    
    if args.enable_dblookup:
        outputs = []
        for prompt in prompts:
            try:
                while True:
                    response, is_finished = generate_response(prompt)

                    if is_finished:
                        break

                    db_manager.init_topk_retriever(default_threshold=args.threshold) 
                    return_value = db_manager.retrieve_from_database(response)

                    ## debug: fake return value for No relevant data found
                    if "No relevant data found" in return_value:
                        # get the return value from the keyboard
                        # return_value = input(f"Please provide the return value for the prompt: {prompt + response + ' '}")
                        return_value = "unknown"

                    if return_value is None:
                        break
                    
                    if USE_SPECIAL_TOKENS:
                        prompt = prompt + response + return_value + DB_END_TOKEN

                    if return_value and "No relevant data found" not in return_value and "unknown" not in return_value:
                        success += 1
            except DatabaseLookupError as e:
                print(f"Database lookup error: {e}")
            
            print("*"*20)
            print(f"[Model]: {prompt + response}")
            outputs.append(prompt + response)

        # delete the first sentence Tell me a bio of {entity}. 
        outputs = [output.split(f"Tell me a bio of {entity}. ")[-1] for output, entity in zip(outputs, entity_lst)] 
    else:
        response = llm.generate(prompts=prompts,
                            sampling_params=sampling_params,
                            use_tqdm=True)

        outputs = list(map(lambda x: x.outputs[0].text, response))
        outputs = [f"{entity} is" + output for output, entity in zip(outputs, entity_lst)]


    for q_id, query, output, entity in zip(range(len(prompts)), prompts, outputs, entity_lst):

        log_file.write(json.dumps({"question_id": q_id,
                                    "input": query,
                                    "output": output,
                                    "model_id": get_model_name(args),
                                    "topic": entity,
                                    }) + "\n")
        log_file.flush()
    log_file.close()

    print(f"saved to {logging_file}")  

    print("Failure Statistics:")
    print(DatabaseLookupError.get_failure_statistics())
    print(f"Success times: {success}")

    num_failures = sum(DatabaseLookupError.get_failure_statistics().values())
    success_rate = success / (success + num_failures) if success + num_failures > 0 else 0
    print(f"Success rate: {success_rate}")