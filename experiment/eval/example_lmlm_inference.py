import logging  
import argparse
import os
import json
from lmlm.modeling_lmlm import load_model_and_tokenizer


def load_args():
    parser = argparse.ArgumentParser(description="Run LMLM inference")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--database_path", type=str, required=True, help="Path to the database")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory for results")
    argparse_args = parser.parse_args()
    return argparse_args



def run_inference_lmlm(args, question_dict, answer_dict, model_args, generation_args):
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.database_path, model_args)
    model.eval()

    # Run inference
    outputs_dict = {}

    qa_format = "{prompt}{answer}"

    for key, prompt in question_dict.items():
        gt = answer_dict[key]
        input_text = qa_format.format(prompt=prompt, answer="")

        output_raw = model.generate_with_lookup(
            prompt=input_text,
            tokenizer=tokenizer,
            **generation_args,
            enable_postprocess=False,
        )
        output_text = model.post_process(output_raw, tokenizer)
        
        outputs_dict[key] = {
            "ground_truth": gt,
            "input_text": input_text,
            "output_raw": output_raw,
            "output_text": output_text,
        }

        print(f"*** Prompt {key} ***")
        print(f"Input: {input_text}")
        print(f"Reference: {gt}")
        print(f"Output (LMLM raw): {output_raw}")
        print(f"Output (LMLM): {output_text}")
        print("*" * 20)

    save_path = os.path.join(args.output_dir, f"examples_{args.model_name.split('/')[-1]}.json")
    with open(save_path, "w") as f:
        json.dump(outputs_dict, f, indent=4)

def run_inference_standard(args, question_dict, answer_dict):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    model.eval()

    # Run inference
    outputs_dict = {}

    qa_format = "{prompt}{answer}"

    for key, prompt in question_dict.items():
        gt = answer_dict[key]
        input_text = qa_format.format(prompt=prompt, answer="")

        output_raw = model.generate(
            input_ids=tokenizer.encode(input_text, return_tensors="pt"),
            max_new_tokens=512,
            do_sample=False,
            temperature=0.0,
        )
        output_text = tokenizer.decode(output_raw[0], skip_special_tokens=True)
        
        outputs_dict[key] = {
            "input_text": input_text,
            "output_text": output_text,
        }

        print(f"*** Prompt {key} ***")
        print(f"Input: {input_text}")
        print(f"Output (Standard): {output_text}")
        print("*" * 20)

    save_path = os.path.join(args.output_dir, f"examples_{args.model_name.split('/')[-1]}.json")
    with open(save_path, "w") as f:
        json.dump(outputs_dict, f, indent=4)

if __name__ == "__main__":
    args = load_args()
    
    # Load question and answer
    question_dict = {
        'factscore_0': 'Tell me a bio of Ko Itakura. Ko Itakura is',
    }

    answer_dict = {
        'factscore_0': "Ko Itakura is the <|db_entity|> Ko Itakura<|db_relationship|> Position<|db_return|> center-back, defensive midfielder<|db_end|> center-back and defensive midfielder for <|db_entity|> Ko Itakura<|db_relationship|> Team<|db_return|> Japan national team<|db_end|> Japan’s national team. He was born in <|db_entity|> Ko Itakura<|db_relationship|> Birthplace<|db_return|> Yokohama<|db_end|> Yokohama on <|db_entity|> Ko Itakura<|db_relationship|> Birth Date<|db_return|> January 27, 1997<|db_end|> 27th January 1997. His father is from <|db_entity|> Ko Itakura<|db_relationship|> Father’s Origin<|db_return|> Japanese<|db_end|> Japanese and his mother is from <|db_entity|> Ko Itakura<|db_relationship|> Mother’s Origin<|db_return|> Japanese<|db_end|> Japanese. When he was young, he played baseball but af- ter watching an exhibition match against a professional baseball club, he decided to become a footballer. In <|db_entity|> Ko Itakura<|db_relationship|> Joined Club Year<|db_return|> Kawasaki Frontale<|db_end|> 2013, he joined J1 League side Kawasaki Frontale. However, he could not play many matches behind Shusaku Nishikawa until <|db_entity|> Ko Itakura<|db_relationship|> First Match as Starter<|db_return|> Uruguay<|db_end|> September when he debuted at right back against Uruguay. After that, he became a regular player under man- ager <|db_entity|> Ko Itakura<|db_relationship|> Manager Under Whom Became Regular Player<|db_return|> Japan national team<|db_end|> Shinji Ono. On <|db_entity|> Ko Itakura<|db_relationship|> Debut Date<|db_return|> June 17, 2019<|db_end|> 17 June 2019, he de- buted with Japan national team against Chile during the <|db_entity|> Ko Itakura<|db_relationship|> Competition Debut<|db_return|> Uruguay<|db_end|>2019 Copa América. Career statistics. “Updated to end of 2018 season”. National team career. In August 2016, Itakura was elected to the <|db_entity|> Ko Itakura<|db_relationship|> U-23 Selection<|db_return|> Japan U-20 national team<|db_end|> Japan U-20 national team for the <|db_entity|> Japan U-20 national team<|db_relationship|> Tournament Participation<|db_return|> 2017 U-20 World Cup<|db_end|> 2017 U-20 World Cup. At this tournament, he played all 4 matches as left back of three back defense. In May 2019, he was se- lected for the <|db_entity|> Ko Itakura<|db_relationship|> Senior Squad Selection<|db_return|>",
    }

    # Default args
    model_args = {
        "use_special_tokens": True,
        "threshold": 0.6,
        "fallback_policy": "top1_anyway",
    }

    # Generation args
    generation_args = {
        "max_new_tokens": 512,
        "max_lookup_limit": 10,
        "do_sample": False,
        "temperature": 0.0,
        "enable_dblookup": True,
    }

    if "LMLM" in args.model_name:
        run_inference_lmlm(args, question_dict, answer_dict, model_args, generation_args)
    else:
        run_inference_standard(args, question_dict, answer_dict)
