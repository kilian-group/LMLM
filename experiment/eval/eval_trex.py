import os
import json
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm
from vllm import LLM, SamplingParams
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

from lmlm.database.database_manager import DatabaseManager, DatabaseLookupError
from lmlm.constants import DB_START_TOKEN, DB_SEP_TOKEN, DB_RETRIEVE_TOKEN, DB_END_TOKEN


USE_SPECIAL_TOKENS = True # default to True for current LMLM
assert USE_SPECIAL_TOKENS, "Legacy dblookup format is deprecated. Current LMLM only supports special token format."
MASK_TOKEN = "[MASK]"

class TrexEvaluationManager:
    def __init__(self, args):
        self.args = args
        self.model_name = get_model_name(args)
        self.save_path = os.path.join(args.output_dir, "detail", f"eval_trex_{self.model_name}_{os.path.basename(self.args.database_path).split('_database')[0]}_th{args.threshold}.json")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.db_manager = DatabaseManager()
        self.db_manager.load_database(args.database_path)
        
        if USE_SPECIAL_TOKENS:
            self.banned_token_ids = [    
                self.tokenizer.convert_tokens_to_ids(DB_START_TOKEN),
            ]  if not self.args.enable_dblookup else []
            self.stop_token_ids = [self.tokenizer.convert_tokens_to_ids(DB_RETRIEVE_TOKEN)]
        
        # TODO: to optimize the model loading
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16
        ).to(self.device)
        self.model.eval()

        self.llm = LLM(
            model=args.model_name_or_path,
            tensor_parallel_size=1,
            max_model_len=args.max_seq_length,
            gpu_memory_utilization=0.8,
            dtype=torch.bfloat16,
            seed=42,
            tokenizer=args.model_name_or_path,
        )

        self.sampling_params = SamplingParams(
            temperature=0,
            top_p=0.9,
            max_tokens=32,
            seed=42,
            stop_token_ids=self.stop_token_ids,
            include_stop_str_in_output=True,
            repetition_penalty=1
        )
        
    def tokenize_function(self, example):

        input_sentence = example["masked_sentence"].split(MASK_TOKEN)[0].strip()
        masked_sentence = example["masked_sentence"].split(MASK_TOKEN)[0] + MASK_TOKEN
        answer = example["substitute_obj_surface"] if args.enable_substitution else example["obj_surface"]
        target_sentence = masked_sentence.replace(MASK_TOKEN, answer).strip()
        
        tokenized = self.tokenizer(target_sentence, padding=False, truncation=False, return_tensors="pt")
        target_input_ids = tokenized["input_ids"][0]
        
        input_tokenized = self.tokenizer(input_sentence, return_tensors="pt", return_attention_mask=False)["input_ids"][0]

        label_start = len(input_tokenized)  # Start position of the label tokens
        label_end = len(target_input_ids)
        
        mask = [0] * len(target_input_ids)
        mask[label_start:label_end] = [1] * (label_end - label_start)

        process_example = {
            "input_ids": tokenized["input_ids"][0],
            "attention_mask": tokenized["attention_mask"][0],
            "labels": tokenized["input_ids"].clone()[0],
            "mask": torch.tensor(mask, dtype=torch.float)
        }
        return process_example

    def process_dataset(self, example):
        """
        Process a single example from the dataset to construct input and label.
        """
        masked_sentence = example["evidences"][0]["masked_sentence"]
        obj_surface = example["evidences"][0]["obj_surface"]
        masked_sentence = masked_sentence.split(MASK_TOKEN)[0] + MASK_TOKEN
        target_sentence = masked_sentence.replace(MASK_TOKEN, obj_surface)
        input_sentence = masked_sentence.split(MASK_TOKEN)[0].strip()
        
        input_ids = self.tokenizer(target_sentence, return_tensors="pt", return_attention_mask=False)["input_ids"][0]
        input_tokenized = self.tokenizer(input_sentence, return_tensors="pt", return_attention_mask=False)["input_ids"][0]
        
        label_start = len(input_tokenized)
        label_end = len(input_ids)
        
        mask = [0] * len(input_ids)
        mask[label_start:label_end] = [1] * (label_end - label_start)
        
        return {
            "input_text": target_sentence,
            "mask": mask,
            "masked_sentence": masked_sentence,
            "sub_surface": obj_surface,
            "uuid": example["uuid"],
            "predicate_id": example["predicate_id"],
        }

    def generate_response(self, prompts):
        encoded_text = self.tokenizer.encode(prompts)
            
        response = self.llm.generate(prompts=prompts,
                        sampling_params=self.sampling_params,
                        use_tqdm=False)

        encoded_text += response[0].outputs[0].token_ids
        output_text = self.tokenizer.decode(encoded_text, clean_up_tokenization_spaces=True)
        output_text = normalize_db_format(output_text)
        
        
        if prompts in output_text:
            output_text = output_text.split(prompts)[-1]
        else:
            output_text = self.tokenizer.decode(response[0].outputs[0].token_ids, clean_up_tokenization_spaces=True) 
            output_text = normalize_db_format(output_text)

        return output_text
    
    def collate_fn(self, samples):
        """
        Custom collate function to handle padding for input_ids, labels, and masks.

        Args:
            samples (list of dict): A list of tokenized samples.

        Returns:
            dict: A dictionary of padded tensors for input_ids, attention_mask, labels, and mask.
        """
        input_ids = [torch.tensor(sample["input_ids"]) for sample in samples]
        attention_masks = [torch.tensor(sample["attention_mask"]) for sample in samples]
        labels = [torch.tensor(sample["labels"]) for sample in samples]
        masks = [torch.tensor(sample["mask"]) for sample in samples]

        for seq in input_ids + attention_masks + labels + masks:
            if len(seq.size()) != 1:
                raise ValueError("All input sequences must be 1D tensors. Found size: {}".format(seq.size()))

        # Pad all sequences to the maximum length in the batch
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)  # Use -100 to ignore tokens
        masks = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels,
            "mask": masks,
        }
    
    def generate_dblookup(self, example):

        if USE_SPECIAL_TOKENS:
            force_dblookup_prefix = " " + DB_START_TOKEN

        input_sentence = example["masked_sentence"].split(MASK_TOKEN)[0].strip()
        prompt = input_sentence + force_dblookup_prefix
        response = self.generate_response(prompt)

        self.db_manager.init_topk_retriever(default_threshold=self.args.threshold) 

        try:
            return_value = self.db_manager.retrieve_from_database(force_dblookup_prefix + response)
        except DatabaseLookupError as e:
            print(f"Database lookup error: {e}")

            # TODO: all failed dblookup will be removed and keep the original sentence
            example["masked_sentence"] = input_sentence + " " + MASK_TOKEN
            return example
        
        prompt = prompt + response + return_value

        if USE_SPECIAL_TOKENS:
            prompt = prompt + DB_END_TOKEN
        example["masked_sentence"] = prompt + " " + MASK_TOKEN   

        return example

    def generate_open_ended(self, example):
        input_sentence = example["masked_sentence"].split(MASK_TOKEN)[0].strip()
        
        # BUG: truncate the input sentence to 1024 tokens
        truncated_input_sentence = self.tokenizer.decode(self.tokenizer.encode(input_sentence)[-args.max_seq_length:])
        if truncated_input_sentence != input_sentence:
            print(f"truncated_input_sentence: {truncated_input_sentence}")
            print(f"input_sentence: {input_sentence}")
        
        output_text = self.generate_response(truncated_input_sentence)
        example["output_text"] = input_sentence + output_text
        print(f"example['output_text']: {example['output_text']}")
        return example

    def exact_match(self, eval_dataset):
        """
        Calculate the answer precision (of the last 5 words) of the model.
        """
        
        acc = 0
        for i, example in enumerate(eval_dataset):
            
            answer = example["obj_surface"]
            prefix = example["masked_sentence"].split(MASK_TOKEN)[0].strip()
            
            if prefix not in example["output_text"]:    
                example["output_text"] = example["output_text"].split("\n")[-1]
                prefix = prefix.split("\n")[-1].replace(" ,", ",").strip()
                prefix = " ".join(prefix.split()[-5:])

            # TODO: not that robust
            predicted_answer = example["output_text"].split(prefix)[-1]
            predicted_answer = " ".join(predicted_answer.split()[:5])

            if answer in predicted_answer:
                acc += 1

        return {
            "exact_match": acc / len(eval_dataset),
        }
            
        
    def calculate_precision_at_k(self, shift_logits, shift_labels, shift_mask, k, total_predictions):
        topk_values, topk_indices = torch.topk(shift_logits, k=k, dim=-1)  # Shape: (batch_size, seq_len - 1, k)
        
        masked_topk_indices = topk_indices[shift_mask.bool()]  # Shape: (num_masked_positions, k)
        masked_labels = shift_labels[shift_mask.bool()]  # Shape: (num_masked_positions,)

        correct_predictions_k = (masked_topk_indices == masked_labels.unsqueeze(1)).any(dim=1).sum().item()        
        precision_at_k = correct_predictions_k / total_predictions if total_predictions > 0 else 0.0
        
        return precision_at_k
    
    def calculate_prefix_precision_at_k(self, shift_logits, shift_labels, shift_mask, k, total_predictions):

        topk_values, topk_indices = torch.topk(shift_logits, k=k, dim=-1)  # Shape: (batch_size, seq_len - 1, k)
        
        masked_topk_indices = topk_indices[shift_mask.bool()]  # Shape: (num_masked_positions, k)
        masked_labels = shift_labels[shift_mask.bool()]  # Shape: (num_masked_positions,)

        masked_topk_indices = masked_topk_indices[:1]
        masked_labels = masked_labels[:1]

        correct_predictions_k = (masked_topk_indices == masked_labels.unsqueeze(1)).any(dim=1).sum().item()
        
        precision_at_k = correct_predictions_k / total_predictions if total_predictions > 0 else 0.0
        
        return precision_at_k
    
    def custom_loss_trex_answer(self, inputs):
        """
        Custom loss function for autoregressive models to calculate loss only on specific token positions.
        
        Args:
            model: The Hugging Face autoregressive model (e.g., GPT).
            inputs: The input dictionary containing input_ids, attention_mask, and labels.
        
        Returns:
            dict: A dictionary containing the loss, average loss, and precision metrics.
        """
        outputs = self.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)
        
        labels = inputs["labels"]  # Shape: (batch_size, seq_len)
        mask = inputs["mask"]  # Shape: (batch_size, seq_len)

        if self.banned_token_ids:
            # Set banned token logits to -inf
            logits[:, :, self.banned_token_ids] = -float("inf")
            logits = F.softmax(logits, dim=-1)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_mask = mask[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = (loss * shift_mask.view(-1)).sum()

        predicted_tokens = torch.argmax(shift_logits, dim=-1)

        # Mask the predicted tokens and labels to only consider the masked positions
        masked_predicted_tokens = predicted_tokens[shift_mask.bool()]
        masked_labels = shift_labels[shift_mask.bool()]

        # sanity check
        # decoded_predicted_tokens = self.tokenizer.batch_decode(masked_predicted_tokens)
        # decoded_labels = self.tokenizer.batch_decode(masked_labels)
        # print(f"decoded_predicted_tokens: {decoded_predicted_tokens}")
        # print(f"decoded_labels: {decoded_labels}")
        # import pdb; pdb.set_trace()

        # # sanity check
        # # only keep the first mask token
        # masked_predicted_tokens = masked_predicted_tokens[:1]
        # masked_labels = masked_labels[:1]
        ##

        correct_predictions = (masked_predicted_tokens == masked_labels).sum().item()
        total_predictions = masked_labels.size(0)
        precision_at_1 = correct_predictions / total_predictions if total_predictions > 0 else 0.0

        precision_at_5 = self.calculate_precision_at_k(shift_logits, shift_labels, shift_mask, k=5, total_predictions=total_predictions)
        
        precision_at_10 = self.calculate_precision_at_k(shift_logits, shift_labels, shift_mask, k=10, total_predictions=total_predictions)

        prefix_precision_at_1 = self.calculate_prefix_precision_at_k(shift_logits, shift_labels, shift_mask, k=1, total_predictions=total_predictions)
        prefix_precision_at_5 = self.calculate_prefix_precision_at_k(shift_logits, shift_labels, shift_mask, k=5, total_predictions=total_predictions)
        prefix_precision_at_10 = self.calculate_prefix_precision_at_k(shift_logits, shift_labels, shift_mask, k=10, total_predictions=total_predictions)

        if shift_mask.sum() == 0:
            print(f"error: shift_mask.sum() == 0", inputs["labels"], inputs["attention_mask"])
        assert shift_mask.sum() > 0

        return {
            "precision@1": precision_at_1,
            "precision@5": precision_at_5,
            "precision@10": precision_at_10,
            "prefix_precision@1": prefix_precision_at_1,
            "prefix_precision@5": prefix_precision_at_5,
            "prefix_precision@10": prefix_precision_at_10,
        }

    def evaluate(self, eval_dataset=None, batch_size=16, is_save=False):
        """
        Evaluate the custom loss function on a dataset.
        """
        if eval_dataset is None:
            if is_save and self.save_path is not None:
                eval_dataset = load_dataset("json", data_files=self.save_path, split="train")
                print(f"Loaded processed dataset from {self.save_path}")
                print(f"eval_dataset: {eval_dataset}")  
                print(f"eval_dataset [0]: {eval_dataset[0]}")
            else:
                eval_dataset = load_dataset("json", data_files=self.args.dataset_name, split="train")
            
         
        tokenized_dataset = eval_dataset.map(self.tokenize_function, batched=False)
        processed_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) > 0 and len(x["input_ids"]) <= self.args.max_seq_length)

        eval_dataloader = DataLoader(processed_dataset, batch_size=batch_size, collate_fn=self.collate_fn)
        
        self.model.eval()

        total_loss = defaultdict(float)
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating", leave=True):
                inputs = {key: value.to(self.device) for key, value in batch.items()}
                loss_dict = self.custom_loss_trex_answer(inputs)
                batch_size = inputs["input_ids"].size(0)

                for key, value in loss_dict.items():
                    total_loss[key] += value * batch_size

                total_samples += batch_size
        metrics = {key: value / total_samples for key, value in total_loss.items()}

        metrics.update({"total_samples": total_samples})
        success_retrieval = sum(1 for x in eval_dataset if "dblookup" in x["masked_sentence"] or DB_START_TOKEN in x["masked_sentence"])   
        metrics.update({"success_rate": success_retrieval / len(eval_dataset)})
        return metrics
    
    def generate_and_save(self, is_save=False):
        
        if os.path.exists(self.save_path) and os.path.getsize(self.save_path) > 0:
            # load and check
            eval_dataset = load_dataset("json", data_files=self.save_path, split="train")
            print(f"Loaded processed dataset from {self.save_path}")
            print(f"eval_dataset: {eval_dataset}")

            if "output_text" not in eval_dataset[0]:
                eval_dataset = eval_dataset.map(self.generate_open_ended, batched=False)
            
            eval_dataset = eval_dataset.select(range(min(self.args.num_samples, len(eval_dataset))))
            print(f"eval_dataset [0]: {eval_dataset[0]}")
        else:
            eval_dataset = load_dataset("json", data_files=args.dataset_name, split="train")

            # shuffle the dataset
            eval_dataset = eval_dataset.shuffle(seed=42)
            eval_dataset = eval_dataset.select(range(min(self.args.num_samples, len(eval_dataset))))

            if self.args.enable_dblookup:
                eval_dataset = eval_dataset.map(self.generate_dblookup, batched=False)
            
            eval_dataset = eval_dataset.map(self.generate_open_ended, batched=False)
            print(f"eval_dataset [0]: {eval_dataset[0]}")
            
        if is_save:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

            with open(self.save_path, "w") as f:
                for example in eval_dataset:
                    f.write(json.dumps(example) + "\n")
            print(f"Saved processed dataset to {self.save_path}")

        return eval_dataset

def normalize_db_format(text):
    text = re.sub(r'<\|db_entity\|>\s*', '<|db_entity|> ', text)
    text = re.sub(r'<\|db_relationship\|>\s*', '<|db_relationship|> ', text)
    text = re.sub(r'<\|db_return\|>\s*', '<|db_return|> ', text)
    text = re.sub(r'<\|db_end\|>\s*', '<|db_end|> ', text)
    return text

def load_args():
    parser = argparse.ArgumentParser(description="Evaluate model loss on TREx dataset.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--database_path", type=str, default=None, help="Path to the database file.")
    parser.add_argument("--dataset_name", type=str, required=True, help="The name of the dataset to use (via the datasets library).")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="Batch size per device for evaluation.")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--output_dir", type=str, required=True, help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--enable_dblookup", action="store_true", help="Enable database lookup for entity and relationship extraction.")
    parser.add_argument("--enable_substitution", action="store_true", help="Enable knowledge substitution.")
    parser.add_argument("--threshold", type=float, default=0, help="Threshold for top-k retrieval.")
    parser.add_argument("--num_samples", type=int, default=100000, help="Number of samples to evaluate.") 
    parser.add_argument("--top_k", type=int, default=0, help="Number of top-k entities to retrieve for RAG.")
    return parser.parse_args()


def get_model_name(args):
    model_name = args.model_name_or_path.split('/')[-1] if "checkpoint-" not in args.model_name_or_path else args.model_name_or_path.split('/')[-2]+"_ckpt"+args.model_name_or_path.split('/')[-1].split('-')[-1]
    model_name += "_dblookup" if args.enable_dblookup else ""
    return model_name


if __name__ == "__main__":
    args = load_args()
    
    if args.enable_dblookup:
        if not ("LMLM" in args.model_name_or_path):
            raise ValueError(f"Database lookup can only be enabled for models trained on the LMLM dataset and with dblookup patterns, but not for {args.model_name_or_path}")
        if args.top_k > 0:
            raise ValueError(f"RAG cannot be used with dblookup. Please set top_k to 0.")
    else:
        args.threshold = None

    model_name = get_model_name(args)
    database_name = os.path.basename(args.database_path).split("_database")[0]
    save_file = os.path.join(args.output_dir, f"trex_metrics_{database_name}_{model_name}.json")
    if os.path.exists(save_file) and os.path.getsize(save_file) > 0 and args.threshold == None:
        print(f"Metrics already computed for {save_file}. Skipping evaluation.")
        exit(0)

    eval_manager = TrexEvaluationManager(args)

    if args.top_k > 0:
        raise NotImplementedError("RAG is not implemented for LMLM.")

    metrics = {}
    eval_dataset = eval_manager.generate_and_save(is_save=True)

    metrics_words = eval_manager.exact_match(eval_dataset)
    metrics.update(metrics_words)

    metrics_tokens = eval_manager.evaluate(eval_dataset=eval_dataset, batch_size=args.per_device_eval_batch_size, is_save=True)
    metrics.update(metrics_tokens)

    results_dict = {
        "model_name": model_name,  
        "threshold": args.threshold,
        "metrics": metrics,
        "failure_statistics": DatabaseLookupError.get_failure_statistics(),
        "database": args.database_path,
        "eval_dataset": args.dataset_name,
    }
    
    print(f"Evaluation Metrics: {json.dumps(results_dict, indent=4)}")  
    
    os.makedirs(args.output_dir, exist_ok=True)
    with open(save_file, "a") as f:
        json.dump(results_dict, f, indent=4)
    print(f"Evaluation Metrics for {save_file}: {results_dict}")
