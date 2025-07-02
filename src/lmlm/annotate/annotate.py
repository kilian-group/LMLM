import argparse
import json
import os
import warnings
from tqdm import tqdm

from lmlm.annotate.annotators import ChatGPTAnnotator, LlamaAnnotator, LlamaLoraAnnotator
from lmlm.annotate.data_manager import Example, DataManager
from lmlm.annotate.utils import get_save_name
from lmlm.annotate.dataloader import prepare_data
from lmlm.constants import DATA_DIR


if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--annotator", type=str, required=True, choices=["chatgpt", "llama", "llama-lora-ft"], help="Annotator to use")
    args.add_argument("--model-id", type=str, required=True, help="Model ID")
    args.add_argument("--prompt-id", type=str, required=True, help="Prompt ID")
    args.add_argument("--manager", type=str, required=True, help="Name of the data manager")
    args.add_argument("--config-file", type=str, default=None, help="Configuration file for the annotator")
    args.add_argument("--dataset", type=str, default="squad", help="Dataset to use")
    args.add_argument("--subset", type=str, default=None, help="Subset of the dataset to use")
    args.add_argument("--format", type=str, default="json", help="Format of the dataset")
    args.add_argument("--save-every", type=int, default=None, help="Save every")
    args.add_argument("--postprocess", action="store_true", help="Postprocess the annotated data")
    args.add_argument("--seed", type=int, default=0)
    args.add_argument("--save-dir", type=str, default="./output/annotation", help="Directory to save the annotated data")
    args = args.parse_args()

        
    save_name = get_save_name(args)
    print(f"Save name: {save_name}")    

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    if os.path.exists(os.path.join(args.save_dir, f"{save_name}.{args.format}")):
        print("Loading from ", os.path.join(args.save_dir, f"{save_name}.{args.format}"))
        manager = DataManager.load(f"{save_name}", format=args.format, save_dir=args.save_dir)
    else:
        manager = DataManager()

    # prepare data
    if args.subset is not None:
        with open(os.path.join(DATA_DIR, f"{args.subset}.json"), 'r') as f:
            subset = json.load(f)
    else:
        subset = None
    
    dataset_split = 'train' if 'train' in args.manager else 'validation'
    texts, text_ids = prepare_data(dataset_name=args.dataset, dataset_split=dataset_split, subset_ids=subset)

    # check if the data manager has already annotated the data
    already_annotated_indices = []
    for i, _text_ids in enumerate(text_ids):
        for example in manager.get_examples_by_source(args.dataset):
            if example.original_dataset_ids == _text_ids:
                warnings.warn(f"skipping example already annotated by {example.model_id} with prompt {example.prompt_id}")
                already_annotated_indices.append(i)
                break

    texts = [text for i, text in enumerate(texts) if i not in already_annotated_indices]
    text_ids = [text_id for i, text_id in enumerate(text_ids) if i not in already_annotated_indices]
    print(f"Annotating {len(texts)} examples, already annotated {len(already_annotated_indices)} examples")

    if len(texts) == 0:
        # exit if all examples are already annotated    
        print("All examples are already annotated")
        exit()
    
    # annotator
    if args.annotator == "chatgpt":
        annotator = ChatGPTAnnotator(args.model_id, args.prompt_id, args.config_file)
    elif args.annotator == "llama":
        if args.config_file is None:
            args.config_file = "llama/default"
        annotator = LlamaAnnotator(args.model_id, args.prompt_id, args.config_file)
    elif args.annotator == "llama-lora-ft":
        if args.config_file is None:
            args.config_file = "llama/lora-ft"
        annotator = LlamaLoraAnnotator(args.model_id, args.prompt_id, args.config_file)
    else:
        raise ValueError(f"Annotator {args.annotator} not supported")

    # annotate data
    k = args.save_every if args.save_every is not None else len(texts)
    batched_texts = [texts[i:i+k] for i in range(0, len(texts), k)]
    batched_ids = [text_ids[i:i+k] for i in range(0, len(text_ids), k)]
    print(f"Annotating {len(texts)} examples in {len(batched_texts)} batches")

    for batch, text_ids_lst in tqdm(zip(batched_texts, batched_ids), total=len(batched_texts), desc="Annotating"):
        annotated_batch = annotator.annotate(batch)
        if args.postprocess:
            annotated_batch = annotator.postprocess(annotated_batch)

        # add annotated data to data manager
        for annotated_text, ids, text in zip(annotated_batch, text_ids_lst, batch):
            manager.add_example(
                Example(
                    annotated_text=annotated_text,
                    text=text,
                    model_id=args.model_id,
                    prompt_id=args.prompt_id,
                    original_dataset=args.dataset,
                    original_dataset_ids=ids,
                )
            )
        if len(annotated_batch) == 0:
            raise ValueError("No data annotated")

        # save data manager
        manager.save(f"{save_name}", format=args.format, save_dir=args.save_dir)
        print(f"Saved data manager to {args.save_dir}/{save_name}.{args.format}")
