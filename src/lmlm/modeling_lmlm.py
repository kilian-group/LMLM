
import torch
from transformers import LlamaForCausalLM, AutoTokenizer
import re
from lmlm.training.utils.utils_filter import remove_unwanted_dblookups, set_use_special_dblookup_tokens, filter_incomplete_dblookups
from lmlm.database.database_manager import DatabaseManager, DatabaseLookupError
import logging  
from transformers import LlamaForCausalLM, AutoConfig
from transformers import LogitsProcessor
from lmlm.constants import DB_START_TOKEN, DB_SEP_TOKEN, DB_RETRIEVE_TOKEN, DB_END_TOKEN


logger = logging.getLogger(__name__)

class LlamaForLMLM(LlamaForCausalLM):
    def __init__(self, config, db_manager=None, use_special_tokens=True, threshold=None, fallback_policy="top1_anyway"):
        super().__init__(config)
        self.use_special_tokens = use_special_tokens
        self.fallback_policy = fallback_policy

        self.db_manager = db_manager
        if db_manager is not None:
            # Initialize the database manager
            self.db_manager.init_topk_retriever(default_threshold=threshold)
        
        self.logits_processor = None

    @classmethod
    def from_pretrained_with_db(cls, model_path, db_manager=None, use_special_tokens=True, threshold=None, fallback_policy=None, **kwargs):
        config = AutoConfig.from_pretrained(model_path, **kwargs)

        model = super().from_pretrained(model_path, config=config, **kwargs)

        if not isinstance(model, cls):
            model.__class__ = cls

        model.db_manager = db_manager
        model.use_special_tokens = use_special_tokens
        model.fallback_policy = fallback_policy

        if model.db_manager is not None:
            model.db_manager.init_topk_retriever(default_threshold=threshold)

        return model

    def normalize_db_format(self, text):
        text = re.sub(r'<\|db_entity\|>\s*', DB_START_TOKEN + ' ', text)
        text = re.sub(r'<\|db_relationship\|>\s*', DB_SEP_TOKEN + ' ', text)
        text = re.sub(r'<\|db_return\|>\s*', DB_RETRIEVE_TOKEN + ' ', text)
        text = re.sub(r'<\|db_end\|>\s*', DB_END_TOKEN + ' ', text)
        return text

    def token_len_without_dblookups(self, text, tokenizer):
        set_use_special_dblookup_tokens(use_special_dblookup_tokens=True)
        org_text = remove_unwanted_dblookups(text, triplets_to_keep=[])
        return len(tokenizer.encode(org_text))
    
    def post_process(self, output_text, tokenizer):
        """ Post-process the generated text. 
        Args:
            output_text (str): Generated text.
            tokenizer (PreTrainedTokenizer): Tokenizer for decoding.
        """         

        if tokenizer.bos_token:
            output_text = output_text.replace(tokenizer.bos_token, "")
        if tokenizer.eos_token:
            output_text = output_text.replace(tokenizer.eos_token, "")
        
        output_text = remove_unwanted_dblookups(output_text, triplets_to_keep=[])
        output_text = filter_incomplete_dblookups(output_text)

        return output_text
        
    def generate_with_lookup(self, prompt, tokenizer, enable_dblookup, enable_postprocess=True, **kwargs):
        """
        Generate text with optional database lookup.
        Args:
            prompt (str): Input prompt for generation.
            tokenizer (PreTrainedTokenizer): Tokenizer for encoding and decoding.
            max_new_tokens (int): Maximum number of new tokens to generate.
            temperature (float): Temperature for sampling.
            top_p (float): Top-p sampling parameter.
            repetition_penalty (float): Repetition penalty.
            enable_dblookup (bool): Whether to enable database lookup.
        """
        
        kwargs.pop("input_ids", None)
        kwargs.pop("attention_mask", None)
        kwargs.pop("pad_token_id", None)
        kwargs.pop("eos_token_id", None)
        kwargs.pop("return_dict_in_generate", None)
        kwargs.pop("output_scores", None)
        kwargs.pop("logits_processor", None)
        kwargs.pop("do_sample", None)


        max_new_tokens = kwargs.pop("max_new_tokens", 256)
        do_sample = kwargs.pop("do_sample", False)
        temperature = kwargs.pop("temperature", 0.0)
        top_p = kwargs.pop("top_p", 0.9)
        repetition_penalty = kwargs.pop("repetition_penalty", 1.2)
        max_lookup_limit = kwargs.pop("max_lookup_limit", 5)

        self.eval()
        device = self.device
        finished = False  

        generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=False,
        )

        if do_sample:
            generate_kwargs.update(
                dict(
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                )
            )
        else:
            generate_kwargs.update(
                dict(
                    do_sample=False,
                )
            )

        if not enable_dblookup:
            # TODO: disable lookup tokens
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_len = inputs["input_ids"].shape[1]

            outputs = self.generate(input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **generate_kwargs, 
                **kwargs)

            output_text = self.normalize_db_format(tokenizer.decode(outputs[0], skip_special_tokens=False))
            output_text = output_text.split(prompt)[-1]

            if enable_postprocess:
                output_text = self.post_process(output_text, tokenizer)
            return output_text

        # Set logits bias for special tokens
        self.set_logits_bias(tokenizer)

        input_text = prompt
        stop_token_ids = [
            tokenizer.convert_tokens_to_ids(DB_RETRIEVE_TOKEN),
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
        ]

        generate_kwargs["eos_token_id"] = stop_token_ids

        while not finished:
            #### Step 1: Prepare input
            inputs = tokenizer(input_text, return_tensors="pt").to(device)
            input_len = inputs["input_ids"].shape[1]

            #### Step 2: Generate until DB_RETRIEVE_TOKEN
            with torch.no_grad():
                outputs = self.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    logits_processor=self.logits_processor,
                    **generate_kwargs,
                    **kwargs
                )
            
            output_text = self._decode_with_special_tokens(outputs, tokenizer, input_len, input_text)

            input_text += output_text

            # Check if <|db_return|> is present
            if DB_RETRIEVE_TOKEN not in output_text:
                break

            #### Step 3: Perform DB lookup
            try:
                return_value = self.db_manager.retrieve_from_database(output_text)
            except DatabaseLookupError as e:
                logger.warning(f"Database lookup failed: {e}")
                # Handle DB lookup failure with fallback policy
                return_value, should_regenerate = self.handle_dblookup_failure(output_text)

            #### Step 4: Append retrieved value and db_end token
            input_text += return_value + DB_END_TOKEN

            # Optional: early stopping condition based on token length
            if self.token_len_without_dblookups(input_text, tokenizer) >= max_new_tokens:
                finished = True
                logger.warning(f"Prompt exceeded max new tokens")
            
            if len(input_text.split(DB_START_TOKEN)) >= max_lookup_limit:
                finished = True
                logger.warning(f"Prompt exceeded max lookup limit")
        
        output_text = input_text.split(prompt)[-1]

        if enable_postprocess:
            # Post-process the final output
            output_text = self.post_process(output_text, tokenizer)
        return output_text

    
    def set_logits_bias(self, tokenizer):

        if self.logits_processor is not None:
            return
        
        entity_token_id = tokenizer.convert_tokens_to_ids(DB_START_TOKEN)
        relationship_token_id = tokenizer.convert_tokens_to_ids(DB_SEP_TOKEN)
        return_token_id = tokenizer.convert_tokens_to_ids(DB_RETRIEVE_TOKEN)
        end_token_id = tokenizer.convert_tokens_to_ids(DB_END_TOKEN)

        # === Logit Bias ===
        bias = 2
        logit_bias = {
            entity_token_id: bias*2,
            relationship_token_id: bias,
            return_token_id: bias,
            end_token_id: bias
        }
        self.logits_processor = [LogitBiasProcessor(logit_bias)]
        return
    
    def _decode_with_special_tokens(self, outputs, tokenizer, input_len, input_text):
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        output_text = self.normalize_db_format(output_text)
        # logger.info(f"Output text: {output_text}")

        if input_text in output_text:
            output_text = output_text.split(input_text)[-1]
        else:
            output_text = tokenizer.decode(outputs[0][input_len:], clean_up_tokenization_spaces=True) 
            output_text = self.normalize_db_format(output_text)
            # logger.info(f"decode again: {output_text}")
        return output_text  
    
    def handle_dblookup_failure(self, output_text: str):
        """Handle DB lookup with fallback policy on failure."""
        if self.fallback_policy == "unknown":
            return "unknown", False
        
        elif self.fallback_policy == "top1_anyway":
            logger.info("Using top1 anyway as fallback policy.")
            try:
                return_value = self.db_manager.retrieve_from_database(output_text, threshold=0.0)
                return return_value, False
            except DatabaseLookupError:
                return "unknown", False
            
        elif self.fallback_policy == "regenerate_query":
            logger.info("Retrying query generation after dblookup failure.")
            raise NotImplementedError("Regenerate query not implemented yet.")
        else:
            logger.error(f"Unknown fallback policy: {self.fallback_policy}. Defaulting to 'unknown'.")
            return "unknown", False


class LogitBiasProcessor(LogitsProcessor):
    def __init__(self, bias_dict: dict):
        """
        bias_dict: {token_id: bias_value (positive = more likely)}
        """
        super().__init__()
        self.bias_dict = bias_dict

    def __call__(self, input_ids, scores):
        for token_id, bias in self.bias_dict.items():
            scores[:, token_id] += bias
        return scores


def load_model_and_tokenizer(model_path, database_path=None, model_args=None, device="cuda"):
    """
    Load the LMLM model and tokenizer from pretrained checkpoints.
    
    Args:
        model_path (str): Path to the pretrained model.
        database_path (str, optional): Path to the database JSON file.
        model_args (dict, optional): Additional arguments for model initialization.
        device (str): Device to load the model onto.
        
    Returns:
        model (PreTrainedModel): The loaded LMLM model.
        tokenizer (PreTrainedTokenizer): The corresponding tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    db_manager = None
    if database_path:
        db_manager = DatabaseManager()
        db_manager.load_database(database_path)

    model = LlamaForLMLM.from_pretrained_with_db(
        model_path,
        db_manager=db_manager,
        **(model_args or {}),
    )
    model.to(device)

    return model, tokenizer

