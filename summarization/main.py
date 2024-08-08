import time
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
from pathlib import Path

import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_metric
from datasets import Dataset, DatasetDict

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers import AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.5.0.dev0")

logger = logging.getLogger(__name__)
nltk.data.path.append("/usr/project/xtmp/rz95/")

try:
    nltk.data.find("tokenizers/punkt")
    print("Found nltk")
except (LookupError, OSError):
    with FileLock(".lock") as lock:
        nltk.download("punkt", download_dir='/usr/project/xtmp/rz95/', quiet=True)
        print("Finished downloading punkt")
        
def _preprocess_dataset_for_summarization(dataset, sep_token):

    # For filtering out continued patents from our dataset
    decision_to_str = {
        'REJECTED': 0,
        'ACCEPTED': 1,
        'PENDING': 2,
        'CONT-REJECTED': 3,
        'CONT-ACCEPTED': 4,
        'CONT-PENDING': 5
    }

    # Indices of cont patents
    indices_of_cont_patents = {v for k, v in decision_to_str.items() if k.startswith('CONT-')}

    def map_decision_to_string(example):
        return {'decision': decision_to_str[example['decision']]}

    dataset = dataset.map(map_decision_to_string)

    def filter_cont_patents(e):
        return e['decision'] not in indices_of_cont_patents

    def format_example_for_summarization(e):
        if 'What is claimed is:' in e['claims'][:50]:
            e['claims'] = e['claims'].replace('What is claimed is:', '')

        # Format
        # NOTE: The tokenizer will add `bos` and `eos` tokens automatically, so we do not add them here
        text = 'TITLE {title} {sep} CLAIMS {claims}'.format(
            sep=sep_token, title=e['title'], claims=e['claims']
        )
        return {
            'claims_for_summarization': text,
            'abstract_for_summarization': e['abstract']
        }

    dataset = dataset.filter(filter_cont_patents)
    # Format examples
    dataset = dataset.map(format_example_for_summarization, batched=False)
    return dataset

def preprocess_dataset_for_summarization(dataset_dict, tokenizer):
    """ Loads dataset for language modeling. Note that the tokenizer is needed in order 
        to add [CLS] and [BOS] tokens to the data. """

    start_time = time.time()
    # Add new tokens to the tokenizer
    new_tokens = Path('ipc_labels.txt').read_text().splitlines(keepends=False)
    new_tokens += ['TITLE', 'CLAIMS']
    tokenizer.add_tokens(new_tokens)
    if not bool(tokenizer.sep_token):  # we need to add a <sep> token
        tokenizer.sep_token = '<sep>'

    # Create training and validation datasets
    print('>>> Training dataset')
    dataset_dict["train"] = _preprocess_dataset_for_summarization(
        dataset_dict["train"],
        sep_token=tokenizer.special_tokens_map['sep_token']
    )
    print('>>> Validation dataset')
    dataset_dict["validation"] = _preprocess_dataset_for_summarization(
        dataset_dict["validation"], 
        sep_token=tokenizer.special_tokens_map['sep_token']
    )
    
    print(f'****************** Finished loading dataset in {time.time() - start_time:.1f} seconds ******************')
    return dataset_dict, tokenizer

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    
    cache_dir = "/usr/project/xtmp/rz95/.cache/huggingface/" #<YOUR_OWN_PATH>

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=cache_dir,
        revision="main",
        use_auth_token=None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=cache_dir,
        use_fast=True,
        revision="main",
        use_auth_token=None,
    )

    print('Loading model from pretrained checkpoint')
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=cache_dir,
        revision="main",
        use_auth_token=None,
    )
    
    train_df = []
    path = "/usr/project/xtmp/rz95/InterpretableQA-LLMTools"
    for sub in range(2004, 2006):
        train_df.append(pd.read_csv("{}/data/external_corpus/hupd/hupd_{}.csv".format(path, sub)))
    train_df = pd.concat(train_df, ignore_index=True)
    val_df = pd.read_csv("{}/data/external_corpus/hupd/hupd_{}.csv".format(path, 2007))
    test_df = pd.read_csv("{}/data/external_corpus/hupd/hupd_{}.csv".format(path, 2008))
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    datasets = DatasetDict({"train": train_dataset, "validation": val_dataset, "test": test_dataset})

    # Preprocess dataset
    datasets, tokenizer = preprocess_dataset_for_summarization(datasets, tokenizer)
    
    # Resize model embeddings for tokenizer
    model.resize_token_embeddings(len(tokenizer))
    
    # Prediction
    # print("Using validation dataset for prediction")
    datasets["test"] = datasets["validation"]

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = "summarize: "

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = datasets["train"].column_names

    # Get the column names for input/target.
    dataset_columns = ("claims_for_summarization", "abstract_for_summarization")
    if data_args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = data_args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = data_args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Temporarily set max_target_length for training.
    max_target_length = 128 #
    padding = "max_length" #False

    def preprocess_function(examples):
        inputs = examples[text_column]
        targets = examples[summary_column]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=1024, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_dataset = datasets["train"]
    if "train" not in datasets:
        raise ValueError("--do_train requires a train dataset")
    if "test" not in datasets:
        raise ValueError("--do_predict requires a test dataset")
    test_dataset = datasets["test"]
    test_dataset = test_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
        load_from_cache_file=True,
    )

    # Data collator
    label_pad_token_id = -100 
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    metric = load_metric("rouge")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None, #eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    if last_checkpoint is not None:
        checkpoint = last_checkpoint
    elif os.path.isdir(model_args.model_name_or_path):
        checkpoint = model_args.model_name_or_path
    else:
        checkpoint = None
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics
    max_train_samples = (
        len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Evaluation
    results = {}
    logger.info("*** Test ***")
    test_results = trainer.predict(
        test_dataset,
        metric_key_prefix="test"
    )
    metrics = test_results.metrics
    metrics["test_samples"] = len(test_dataset)
    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)

    if trainer.is_world_process_zero():
        test_preds = tokenizer.batch_decode(
            test_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        test_preds = [pred.strip() for pred in test_preds]
        output_test_preds_file = os.path.join(training_args.output_dir, "test_generations.txt")
        with open(output_test_preds_file, "w") as writer:
            writer.write("\n".join(test_preds))

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
