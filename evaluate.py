import copy
import json
import os
import random
import re
import sys
import argparse
import time
from datetime import datetime

import torch

sys.path.append(os.path.join(os.getcwd(), "peft/src/"))
from peft import PeftModel
from tqdm import tqdm
from transformers import GenerationConfig, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
        load_8bit: bool = False,
        base_model: str = "",
        lora_weights: str = "tloen/alpaca-lora-7b",
        baseline: bool = False,
        share_gradio: bool = False,
):
    args = parse_args()
    set_random_seed(args.seed)

    def evaluate_batch(
            instructions,
            input=None,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            max_new_tokens=256,
            **kwargs,
    ):
        prompts = [generate_prompt(args, instruction, input) for instruction in instructions]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            do_sample=False,
            **kwargs,
        )
        with torch.no_grad():
            sequences = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                max_new_tokens=max_new_tokens,
            )
        generated_tokens = sequences[:, input_ids.shape[1]:]
        outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return [output.strip() for output in outputs]

    def evaluate_legacy(
            instruction,
            input=None,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            max_new_tokens=256,
            **kwargs,
    ):
        prompt = generate_prompt(args, instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                use_cache=False,
            )
        sequence = generation_output.sequences[0]
        output = tokenizer.decode(sequence)
        return output.split("### Response:")[1].strip()

    """
    # testing code for readme
    for instruction in [
        "Tell me about alpacas.",
        "Tell me about the president of Mexico in 2019.",
        "Tell me about the king of France in 2019.",
        "List all Canadian provinces in alphabetical order.",
        "Write a Python program that prints the first 10 Fibonacci numbers.",
        "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",  # noqa: E501
        "Tell me five words that rhyme with 'shock'.",
        "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
        "Count up from 1 to 500.",
    ]:
        print("Instruction:", instruction)
        print("Response:", evaluate(instruction))
        print()
    """
    eval_start_time = time.time()
    output_name = f'{args.model}-{args.adapter}-{args.dataset}'
    if args.output_tag:
        output_name = f'{output_name}-{args.output_tag}'
    save_file = f'experiment/{output_name}.json'
    summary_suffix = ""
    if args.output_tag:
        summary_suffix = f"_{sanitize_filename(args.output_tag)}"
    summary_jsonl = f'experiment/eval_summary{summary_suffix}.jsonl'
    summary_tsv = f'experiment/eval_summary{summary_suffix}.tsv'
    create_dir('experiment/')

    dataset = load_data(args)
    if args.shuffle_data:
        rng = random.Random(args.seed)
        rng.shuffle(dataset)
    dataset = dataset[args.sample_offset:]
    if args.max_samples is not None:
        dataset = dataset[:args.max_samples]
    tokenizer, model = load_model(args)
    total = len(dataset)
    miss = 0.001
    output_data = []
    start_idx = 0
    correct = 0

    if args.resume and os.path.exists(save_file):
        with open(save_file, 'r') as f:
            output_data = json.load(f)
        start_idx = len(output_data)
        correct = sum(1 for item in output_data if item.get('flag'))
        print(f"Resuming from {save_file}: {start_idx}/{total} samples already evaluated.")

    if start_idx >= total:
        print("All samples are already evaluated. Skipping generation.")
        total_to_run = 0
    else:
        total_to_run = total - start_idx
    pbar = tqdm(total=total_to_run, initial=0)
    effective_batch_size = 1 if args.legacy_eval_mode else args.batch_size
    for batch_start in range(start_idx, total, effective_batch_size):
        batch = dataset[batch_start: batch_start + effective_batch_size]
        instructions = [data.get('instruction') for data in batch]
        if args.legacy_eval_mode:
            outputs = [
                evaluate_legacy(
                    instruction,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                ) for instruction in instructions
            ]
        else:
            outputs = evaluate_batch(
                instructions,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
            )

        for sample_offset, (data, output_text) in enumerate(zip(batch, outputs)):
            idx = batch_start + sample_offset
            label = data.get('answer')
            flag = False
            if args.dataset.lower() in ['aqua']:
                predict = extract_answer_letter(args, output_text)
                if label == predict:
                    correct += 1
                    flag = True
            else:
                if isinstance(label, str):
                    label = float(label)
                predict = extract_answer_number(args, output_text)
                if abs(label - predict) <= miss:
                    correct += 1
                    flag = True
            new_data = copy.deepcopy(data)
            new_data['output_pred'] = output_text
            new_data['pred'] = predict
            new_data['flag'] = flag
            output_data.append(new_data)

            if args.verbose:
                print(' ')
                print('---------------')
                print(output_text)
                print('prediction:', predict)
                print('label:', label)
                print('---------------')

            current_done = idx + 1
            if current_done % args.log_every == 0 or current_done == total:
                print(f'\rtest:{current_done}/{total} | accuracy {correct}  {correct / current_done}')

        if (
            len(output_data) % args.save_every == 0
            or len(output_data) == total
        ):
            with open(save_file, 'w') as f:
                json.dump(output_data, f, indent=4)
        pbar.update(len(batch))
    pbar.close()
    eval_time_seconds = time.time() - eval_start_time
    accuracy = correct / total if total else 0.0
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "adapter": args.adapter,
        "dataset": args.dataset,
        "output_tag": args.output_tag if args.output_tag else "N/A",
        "base_model": args.base_model,
        "weights": args.lora_weights if args.lora_weights else "N/A",
        "seed": args.seed,
        "shuffle_data": args.shuffle_data,
        "sample_offset": args.sample_offset,
        "max_samples": total,
        "legacy_eval_mode": args.legacy_eval_mode,
        "multiple_choice_direct_answer": args.multiple_choice_direct_answer,
        "batch_size": effective_batch_size,
        "num_beams": args.num_beams,
        "max_new_tokens": args.max_new_tokens,
        "total_questions": total,
        "correct": correct,
        "accuracy": accuracy,
        "eval_time_seconds": round(eval_time_seconds, 4),
        "eval_time_hms": format_duration(eval_time_seconds),
    }
    append_summary(summary_jsonl, summary_tsv, summary)
    print('\n')
    print(
        "FINAL_RESULT | "
        f"time={summary['timestamp']} | "
        f"model={args.model} | "
        f"adapter={args.adapter} | "
        f"dataset={args.dataset} | "
        f"total={total} | "
        f"correct={correct} | "
        f"accuracy={accuracy:.6f} | "
        f"eval_time={format_duration(eval_time_seconds)}"
    )
    print('test finished')


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return


def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sanitize_filename(value):
    return re.sub(r'[^A-Za-z0-9._-]+', '_', value)


def append_summary(jsonl_path, tsv_path, summary):
    with open(jsonl_path, 'a') as f:
        f.write(json.dumps(summary) + '\n')

    header = [
        "timestamp",
        "model",
        "adapter",
        "dataset",
        "output_tag",
        "base_model",
        "weights",
        "seed",
        "shuffle_data",
        "sample_offset",
        "max_samples",
        "legacy_eval_mode",
        "multiple_choice_direct_answer",
        "batch_size",
        "num_beams",
        "max_new_tokens",
        "total_questions",
        "correct",
        "accuracy",
        "eval_time_seconds",
        "eval_time_hms",
    ]
    write_header = not os.path.exists(tsv_path)
    with open(tsv_path, 'a') as f:
        if write_header:
            f.write('\t'.join(header) + '\n')
        row = [str(summary[key]) for key in header]
        f.write('\t'.join(row) + '\n')


def format_duration(seconds):
    total_seconds = int(round(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def generate_prompt(args, instruction, input=None):
    if args.multiple_choice_direct_answer and args.dataset.lower() == 'aqua':
        if input:
            return f"""Below is a multiple-choice math question with additional context. Choose the single correct option.

                ### Instruction:
                {instruction}

                ### Input:
                {input}

                ### Response:
                Reply with only the answer letter (A, B, C, D, or E). Do not include any explanation, reasoning, or extra words.
                """  # noqa: E501
        else:
            return f"""Below is a multiple-choice math question. Choose the single correct option.

                ### Instruction:
                {instruction}

                ### Response:
                Reply with only the answer letter (A, B, C, D, or E). Do not include any explanation, reasoning, or extra words.
                """  # noqa: E501
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Input:
                {input}

                ### Response:
                """  # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

                ### Instruction:
                {instruction}

                ### Response:
                """  # noqa: E501


def load_data(args) -> list:
    """
    read data from dataset file
    Args:
        args:

    Returns:

    """
    file_path = f'dataset/{args.dataset}/test.json'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"can not find dataset file : {file_path}")
    json_data = json.load(open(file_path, 'r'))
    return json_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['AddSub', 'MultiArith', 'SingleEq', 'gsm8k', 'AQuA', 'SVAMP'],
                        required=True)
    parser.add_argument('--model', choices=['LLaMA-7B', 'TinyLlama', 'BLOOM-7B', 'GPT-j-6B'], default="")
    parser.add_argument('--adapter', choices=['Base', 'LoRA', 'AdapterP', 'AdapterH', 'Parallel', 'Prefix'],
                        required=True)
    parser.add_argument('--base_model', required=True)
    parser.add_argument('--lora_weights')
    parser.add_argument('--baseline', action='store_true', default=False)
    parser.add_argument('--load_8bit', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--save_every', type=int, default=20)
    parser.add_argument('--log_every', type=int, default=20)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--shuffle_data', action='store_true', default=False)
    parser.add_argument('--sample_offset', type=int, default=0)
    parser.add_argument('--max_samples', type=int)
    parser.add_argument('--output_tag', default="")
    parser.add_argument('--legacy_eval_mode', action='store_true', default=False)
    parser.add_argument('--multiple_choice_direct_answer', action='store_true', default=False)

    args = parser.parse_args()
    if not args.baseline and not args.lora_weights:
        parser.error("--lora_weights is required unless --baseline is set")
    if args.batch_size < 1:
        parser.error("--batch_size must be >= 1")
    if args.num_beams < 1:
        parser.error("--num_beams must be >= 1")
    if args.max_new_tokens < 1:
        parser.error("--max_new_tokens must be >= 1")
    if args.save_every < 1:
        parser.error("--save_every must be >= 1")
    if args.log_every < 1:
        parser.error("--log_every must be >= 1")
    if args.sample_offset < 0:
        parser.error("--sample_offset must be >= 0")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max_samples must be >= 1")
    return args


def load_model(args) -> tuple:
    """
    load tuned model
    Args:
        args:

    Returns:
        tuple(tokenizer, model)
    """
    base_model = args.base_model
    if not base_model:
        raise ValueError(f'can not find base model name by the value: {args.model}')
    lora_weights = args.lora_weights
    if not args.baseline and not lora_weights:
        raise ValueError(f'can not find lora weight, the value is: {lora_weights}')

    load_8bit = args.load_8bit
    if args.model in ['LLaMA-7B', 'TinyLlama']:
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        ) # fix zwq
        if not args.baseline:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
                device_map={"":0}
            )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        if not args.baseline:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        if not args.baseline:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
            )

        if not load_8bit:
            model.half()  # seems to fix bugs for some users.

    # Keep generation config consistent across all devices.
    model.config.pad_token_id = tokenizer.pad_token_id
    if model.config.bos_token_id is None:
        model.config.bos_token_id = tokenizer.bos_token_id
    if model.config.eos_token_id is None:
        model.config.eos_token_id = tokenizer.eos_token_id

    model.eval()

    return tokenizer, model


def load_instruction(args) -> str:
    instruction = ''
    if not instruction:
        raise ValueError('instruct not initialized')
    return instruction


def extract_answer_number(args, sentence: str) -> float:
    dataset = args.dataset.lower()
    if dataset in ["multiarith", "addsub", "singleeq", "gsm8k", "svamp"]:
        sentence = sentence.replace(',', '')
        pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
        if not pred:
            return float('inf')
        pred_answer = float(pred[-1])
    else:
        raise NotImplementedError(' not support dataset: {}'.format(dataset))
    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float('inf')
    return pred_answer


def extract_answer_letter(args, sentence: str) -> str:
    sentence_ = sentence.strip()

    if args.multiple_choice_direct_answer:
        direct_patterns = [
            r'^\s*\(?\s*([A-E])\s*\)?[\s\.\,\!\?\:\;]*$',
            r'(?i)^\s*answer\s*[:\-]?\s*\(?\s*([A-E])\s*\)?[\s\.\,\!\?\:\;]*$',
            r'(?i)^\s*option\s*[:\-]?\s*\(?\s*([A-E])\s*\)?[\s\.\,\!\?\:\;]*$',
        ]
        for pattern in direct_patterns:
            match = re.search(pattern, sentence_)
            if match:
                return match.group(1).upper()

        # In direct-answer mode, prefer the earliest option token if the model
        # still emits a short phrase instead of a bare letter.
        fallback_matches = re.findall(r'\(?\b([A-E])\b\)?', sentence_, flags=re.IGNORECASE)
        if fallback_matches:
            return fallback_matches[0].upper()
        return ''

    # Prefer explicit answer phrases before falling back to standalone options.
    explicit_patterns = [
        r'(?i)\bfinal\s+answer\s+is\s*\(?\s*([A-E])\s*\)?',
        r'(?i)\banswer\s+is\s*\(?\s*([A-E])\s*\)?',
        r'(?i)\banswer\s*:\s*\(?\s*([A-E])\s*\)?',
        r'(?i)\boption(?:\s+is)?\s*\(?\s*([A-E])\s*\)?',
    ]

    matches = []
    for pattern in explicit_patterns:
        matches.extend(re.findall(pattern, sentence_))

    if matches:
        return matches[-1].upper()

    fallback_matches = re.findall(r'\(?\b([A-E])\b\)?', sentence_, flags=re.IGNORECASE)
    if fallback_matches:
        return fallback_matches[-1].upper()

    return ''


if __name__ == "__main__":
    main()
