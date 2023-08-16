import time
import click
from model import ExLlama, ExLlamaCache, ExLlamaConfig
from tokenizer import ExLlamaTokenizer
from generator import ExLlamaGenerator
import os, glob
from rich import print
from string import Template

OPEN_INSTRUCT_TEMPLATE = Template("""
Below is an instruction that describes a task. Write a response that appropriately completes the request

### Instruction: $prompt

### Response:
""")


@click.command()
@click.option(
    "--model-dir",
    default="/mnt/str/models/llama-13b-4bit-128g/",
    help="Directory containing model, tokenizer, generator",
)
@click.option(
    "--batch-size",
    default=4,
    help="Batch size",
)
@click.option(
    "--max-new-tokens",
    default=300,
    help="Maximum number of tokens to generate",
)
def batch(model_dir, batch_size, max_new_tokens):
    print(f"Using model directory: {model_dir} and batch size: {batch_size}")

    # Locate files we need within that directory

    tokenizer_path = os.path.join(model_dir, "tokenizer.model")
    model_config_path = os.path.join(model_dir, "config.json")
    st_pattern = os.path.join(model_dir, "*.safetensors")
    model_path = glob.glob(st_pattern)[0]

    # Batched prompts

    inputs = [
        "Show me the steps for determining the first 5 digits of pi",
        "Write a short python script to generate fibonacci numbers",
        "Translate this text into French: 'The quick brown fox jumped over the lazy dog'",
    ]

    prompts = [OPEN_INSTRUCT_TEMPLATE.substitute(prompt=inp) for inp in inputs]

    # Create config, model, tokenizer and generator

    t1 = time.time()

    config = ExLlamaConfig(model_config_path)  # create config from config.json
    config.model_path = model_path  # supply path to model weights file

    model = ExLlama(config)  # create ExLlama instance and load the weights
    tokenizer = ExLlamaTokenizer(
        tokenizer_path
    )  # create tokenizer from tokenizer model file

    cache = ExLlamaCache(model, batch_size=batch_size)  # create cache for inference
    generator = ExLlamaGenerator(model, tokenizer, cache)  # create generator

    # Configure generator

    generator.disallow_tokens([tokenizer.eos_token_id])

    generator.settings.token_repetition_penalty_max = 1.2
    generator.settings.temperature = 0.95
    generator.settings.top_p = 0.65
    generator.settings.top_k = 100
    generator.settings.typical = 0.5

    # Generate, batched

    for batch_idx in range(0, len(prompts), batch_size):
        output = generator.generate_simple(
            prompts[batch_idx : batch_idx + batch_size], max_new_tokens=max_new_tokens
        )

        if type(output) == str:
            output = [output]
        else:
            prompts_and_outputs = list(zip(prompts, output))
            for pair in prompts_and_outputs:
                print(f"\n------------------------\nPrompt: \n```\n{pair[0]}\n```\n\nOutput: \n```\n{pair[1]}\n```\n")

    t2 = time.time()
    print(f"Time taken: {t2-t1:,.2f} seconds")


if __name__ == "__main__":
    batch()
