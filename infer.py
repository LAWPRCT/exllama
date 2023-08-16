import time
import click
from model import ExLlama, ExLlamaCache, ExLlamaConfig
from tokenizer import ExLlamaTokenizer
from generator import ExLlamaGenerator
import os, glob
from rich import print
from string import Template

OPEN_INSTRUCT_TEMPLATE = Template("""
### System:
Read the user's text and return valid JSON with the following keys:
- "corrected": the text with line-to-line hyphens removed and errors replaced
- "paraphrase": a paraphrase of the text, written in third person
- "people": an array of strings of people mentioned in the text
- "places": an array of strings of places mentioned in the text
- "organizations": an array of strings organizations mentioned in the text
- "key_date": the most important date mentioned in the text as an ISO date string
- "key_date_precision": datetime|date|month|year
- "topics": an array of topics mentioned in the text

### User:
$prompt

### Assistant:
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
        "On July 22, 2016, Rebekah Mercer — Robert’s powerful daughter — emailed Steve Bannon from her Stanford alumni account. She wanted the Breitbart executive chairman, whom she introduced as “one of the greatest living defenders of Liberty,” to meet an app developer she knew. Apple had rejected the man’s game (Capitol HillAwry, in which players delete emails a la Hillary Clinton) from the App Store, and the younger Mercer wondered “if we could put an article up detailing his 1st amendment political persecution.”",
        "twas a Friday night in early November and Fox News host Tucker Carlson was preaching to the camera about leftists sowing “racial divisions” throughout America. Reacting to an article in The Washington Post about a principal at a Maryland high school investigating an incident in which someone posted fliers proclaiming, “It’s Okay to Be White.” Carlson claimed to see evidence of an anti-white agenda at play in the Post report. The segment was typical Fox News fodder, but with an exception: Carlson was 19 forward a meme promoted by white supremacists, and he was doing so exactly as they had intended him to do",
        "The purpose of creating this network was to obscure Valeant’s role in the aggressive pharmacy-led promotion and sale of its price-gouged drugs, and to create the appearance that ostensibly independent pharmacies across the country were promoting and selling Valeant products of their own volition and based on the merits of Valeant’s products. Moreover, as described herein, the network also allowed Valeant to implement deceptive practices designed to trick payors into reimbursing Valeant for drugs",
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
