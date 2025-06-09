# main.py

import os
import asyncio
import logging
import subprocess
import re
from urllib.parse import urlparse

import discord
import torch
from discord import Intents

from config import get_config
from tokenizer import CharTokenizer
from model import NoraTransformerLM, top_k_logits
from data_loader import get_dataloader
from utils import setup_logging, load_checkpoint, save_checkpoint
from knowledge_retriever import fetch_url, clean_html, save_text
import persona_manager  # <— PERSONA CHANGE

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Keep the SCRAPE regex as before
SCRAPE_PATTERN = re.compile(r"<<SCRAPE:(https?://[^\s>]+)>>", re.IGNORECASE)

# ---------------------------------------------------
# 1) Build or Reload Nora’s Model & Tokenizer
# ---------------------------------------------------
def build_nora(config, device):
    """
    - Loads tokenizer
    - Instantiates NoraTransformerLM
    - Loads latest checkpoint (if any)
    """
    tokenizer = CharTokenizer(vocab_path=config.vocab_path, data_dir=config.data_dir)

    model = NoraTransformerLM(
        vocab_size=tokenizer.vocab_size(),
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        max_seq_len=config.seq_length,
    )

    # Find latest checkpoint
    latest_ckpt = None
    if os.path.isdir(config.checkpoint_dir):
        ckpts = [
            f
            for f in os.listdir(config.checkpoint_dir)
            if f.startswith("nora_step_") and f.endswith(".pt")
        ]
        if ckpts:
            latest_ckpt = sorted(
                ckpts, key=lambda x: int(x.split("_")[-1].split(".")[0])
            )[-1]

    if latest_ckpt:
        ckpt_path = os.path.join(config.checkpoint_dir, latest_ckpt)
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state["model_state_dict"])
        logger.info(f"Loaded Nora from checkpoint {latest_ckpt}")
    else:
        logger.warning("No checkpoints found. Nora starts untrained.")

    model.to(device)
    model.eval()
    return model, tokenizer


# ---------------------------------------------------
# 2) Autoregressive Sampling Function
# ---------------------------------------------------
def generate_text(
    model: NoraTransformerLM,
    tokenizer: CharTokenizer,
    device: str,
    prompt: str,
    max_length: int = 128,
    temperature: float = 1.0,
    top_k: int = 50,
) -> str:
    """
    Wrapper around model.generate. Returns raw generated string.
    """
    return model.generate(
        tokenizer=tokenizer,
        device=device,
        prompt=prompt,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
    )


# ---------------------------------------------------
# 3) SCRAPE Directive Handler (same as before)
# ---------------------------------------------------
async def handle_scrape_directives(text: str, data_dir: str) -> bool:
    urls = set(m.group(1) for m in SCRAPE_PATTERN.finditer(text))
    if not urls:
        return False

    for url in urls:
        logger.info(f"Directive: Scraping {url}")
        html = fetch_url(url)
        if not html:
            logger.error(f"Failed to fetch {url}")
            continue
        plain = clean_html(html)
        title = ""
        m = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        if m:
            title = m.group(1).strip()[:50]
        else:
            title = urlparse(url).netloc
        save_text(plain, title)
    return True


# ---------------------------------------------------
# 4) Discord Client (“Nora” class) with Persona
# ---------------------------------------------------
class Nora(discord.Client):
    def __init__(self, model, tokenizer, config, device):
        intents = Intents.default()
        intents.messages = True
        intents.message_content = True
        super().__init__(intents=intents)

        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device

        # history per channel (last 5 user/nora pairs)
        self.history = {}

        # Load Nora’s persona from disk (it’s guaranteed to exist by startup)
        with open(persona_manager.PERSONA_PATH, "r", encoding="utf-8") as f:
            self.persona_text = f.read()

    async def on_ready(self):
        logger.info(f"Logged in as {self.user} (ID: {self.user.id})")
        logger.info("Nora is online and ready to be herself.")

        # Background task: reload model if a new checkpoint shows up
        self.loop.create_task(self._reload_model_periodically())

    async def _reload_model_periodically(self, interval: int = 600):
        """
        Every `interval` seconds, check for newer checkpoint & reload.
        """
        while True:
            await asyncio.sleep(interval)
            ckpts = [
                f
                for f in os.listdir(self.config.checkpoint_dir)
                if f.startswith("nora_step_") and f.endswith(".pt")
            ]
            if not ckpts:
                continue
            latest = sorted(
                ckpts, key=lambda x: int(x.split("_")[-1].split(".")[0])
            )[-1]
            ckpt_path = os.path.join(self.config.checkpoint_dir, latest)
            state = torch.load(ckpt_path, map_location="cpu")
            self.model.load_state_dict(state["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Reloaded Nora’s model from {latest}")

    async def on_message(self, message: discord.Message):
        # 1) Ignore self-messages
        if message.author.id == self.user.id:
            return

        content = message.content.strip()
        prompt = None

        # 2) If in DM, treat entire content as prompt
        if isinstance(message.channel, discord.DMChannel):
            prompt = content

            # Also allow “update persona” in DM to regenerate persona file
            if content.lower().startswith("update persona"):
                # Regenerate persona asynchronously
                new_persona = await persona_manager.maybe_update_persona(
                    self.model, self.tokenizer, self.device
                )
                self.persona_text = new_persona
                await message.channel.send(
                    "I have rewritten my persona. Thank you! ❤️"
                )
                return

        # 3) Otherwise (guild), require mention or “nora,” prefix
        else:
            if self.user.mention in content:
                prompt = content.replace(self.user.mention, "").strip()
            elif content.lower().startswith("nora,"):
                prompt = content[len("nora,"):].strip()
            else:
                return

        if not prompt:
            return  # e.g. user only said “Nora,” with no text after

        # 4) Show typing indicator if in a guild text channel
        if isinstance(message.channel, discord.TextChannel):
            await message.channel.trigger_typing()

        # 5) Build the full prompt: persona + history + user’s prompt
        chan_id = str(message.channel.id)
        history = self.history.get(chan_id, [])

        prompt_lines = []
        # 5.1) Insert Nora’s persona (so she “speaks as herself”)
        prompt_lines.append(self.persona_text)
        prompt_lines.append("")  # blank line

        # 5.2) Insert up to the last 4 exchanges
        for user_msg, nora_msg in history[-4:]:
            prompt_lines.append(f"User: {user_msg}")
            prompt_lines.append(f"Nora: {nora_msg}")
            prompt_lines.append("")

        # 5.3) Finally, insert the new user prompt
        prompt_lines.append(f"User: {prompt}")
        prompt_lines.append("Nora:")

        conversation_prompt = "\n".join(prompt_lines)

        # 6) Generate Nora’s reply (tighter sampling: temp=0.8, top_k=20)
        try:
            raw = await asyncio.to_thread(
                self.model.generate,
                self.tokenizer,
                self.device,
                conversation_prompt,
                self.config.seq_length,
                0.8,   # temperature
                20,    # top_k
            )
        except Exception:
            logger.exception("Error in Nora.generate()")
            await message.channel.send("😔 Sorry, I hit an error trying to think.")
            return

        # 7) Extract just Nora’s reply text
        if "Nora:" in raw:
            nora_reply = raw.split("Nora:")[-1].strip()
        else:
            nora_reply = raw[len(conversation_prompt) :].strip()

        # 8) Handle <<SCRAPE:…>> if present
        did_scrape = await handle_scrape_directives(raw, self.config.data_dir)
        if did_scrape:
            logger.info("Detected scrape directive. Triggering incremental retrain.")
            subprocess.Popen(
                ["python", "pretrain.py", "--resume"],  # note: pretrain.py was your old main.py
                cwd=os.getcwd(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            nora_reply += "\n\n*I found some new things and am updating myself…*"

        # 9) Save to history and prune to the last 5
        self.history.setdefault(chan_id, []).append((prompt, nora_reply))
        self.history[chan_id] = self.history[chan_id][-5 :]

        # 10) Send Nora’s reply
        await message.channel.send(nora_reply)


# ---------------------------------------------------
# 4) Entrypoint
# ---------------------------------------------------
if __name__ == "__main__":
    config = get_config()
    device = config.device

    # 4.1) Build/load model & tokenizer
    model, tokenizer = build_nora(config, device)

    # 4.2) Ensure a persona exists—if not, generate one now
    persona_manager.ensure_persona_file(model, tokenizer, device)

    # 4.3) After that, we can proceed to start the agent
    discord_token = os.getenv("DISCORD_TOKEN")
    if not discord_token:
        logger.error("Please set DISCORD_TOKEN in your environment.")
        exit(1)

    # Enable CuDNN autotune if on CUDA
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    bot = Nora(model=model, tokenizer=tokenizer, config=config, device=device)
    bot.run(discord_token)
