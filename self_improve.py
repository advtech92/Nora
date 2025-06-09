# self_improve.py

import os
import subprocess
import tempfile
import shutil
import logging

import torch
from tokenizer import CharTokenizer
from model import NoraTransformerLM
from config import get_config

# ------------------------------------------------------
# 1) “Teacher”: Pose a code‐generation prompt to Nora
# ------------------------------------------------------
def propose_patch(model, tokenizer, device, prompt: str) -> str:
    """
    Ask Nora to generate a code snippet given `prompt`.
    e.g. prompt = "### FILE: knowledge_retriever.py\n# Add a new function clean_html(...) that..."
    Returns the raw text (possibly including the prompt).
    """
    raw = model.generate(
        tokenizer=tokenizer,
        device=device,
        prompt=prompt,
        max_length=512,
        temperature=0.7,
        top_k=50,
    )
    return raw


# ------------------------------------------------------
# 2) “Verifier” agent: sandbox + test
# ------------------------------------------------------
class CodeVerifier:
    """
    Given a proposed code patch (as text), this class:
      1. Writes it to a temporary file (or repo clone)
      2. Runs Python’s syntax check (compile) and unit tests
      3. Measures performance changes (e.g. run a small validation set through the model)
      4. Returns True/False + log messages
    """

    def __init__(self, repo_dir: str, test_command: str):
        """
        - repo_dir: path to your Nora project root
        - test_command: a shell command string to run unit tests, e.g. "pytest tests/"
        """
        self.repo_dir = repo_dir
        self.test_command = test_command

    def verify_patch(self, rel_path: str, patch_code: str) -> bool:
        """
        - rel_path: relative path inside repo where the patch should go, e.g. "knowledge_retriever.py"
        - patch_code: entire contents of that file (not a diff).
        Returns True if syntax + tests pass; False otherwise.
        """
        # 1) Copy repo => temp  dir
        tmpdir = tempfile.mkdtemp(prefix="nora_verify_")
        try:
            shutil.copytree(self.repo_dir, os.path.join(tmpdir, "repo"), dirs_exist_ok=True)
            target_file = os.path.join(tmpdir, "repo", rel_path)

            # 2) Write patch_code to target_file
            with open(target_file, "w", encoding="utf-8") as f:
                f.write(patch_code)

            # 3) Syntax check (try compiling)
            try:
                compile(patch_code, target_file, "exec")
            except SyntaxError as se:
                logging.error(f"Syntax error in patch: {se}")
                return False

            # 4) Run unit tests
            result = subprocess.run(
                self.test_command,
                shell=True,
                cwd=os.path.join(tmpdir, "repo"),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if result.returncode != 0:
                logging.error(f"Unit tests failed:\n{result.stdout}")
                return False

            # 5) (Optional) Performance check
            # You could load the updated model and measure perplexity on a tiny validation set here.
            # For now, we assume passing tests = “improvement.”

            return True

        finally:
            shutil.rmtree(tmpdir)

    def merge_patch(self, rel_path: str, patch_code: str) -> None:
        """
        Overwrite the real file in `repo_dir/rel_path` with patch_code,
        then git-add and git-commit (you can also automate a PR).
        """
        target_file = os.path.join(self.repo_dir, rel_path)
        with open(target_file, "w", encoding="utf-8") as f:
            f.write(patch_code)

        # Example: git add + commit
        subprocess.run(f"git add {rel_path}", shell=True, cwd=self.repo_dir)
        subprocess.run(
            f'git commit -m "Auto-update {rel_path} via Nora self-improve."', 
            shell=True,
            cwd=self.repo_dir,
        )


# ------------------------------------------------------
# 3) Main loop: ask → verify → merge (if good) → retrain
# ------------------------------------------------------
def self_improvement_cycle(repo_dir: str, device: str):
    """
    Example cycle:
      1) Nora proposes a new helper in knowledge_retriever.py
      2) Verifier checks syntax + tests
      3) If ok, merge and trigger incremental retraining
    """
    config = get_config()
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
    # Load latest checkpoint
    ckpts = []
    if os.path.isdir(config.checkpoint_dir):
        ckpts = [
            f
            for f in os.listdir(config.checkpoint_dir)
            if f.startswith("nora_step_") and f.endswith(".pt")
        ]
    if ckpts:
        latest = sorted(ckpts, key=lambda x: int(x.split("_")[-1].split(".")[0]))[-1]
        state = torch.load(os.path.join(config.checkpoint_dir, latest), map_location="cpu")
        model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()

    verifier = CodeVerifier(repo_dir=repo_dir, test_command="pytest tests/")

    # Example prompt: ask Nora to extend knowledge_retriever.py
    prompt = (
        "### FILE: knowledge_retriever.py\n"
        "# Add a function clean_html(html: str) -> str that strips tags and scripts.\n"
        "# Use BeautifulSoup if available. Return plain text.\n\n"
        "### START\n"
        "def clean_html(html: str) -> str:\n"
    )
    raw_patch = propose_patch(model, tokenizer, device, prompt)

    # Extract everything from “def clean_html” to end of function (simple heuristic)
    # In practice, you’d parse until the next “\n\n” or rely on indentation.
    patch_code = raw_patch  # for now, assume raw_patch is the full file contents

    # Verify
    if verifier.verify_patch("knowledge_retriever.py", patch_code):
        logging.info("Patch verified. Merging into live code.")
        verifier.merge_patch("knowledge_retriever.py", patch_code)
        # Optionally: trigger incremental retraining here (e.g. call train.py with --resume)
    else:
        logging.warning("Patch failed verification. Discarding.")
