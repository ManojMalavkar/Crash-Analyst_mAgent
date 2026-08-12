"""SafetyAgent - Interactive Setup Wizard.

Run this after cloning the repo and activating your virtual environment.
Guides you through the full setup: install deps, build knowledge base, configure env.

Usage:
    python -m venv .venv
    .venv\Scripts\activate        # Windows
    source .venv/bin/activate      # Linux/Mac
    pip install -r requirements.txt
    python setup.py
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


# =============================================================================
# Display Helpers
# =============================================================================

def banner():
    print()
    print("=" * 60)
    print("  SafetyAgent - Knowledge Base Setup Wizard")
    print("=" * 60)
    print()


def step(num, total, msg):
    print(f"\n[{num}/{total}] {msg}")
    print("-" * 60)


def success(msg):
    print(f"  [OK] {msg}")


def error(msg):
    print(f"  [ERROR] {msg}")


def warn(msg):
    print(f"  [!] {msg}")


def ask(prompt, default=None):
    """Ask user for input with optional default."""
    if default:
        user_input = input(f"  {prompt} [{default}]: ").strip()
        return user_input if user_input else default
    return input(f"  {prompt}: ").strip()


# =============================================================================
# Setup Steps
# =============================================================================

def check_python_version():
    """Verify Python 3.10+."""
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 10):
        error(f"Python 3.10+ required. You have {major}.{minor}")
        sys.exit(1)
    success(f"Python {major}.{minor} detected")


def check_venv():
    """Check if running inside a virtual environment."""
    if sys.prefix == sys.base_prefix:
        warn("Not running in a virtual environment.")
        print()
        print("  Recommended:")
        print("    python -m venv .venv")
        print("    .venv\\Scripts\\activate        # Windows")
        print("    source .venv/bin/activate      # Linux/Mac")
        print("    python setup.py                # Re-run this script")
        print()
        proceed = ask("Continue without venv? (y/n)", "n")
        if proceed.lower() != "y":
            print("\n  Setup cancelled. Create venv and try again.")
            sys.exit(0)
    else:
        success(f"Virtual environment active: {sys.prefix}")


def install_dependencies():
    """Install requirements.txt."""
    req_file = Path("requirements.txt")
    if not req_file.exists():
        error("requirements.txt not found. Are you in the project root?")
        sys.exit(1)

    print("  Installing packages (this takes ~1 minute)...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "--quiet"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        error("pip install failed:")
        print(result.stderr)
        sys.exit(1)
    success("All dependencies installed")


def setup_env_file():
    """Create .env from .env.example if not exists."""
    if Path(".env").exists():
        success(".env file already exists")
        return

    if Path(".env.example").exists():
        shutil.copy(".env.example", ".env")
        success("Created .env from .env.example")
        warn("Edit .env later to add your LLM_API_KEY")
    else:
        warn("No .env.example found. Create .env manually when needed.")


def get_docs_path():
    """Interactively ask user for their ANSA documentation path."""
    print("  The knowledge base needs your ANSA/META documentation files.")
    print("  Point to the folder containing .py, .html, .json, or .md files.")
    print()
    print("  Common paths:")
    print("    Windows: C:\\BETA_CAE_Systems\\ANSA_v2025.2.2\\python\\doc")
    print("    Linux:   /opt/BETA_CAE_Systems/ansa_v2025.2.2/python/doc")
    print()

    while True:
        docs_path = ask("Enter documentation path")

        if not docs_path:
            error("Path cannot be empty.")
            continue

        docs_dir = Path(docs_path)

        if not docs_dir.exists():
            error(f"Directory not found: {docs_dir}")
            retry = ask("Try again? (y/n)", "y")
            if retry.lower() != "y":
                return None
            continue

        # Count parseable files
        extensions = {".py", ".html", ".json", ".jsonl", ".md"}
        file_count = sum(
            1 for f in docs_dir.rglob("*") if f.suffix.lower() in extensions
        )

        if file_count == 0:
            error(f"No parseable files found in: {docs_dir}")
            error("Expected .py, .html, .json, .jsonl, or .md files.")
            retry = ask("Try a different path? (y/n)", "y")
            if retry.lower() != "y":
                return None
            continue

        success(f"Found {file_count} files in: {docs_dir}")
        confirm = ask("Proceed with this path? (y/n)", "y")
        if confirm.lower() == "y":
            return str(docs_dir)


def build_knowledge_base(source_dir):
    """Run build_vector_db.py to ingest and embed documents."""
    print("  Building vector database (this takes 2-5 minutes)...")
    print()

    result = subprocess.run(
        [
            sys.executable,
            "01_ANSA_ApiAgent/bin/build_vector_db.py",
            "--source", source_dir,
            "--rebuild",
        ],
        capture_output=False,  # Show live output
    )

    if result.returncode != 0:
        error("Knowledge base build failed.")
        print("  Check the error above and try again.")
        return False

    success("Knowledge base built successfully")
    return True


def verify_installation():
    """Quick import checks."""
    checks = [
        ("shared module", "from shared import settings"),
        ("chromadb", "import chromadb"),
        ("sentence-transformers", "from sentence_transformers import SentenceTransformer"),
        ("networkx", "import networkx"),
    ]

    all_ok = True
    for name, import_stmt in checks:
        try:
            exec(import_stmt)
            success(f"{name}: OK")
        except ImportError:
            error(f"{name}: FAILED")
            all_ok = False

    return all_ok


# =============================================================================
# Main Wizard
# =============================================================================

def main():
    banner()

    total_steps = 5

    # Step 1: Python version check
    step(1, total_steps, "Checking environment")
    check_python_version()
    check_venv()

    # Step 2: Install dependencies
    step(2, total_steps, "Installing dependencies")
    install_dependencies()

    # Step 3: Environment config
    step(3, total_steps, "Setting up configuration")
    setup_env_file()

    # Step 4: Build knowledge base
    step(4, total_steps, "Building knowledge base")
    docs_path = get_docs_path()

    if docs_path:
        kb_ok = build_knowledge_base(docs_path)
    else:
        warn("Skipped knowledge base build.")
        warn("Run later: python 01_ANSA_ApiAgent/bin/build_vector_db.py --source <path>")
        kb_ok = False

    # Step 5: Verify
    step(5, total_steps, "Verifying installation")
    verify_ok = verify_installation()

    # Summary
    print()
    print("=" * 60)
    if kb_ok and verify_ok:
        print("  SETUP COMPLETE!")
    else:
        print("  SETUP PARTIALLY COMPLETE")
    print("=" * 60)
    print()
    print("  Next steps:")
    print("    1. Add your LLM API key to .env")
    if kb_ok:
        print("    2. Start the agent:")
        print("       python 01_ANSA_ApiAgent/app_gradio.py")
    else:
        print("    2. Build knowledge base:")
        print("       python 01_ANSA_ApiAgent/bin/build_vector_db.py --source <docs_path>")
        print("    3. Start the agent:")
        print("       python 01_ANSA_ApiAgent/app_gradio.py")
    print()


if __name__ == "__main__":
    main()
