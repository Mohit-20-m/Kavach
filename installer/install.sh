#!/usr/bin/env bash
# ============================================================
# KAVACH — Supply Chain Security Installer
# Supports: macOS (zsh/bash), Linux (bash/zsh/fish)
# Works in: Terminal, VS Code, any shell
# ============================================================

set -e

KAVACH_VERSION="1.0.0"
KAVACH_DIR="$HOME/.kavach"
KAVACH_BIN="$KAVACH_DIR/bin"
KAVACH_MODELS="$KAVACH_DIR/models"
KAVACH_VENV="$KAVACH_DIR/venv"
GITHUB_REPO="Mohit-20-m/Kavach"
BOLD="\033[1m"
GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[0;33m"
CYAN="\033[0;36m"
RESET="\033[0m"

print_banner() {
  echo ""
  echo -e "${CYAN}${BOLD}  KAVACH - Supply Chain Security${RESET}"
  echo -e "  Agentic Behavioral Shield for Open Source Supply Chain Security"
  echo -e "  Version ${KAVACH_VERSION}"
  echo ""
}

log_info()    { echo -e "  ${GREEN}[OK]${RESET} $1"; }
log_warn()    { echo -e "  ${YELLOW}[!]${RESET} $1"; }
log_error()   { echo -e "  ${RED}[X]${RESET} $1"; }
log_step()    { echo -e "\n  ${BOLD}${CYAN}->${RESET} ${BOLD}$1${RESET}"; }
log_section() { echo -e "\n${BOLD}=== $1 ===${RESET}"; }

# ─── Check requirements ───────────────────────────────────────────────────────
check_requirements() {
  log_section "Checking Requirements"

  PYTHON=""
  for cmd in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$cmd" &>/dev/null; then
      if "$cmd" -c "import sys; exit(0 if sys.version_info >= (3,10) else 1)" 2>/dev/null; then
        PYTHON="$cmd"
        log_info "Python found: $PYTHON"
        break
      fi
    fi
  done

  if [ -z "$PYTHON" ]; then
    log_error "Python 3.10+ required."
    echo "  Install from: https://www.python.org/downloads/"
    echo "  Or: brew install python@3.11"
    exit 1
  fi

  if ! command -v git &>/dev/null; then
    log_error "git is required."
    echo "  Install from: https://git-scm.com/downloads"
    exit 1
  fi
  log_info "git found"

  if ! "$PYTHON" -m pip --version &>/dev/null; then
    log_error "pip not found. Run: $PYTHON -m ensurepip --upgrade"
    exit 1
  fi
  log_info "pip found"
}

# ─── Download KAVACH ──────────────────────────────────────────────────────────
download_kavach() {
  log_section "Downloading KAVACH"

  mkdir -p "$KAVACH_DIR" "$KAVACH_BIN" "$KAVACH_MODELS"

  if [ -d "$KAVACH_DIR/src/.git" ]; then
    log_info "Updating existing installation..."
    cd "$KAVACH_DIR/src" && git pull --quiet
    cd - > /dev/null
  else
    log_step "Cloning repository..."
    git clone --quiet --depth=1 \
      "https://github.com/${GITHUB_REPO}.git" \
      "$KAVACH_DIR/src" 2>/dev/null || {
        log_warn "git clone failed, trying direct download..."
        curl -sSL "https://github.com/${GITHUB_REPO}/archive/refs/heads/main.zip" \
          -o "$KAVACH_DIR/kavach.zip"
        unzip -q "$KAVACH_DIR/kavach.zip" -d "$KAVACH_DIR/"
        mv "$KAVACH_DIR/Kavach-main" "$KAVACH_DIR/src" 2>/dev/null || \
        mv "$KAVACH_DIR/kavach-main" "$KAVACH_DIR/src" 2>/dev/null || true
        rm "$KAVACH_DIR/kavach.zip"
      }
    log_info "Source downloaded"
  fi
}

# ─── Setup Python venv ────────────────────────────────────────────────────────
setup_venv() {
  log_section "Setting Up Python Environment"

  if [ ! -d "$KAVACH_VENV" ]; then
    log_step "Creating virtual environment..."
    "$PYTHON" -m venv "$KAVACH_VENV"
    log_info "Virtual environment created"
  else
    log_info "Virtual environment already exists"
  fi

  source "$KAVACH_VENV/bin/activate"

  log_step "Installing dependencies (2-3 minutes)..."
  pip install --quiet --upgrade pip
  pip install --quiet \
    numpy==1.26.3 \
    scikit-learn==1.4.0 \
    xgboost==2.0.3 \
    torch --index-url https://download.pytorch.org/whl/cpu \
    sentence-transformers \
    httpx \
    typer \
    rich \
    aiofiles \
    pydantic
  log_info "Dependencies installed"

  if [ -f "$KAVACH_DIR/src/cli/setup.py" ]; then
    pip install --quiet -e "$KAVACH_DIR/src/cli"
    log_info "KAVACH CLI installed"
  fi

  deactivate
}

# ─── Download models ──────────────────────────────────────────────────────────
download_models() {
  log_section "Downloading AI Models"

  if [ -f "$KAVACH_MODELS/code_archaeologist.pkl" ]; then
    log_info "Models already present"
    return
  fi

  log_step "Downloading models package..."
  curl -sSL "https://github.com/Mohit-20-m/Kavach/releases/download/v1.0.0/models.zip" \
    -o "$KAVACH_DIR/models.zip"

  log_step "Extracting models..."
  unzip -q "$KAVACH_DIR/models.zip" -d "$KAVACH_DIR/"
  rm "$KAVACH_DIR/models.zip"
  log_info "Models ready"
}

# ─── Create kavach-standalone wrapper ─────────────────────────────────────────
create_wrapper() {
  log_section "Creating KAVACH Executable"

  WRAPPER="$KAVACH_BIN/kavach-standalone"
  cat > "$WRAPPER" << WRAPPER_EOF
#!/usr/bin/env bash
source "$KAVACH_VENV/bin/activate"
export KAVACH_MODELS_DIR="$KAVACH_MODELS"
exec "$KAVACH_VENV/bin/kavach-standalone" "\$@"
WRAPPER_EOF
  chmod +x "$WRAPPER"
  log_info "kavach-standalone wrapper created"
}

# ─── Create npm and pip wrappers (works in ALL terminals) ─────────────────────
create_npm_pip_wrappers() {
  log_section "Creating npm and pip Wrappers (ALL terminals)"

  KAVACH_STANDALONE="$KAVACH_BIN/kavach-standalone"

  # npm wrapper
  cat > "$KAVACH_BIN/npm" << NPM_EOF
#!/usr/bin/env bash
# KAVACH npm wrapper — intercepts npm install
exec "$KAVACH_STANDALONE" npm "\$@"
NPM_EOF
  chmod +x "$KAVACH_BIN/npm"
  log_info "npm wrapper created"

  # pip wrapper
  cat > "$KAVACH_BIN/pip" << PIP_EOF
#!/usr/bin/env bash
# KAVACH pip wrapper — intercepts pip install
exec "$KAVACH_STANDALONE" pip "\$@"
PIP_EOF
  chmod +x "$KAVACH_BIN/pip"
  log_info "pip wrapper created"

  # pip3 wrapper
  cat > "$KAVACH_BIN/pip3" << PIP3_EOF
#!/usr/bin/env bash
# KAVACH pip3 wrapper — intercepts pip3 install
exec "$KAVACH_STANDALONE" pip "\$@"
PIP3_EOF
  chmod +x "$KAVACH_BIN/pip3"
  log_info "pip3 wrapper created"

  log_info "Wrappers will intercept npm/pip in ALL terminals once PATH is set"
}

# ─── Setup PATH in all shell configs ──────────────────────────────────────────
# The key insight: by putting KAVACH_BIN at the FRONT of PATH,
# the shell finds our npm/pip wrappers BEFORE the real npm/pip.
# This works in every terminal — bash, zsh, fish, VS Code, any shell.
setup_path() {
  log_section "Setting Up PATH (All Shells)"

  HOME_DIR="$HOME"

  PATH_BLOCK="
# --- KAVACH Supply Chain Security ---
export PATH=\"$KAVACH_BIN:\$PATH\"
export KAVACH_MODELS_DIR=\"$KAVACH_MODELS\"
# --- END KAVACH ---"

  # zsh
  if [ -f "$HOME_DIR/.zshrc" ]; then
    if ! grep -q "KAVACH Supply Chain Security" "$HOME_DIR/.zshrc" 2>/dev/null; then
      echo "$PATH_BLOCK" >> "$HOME_DIR/.zshrc"
      log_info "Added to ~/.zshrc"
    else
      log_info "Already in ~/.zshrc"
    fi
  else
    touch "$HOME_DIR/.zshrc"
    echo "$PATH_BLOCK" >> "$HOME_DIR/.zshrc"
    log_info "Created ~/.zshrc with KAVACH PATH"
  fi

  # bash
  for rcfile in ".bashrc" ".bash_profile"; do
    if [ -f "$HOME_DIR/$rcfile" ]; then
      if ! grep -q "KAVACH Supply Chain Security" "$HOME_DIR/$rcfile" 2>/dev/null; then
        echo "$PATH_BLOCK" >> "$HOME_DIR/$rcfile"
        log_info "Added to ~/$rcfile"
      else
        log_info "Already in ~/$rcfile"
      fi
    fi
  done

  # fish
  FISH_CONFIG="$HOME_DIR/.config/fish/config.fish"
  if [ -f "$FISH_CONFIG" ]; then
    if ! grep -q "KAVACH" "$FISH_CONFIG" 2>/dev/null; then
      cat >> "$FISH_CONFIG" << FISH_EOF

# --- KAVACH Supply Chain Security ---
fish_add_path "$KAVACH_BIN"
set -x KAVACH_MODELS_DIR "$KAVACH_MODELS"
# --- END KAVACH ---
FISH_EOF
      log_info "Added to fish shell"
    else
      log_info "Already in fish config"
    fi
  fi

  # VS Code uses the shell's PATH automatically
  # So once zshrc/bashrc is updated, VS Code terminal works too
  log_info "VS Code terminal will use updated PATH automatically"
}

# ─── Create disable/enable scripts ────────────────────────────────────────────
create_toggle_scripts() {
  log_section "Creating Enable/Disable Commands"

  # kavach-disable
  cat > "$KAVACH_BIN/kavach-disable" << 'DISABLE_EOF'
#!/usr/bin/env bash
# Remove KAVACH PATH from all shell configs
CONFIGS=("$HOME/.zshrc" "$HOME/.bashrc" "$HOME/.bash_profile")
for config in "${CONFIGS[@]}"; do
  if [ -f "$config" ] && grep -q "KAVACH Supply Chain Security" "$config"; then
    sed -i.bak '/# --- KAVACH Supply Chain Security ---/,/# --- END KAVACH ---/d' "$config"
    echo "[OK] Removed from $config"
  fi
done

FISH_CONFIG="$HOME/.config/fish/config.fish"
if [ -f "$FISH_CONFIG" ] && grep -q "KAVACH" "$FISH_CONFIG"; then
  sed -i.bak '/# --- KAVACH Supply Chain Security ---/,/# --- END KAVACH ---/d' "$FISH_CONFIG"
  echo "[OK] Removed from fish config"
fi

echo ""
echo "KAVACH disabled. Close and reopen all terminals to apply."
echo "To re-enable: kavach-enable"
DISABLE_EOF
  chmod +x "$KAVACH_BIN/kavach-disable"
  log_info "kavach-disable created"

  # kavach-enable
  WRAPPER_PATH="$KAVACH_BIN"
  MODELS_PATH="$KAVACH_MODELS"
  cat > "$KAVACH_BIN/kavach-enable" << ENABLE_EOF
#!/usr/bin/env bash
# Re-add KAVACH PATH to shell configs
PATH_BLOCK="
# --- KAVACH Supply Chain Security ---
export PATH=\"$WRAPPER_PATH:\\\$PATH\"
export KAVACH_MODELS_DIR=\"$MODELS_PATH\"
# --- END KAVACH ---"

CONFIGS=("\$HOME/.zshrc" "\$HOME/.bashrc" "\$HOME/.bash_profile")
for config in "\${CONFIGS[@]}"; do
  if [ -f "\$config" ] && ! grep -q "KAVACH Supply Chain Security" "\$config"; then
    echo "\$PATH_BLOCK" >> "\$config"
    echo "[OK] Added to \$config"
  fi
done

echo ""
echo "KAVACH enabled. Close and reopen all terminals to apply."
ENABLE_EOF
  chmod +x "$KAVACH_BIN/kavach-enable"
  log_info "kavach-enable created"
}

# ─── Verify ───────────────────────────────────────────────────────────────────
verify_installation() {
  log_section "Verifying Installation"

  for f in "kavach-standalone" "npm" "pip" "pip3" "kavach-disable" "kavach-enable"; do
    if [ -f "$KAVACH_BIN/$f" ] && [ -x "$KAVACH_BIN/$f" ]; then
      log_info "$f: found"
    else
      log_warn "$f: missing"
    fi
  done

  MODEL_COUNT=$(ls "$KAVACH_MODELS"/*.pkl "$KAVACH_MODELS"/*.pt "$KAVACH_MODELS"/*.json "$KAVACH_MODELS"/*.npy 2>/dev/null | wc -l | tr -d ' ')
  log_info "Model files: $MODEL_COUNT"

  if "$KAVACH_VENV/bin/python" -c "import xgboost, sklearn; print('ok')" &>/dev/null; then
    log_info "Python dependencies OK"
  else
    log_warn "Some Python dependencies may be missing"
  fi
}

# ─── Success ──────────────────────────────────────────────────────────────────
print_success() {
  echo ""
  echo -e "${GREEN}${BOLD}=================================================${RESET}"
  echo -e "${GREEN}${BOLD}   KAVACH Installation Complete!${RESET}"
  echo -e "${GREEN}${BOLD}=================================================${RESET}"
  echo ""
  echo -e "  ${YELLOW}${BOLD}IMPORTANT: Close this terminal and open a new one.${RESET}"
  echo -e "  Works in: Terminal, VS Code, iTerm2, any shell"
  echo ""
  echo -e "  Then test:"
  echo -e "    ${CYAN}npm install lodash${RESET}      -> Should show SAFE"
  echo -e "    ${CYAN}npm install yoshi-base${RESET}  -> Should show CRITICAL BLOCKED"
  echo ""
  echo -e "  To disable: ${CYAN}kavach-disable${RESET}"
  echo -e "  To enable:  ${CYAN}kavach-enable${RESET}"
  echo ""
}

# ─── Main ─────────────────────────────────────────────────────────────────────
main() {
  print_banner
  check_requirements
  download_kavach
  setup_venv
  download_models
  create_wrapper
  create_npm_pip_wrappers
  setup_path
  create_toggle_scripts
  verify_installation
  print_success
}

main "$@"
