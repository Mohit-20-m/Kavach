#!/usr/bin/env bash
# ============================================================
# KAVACH — Supply Chain Security Installer
# Supports: macOS (zsh/bash), Linux (bash/zsh)
# ============================================================

set -e

KAVACH_VERSION="1.0.0"
KAVACH_DIR="$HOME/.kavach"
KAVACH_BIN="$KAVACH_DIR/bin"
KAVACH_MODELS="$KAVACH_DIR/models"
KAVACH_VENV="$KAVACH_DIR/venv"
GITHUB_REPO="kavach-security/kavach"
BOLD="\033[1m"
GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[0;33m"
CYAN="\033[0;36m"
RESET="\033[0m"

print_banner() {
  echo ""
  echo -e "${CYAN}${BOLD}"
  echo "  ██╗  ██╗ █████╗ ██╗   ██╗ █████╗  ██████╗██╗  ██╗"
  echo "  ██║ ██╔╝██╔══██╗██║   ██║██╔══██╗██╔════╝██║  ██║"
  echo "  █████╔╝ ███████║██║   ██║███████║██║     ███████║"
  echo "  ██╔═██╗ ██╔══██║╚██╗ ██╔╝██╔══██║██║     ██╔══██║"
  echo "  ██║  ██╗██║  ██║ ╚████╔╝ ██║  ██║╚██████╗██║  ██║"
  echo "  ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝"
  echo -e "${RESET}"
  echo -e "  ${BOLD}Agentic Behavioral Shield for Open Source Supply Chain Security${RESET}"
  echo -e "  Version ${KAVACH_VERSION}"
  echo ""
}

log_info()    { echo -e "  ${GREEN}✓${RESET} $1"; }
log_warn()    { echo -e "  ${YELLOW}⚠${RESET} $1"; }
log_error()   { echo -e "  ${RED}✗${RESET} $1"; }
log_step()    { echo -e "\n  ${BOLD}${CYAN}→${RESET} ${BOLD}$1${RESET}"; }
log_section() { echo -e "\n${BOLD}$1${RESET}"; echo "  $(printf '─%.0s' {1..50})"; }

# ─── Check requirements ───────────────────────────────────────────────────────
check_requirements() {
  log_section "Checking Requirements"

  # Python 3.10+
  PYTHON=""
  for cmd in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$cmd" &>/dev/null; then
      VER=$("$cmd" -c "import sys; print(sys.version_info[:2])" 2>/dev/null)
      if "$cmd" -c "import sys; exit(0 if sys.version_info >= (3,10) else 1)" 2>/dev/null; then
        PYTHON="$cmd"
        log_info "Python found: $PYTHON ($VER)"
        break
      fi
    fi
  done

  if [ -z "$PYTHON" ]; then
    log_error "Python 3.10 or higher is required but not found."
    echo ""
    echo "  Install Python from: https://www.python.org/downloads/"
    echo "  Or with Homebrew:    brew install python@3.11"
    exit 1
  fi

  # git
  if ! command -v git &>/dev/null; then
    log_error "git is required but not found."
    echo "  Install git from: https://git-scm.com/downloads"
    exit 1
  fi
  log_info "git found: $(git --version)"

  # pip
  if ! "$PYTHON" -m pip --version &>/dev/null; then
    log_error "pip not found. Run: $PYTHON -m ensurepip --upgrade"
    exit 1
  fi
  log_info "pip found"
}

# ─── Download KAVACH source ───────────────────────────────────────────────────
download_kavach() {
  log_section "Downloading KAVACH"

  mkdir -p "$KAVACH_DIR" "$KAVACH_BIN" "$KAVACH_MODELS"

  if [ -d "$KAVACH_DIR/src/.git" ]; then
    log_info "Updating existing installation..."
    cd "$KAVACH_DIR/src" && git pull --quiet
  else
    log_step "Cloning repository..."
    git clone --quiet --depth=1 \
      "https://github.com/${GITHUB_REPO}.git" \
      "$KAVACH_DIR/src" 2>/dev/null || {
        # Fallback: download as zip if git clone fails
        log_warn "git clone failed, trying direct download..."
        curl -sSL "https://github.com/${GITHUB_REPO}/archive/refs/heads/main.zip" \
          -o "$KAVACH_DIR/kavach.zip"
        unzip -q "$KAVACH_DIR/kavach.zip" -d "$KAVACH_DIR/"
        mv "$KAVACH_DIR/kavach-main" "$KAVACH_DIR/src"
        rm "$KAVACH_DIR/kavach.zip"
      }
    log_info "Source downloaded"
  fi
}

# ─── Setup Python virtual environment ────────────────────────────────────────
setup_venv() {
  log_section "Setting Up Python Environment"

  if [ ! -d "$KAVACH_VENV" ]; then
    log_step "Creating virtual environment..."
    "$PYTHON" -m venv "$KAVACH_VENV"
    log_info "Virtual environment created"
  else
    log_info "Virtual environment already exists"
  fi

  # Activate venv
  source "$KAVACH_VENV/bin/activate"

  log_step "Installing dependencies (this may take 2-3 minutes)..."
  pip install --quiet --upgrade pip

  # Install core dependencies
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

  # Install kavach CLI
  if [ -f "$KAVACH_DIR/src/cli/setup.py" ]; then
    pip install --quiet -e "$KAVACH_DIR/src/cli"
    log_info "KAVACH CLI installed"
  fi

  deactivate
}

# ─── Download trained models ──────────────────────────────────────────────────
download_models() {
  log_section "Downloading AI Models"

  MODEL_BASE_URL="https://github.com/${GITHUB_REPO}/releases/download/v${KAVACH_VERSION}/models"

  MODELS=(
    "code_archaeologist.pkl"
    "maintainer_isolation_forest.pkl"
    "behavioral_isolation_forest.pkl"
    "lstm_autoencoder.pt"
    "meta_learner.pkl"
    "meta_learner_scaler.pkl"
    "score_thresholds.json"
    "agent_weights.json"
    "behavioral_metrics_col99.npy"
    "maintainer_profile_col99.npy"
  )

  ALL_PRESENT=true
  for model in "${MODELS[@]}"; do
    if [ ! -f "$KAVACH_MODELS/$model" ]; then
      ALL_PRESENT=false
      break
    fi
  done

  if [ "$ALL_PRESENT" = true ]; then
    log_info "All models already present"
    return
  fi

  log_step "Downloading trained models from GitHub Releases..."

  for model in "${MODELS[@]}"; do
    if [ ! -f "$KAVACH_MODELS/$model" ]; then
      curl -sSL "$MODEL_BASE_URL/$model" -o "$KAVACH_MODELS/$model" 2>/dev/null && \
        log_info "Downloaded: $model" || \
        log_warn "Could not download $model — will use defaults"
    else
      log_info "Already have: $model"
    fi
  done

  # Copy SBERT fine-tuned model if available
  if [ -d "$KAVACH_DIR/src/data/models/sbert_fine_tuned" ]; then
    cp -r "$KAVACH_DIR/src/data/models/sbert_fine_tuned" "$KAVACH_MODELS/"
    log_info "SBERT model copied"
  fi
}

# ─── Create wrapper script ────────────────────────────────────────────────────
create_wrapper() {
  log_section "Creating KAVACH Executable"

  WRAPPER="$KAVACH_BIN/kavach-standalone"

  cat > "$WRAPPER" << WRAPPER_EOF
#!/usr/bin/env bash
# KAVACH standalone wrapper
source "$KAVACH_VENV/bin/activate"
export KAVACH_MODELS_DIR="$KAVACH_MODELS"
exec "$KAVACH_VENV/bin/kavach-standalone" "\$@"
WRAPPER_EOF

  chmod +x "$WRAPPER"
  log_info "Wrapper created at $WRAPPER"
}

# ─── Setup shell intercepts ───────────────────────────────────────────────────
setup_shell() {
  log_section "Setting Up Shell Intercepts"

  WRAPPER="$KAVACH_BIN/kavach-standalone"

  KAVACH_SHELL_BLOCK=$(cat << SHELL_EOF

# ─── KAVACH Supply Chain Security ─────────────────────
function npm() { "$WRAPPER" npm "\$@"; }
function pip() { "$WRAPPER" pip "\$@"; }
function pip3() { "$WRAPPER" pip "\$@"; }
export PATH="$KAVACH_BIN:\$PATH"
# ──────────────────────────────────────────────────────
SHELL_EOF
)

  SHELL_CONFIGS=()
  HOME_DIR="$HOME"

  # Detect which shell configs exist
  [ -f "$HOME_DIR/.zshrc" ]        && SHELL_CONFIGS+=("$HOME_DIR/.zshrc")
  [ -f "$HOME_DIR/.bashrc" ]       && SHELL_CONFIGS+=("$HOME_DIR/.bashrc")
  [ -f "$HOME_DIR/.bash_profile" ] && SHELL_CONFIGS+=("$HOME_DIR/.bash_profile")

  # If none exist, create .zshrc on mac, .bashrc on linux
  if [ ${#SHELL_CONFIGS[@]} -eq 0 ]; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
      touch "$HOME_DIR/.zshrc"
      SHELL_CONFIGS+=("$HOME_DIR/.zshrc")
    else
      touch "$HOME_DIR/.bashrc"
      SHELL_CONFIGS+=("$HOME_DIR/.bashrc")
    fi
  fi

  INSTALLED=false
  for config in "${SHELL_CONFIGS[@]}"; do
    if grep -q "KAVACH Supply Chain Security" "$config" 2>/dev/null; then
      log_info "Already configured in $config"
    else
      echo "$KAVACH_SHELL_BLOCK" >> "$config"
      log_info "Added intercepts to $config"
      INSTALLED=true
    fi
  done

  # Also add to fish if present
  FISH_CONFIG="$HOME_DIR/.config/fish/config.fish"
  if [ -f "$FISH_CONFIG" ] && ! grep -q "KAVACH" "$FISH_CONFIG"; then
    cat >> "$FISH_CONFIG" << FISH_EOF

# KAVACH Supply Chain Security
function npm; "$WRAPPER" npm \$argv; end
function pip; "$WRAPPER" pip \$argv; end
function pip3; "$WRAPPER" pip \$argv; end
fish_add_path "$KAVACH_BIN"
FISH_EOF
    log_info "Added intercepts to fish shell"
  fi
}

# ─── Create disable/enable scripts ───────────────────────────────────────────
create_toggle_scripts() {
  log_section "Creating Enable/Disable Commands"

  # kavach-disable
  cat > "$KAVACH_BIN/kavach-disable" << 'DISABLE_EOF'
#!/usr/bin/env bash
# Remove KAVACH from all shell configs

CONFIGS=("$HOME/.zshrc" "$HOME/.bashrc" "$HOME/.bash_profile")
for config in "${CONFIGS[@]}"; do
  if [ -f "$config" ] && grep -q "KAVACH Supply Chain Security" "$config"; then
    # Remove the KAVACH block
    sed -i.bak '/# ─── KAVACH Supply Chain Security/,/# ─────────────────────────────────────────────────────$/d' "$config"
    echo "✓ Removed KAVACH from $config"
  fi
done

# Fish
FISH_CONFIG="$HOME/.config/fish/config.fish"
if [ -f "$FISH_CONFIG" ]; then
  sed -i.bak '/# KAVACH Supply Chain Security/,/fish_add_path.*kavach/d' "$FISH_CONFIG"
fi

echo ""
echo "KAVACH disabled. Restart your terminal or run: source ~/.zshrc"
echo "To re-enable: kavach-enable"
DISABLE_EOF

  chmod +x "$KAVACH_BIN/kavach-disable"
  log_info "kavach-disable command created"

  # kavach-enable (re-adds the block)
  WRAPPER="$KAVACH_BIN/kavach-standalone"
  cat > "$KAVACH_BIN/kavach-enable" << ENABLE_EOF
#!/usr/bin/env bash
# Re-add KAVACH to shell configs

KAVACH_BLOCK='
# ─── KAVACH Supply Chain Security ─────────────────────
function npm() { "$WRAPPER" npm "\$@"; }
function pip() { "$WRAPPER" pip "\$@"; }
function pip3() { "$WRAPPER" pip "\$@"; }
export PATH="$KAVACH_BIN:\$PATH"
# ──────────────────────────────────────────────────────'

CONFIGS=("\$HOME/.zshrc" "\$HOME/.bashrc" "\$HOME/.bash_profile")
for config in "\${CONFIGS[@]}"; do
  if [ -f "\$config" ] && ! grep -q "KAVACH Supply Chain Security" "\$config"; then
    echo "\$KAVACH_BLOCK" >> "\$config"
    echo "✓ Added KAVACH to \$config"
  fi
done

echo ""
echo "KAVACH enabled. Restart your terminal or run: source ~/.zshrc"
ENABLE_EOF

  chmod +x "$KAVACH_BIN/kavach-enable"
  log_info "kavach-enable command created"
}

# ─── Verify installation ──────────────────────────────────────────────────────
verify_installation() {
  log_section "Verifying Installation"

  if [ -f "$KAVACH_BIN/kavach-standalone" ]; then
    log_info "kavach-standalone binary: ✓"
  else
    log_error "kavach-standalone binary missing"
  fi

  MODEL_COUNT=$(ls "$KAVACH_MODELS"/*.pkl "$KAVACH_MODELS"/*.pt "$KAVACH_MODELS"/*.json "$KAVACH_MODELS"/*.npy 2>/dev/null | wc -l | tr -d ' ')
  log_info "Model files present: $MODEL_COUNT"

  if "$KAVACH_VENV/bin/python" -c "import xgboost, sklearn, torch; print('deps ok')" &>/dev/null; then
    log_info "Python dependencies: ✓"
  else
    log_warn "Some Python dependencies may be missing"
  fi
}

# ─── Print success message ────────────────────────────────────────────────────
print_success() {
  echo ""
  echo -e "${GREEN}${BOLD}╔══════════════════════════════════════════════════╗${RESET}"
  echo -e "${GREEN}${BOLD}║         KAVACH Installation Complete! 🛡️          ║${RESET}"
  echo -e "${GREEN}${BOLD}╚══════════════════════════════════════════════════╝${RESET}"
  echo ""
  echo -e "  ${BOLD}Restart your terminal, then try:${RESET}"
  echo ""
  echo -e "    ${CYAN}npm install lodash${RESET}      → Should show SAFE"
  echo -e "    ${CYAN}npm install yoshi-base${RESET}  → Should show CRITICAL and BLOCK"
  echo ""
  echo -e "  ${BOLD}Commands available:${RESET}"
  echo -e "    ${CYAN}kavach-disable${RESET}   Temporarily disable KAVACH"
  echo -e "    ${CYAN}kavach-enable${RESET}    Re-enable KAVACH"
  echo -e "    ${CYAN}kavach-standalone scan <pkg>${RESET}   Manual scan"
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
  setup_shell
  create_toggle_scripts
  verify_installation
  print_success
}

main "$@"
