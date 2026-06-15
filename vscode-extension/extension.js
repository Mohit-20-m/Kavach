/**
 * KAVACH VS Code Extension v3
 * Cross-platform: macOS, Linux, Windows
 * Works with kavach-standalone installed via the KAVACH installer
 */

const vscode = require("vscode");
const { exec, spawn } = require("child_process");
const path = require("path");
const os = require("os");
const fs = require("fs");

let statusBarItem;
let outputChannel;
let kavachPath = null;
let kavachEnabled = true;

// ─── Platform detection ───────────────────────────────────────────────────────
const IS_WINDOWS = process.platform === "win32";
const IS_MAC = process.platform === "darwin";
const HOME = os.homedir();
const KAVACH_DIR = path.join(HOME, ".kavach");
const KAVACH_BIN = path.join(KAVACH_DIR, "bin");

// ─── Find kavach-standalone ───────────────────────────────────────────────────
function findKavach() {
  const candidates = IS_WINDOWS
    ? [
        path.join(KAVACH_BIN, "kavach-standalone.bat"),
        path.join(KAVACH_BIN, "kavach-standalone.cmd"),
        path.join(KAVACH_DIR, "venv", "Scripts", "kavach-standalone.exe"),
        "kavach-standalone",
      ]
    : [
        path.join(KAVACH_BIN, "kavach-standalone"),
        path.join(KAVACH_DIR, "venv", "bin", "kavach-standalone"),
        path.join(HOME, "Downloads", "kavach", "venv", "bin", "kavach-standalone"),
        "/usr/local/bin/kavach-standalone",
        "/opt/homebrew/bin/kavach-standalone",
        "kavach-standalone",
      ];

  // Check file existence first (faster)
  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) {
      return Promise.resolve(candidate);
    }
  }

  // Fall back to which/where
  return new Promise((resolve) => {
    const cmd = IS_WINDOWS ? "where kavach-standalone" : "which kavach-standalone";
    exec(cmd, (err, stdout) => {
      if (!err && stdout.trim()) {
        resolve(stdout.trim().split("\n")[0]);
      } else {
        resolve(null);
      }
    });
  });
}

// ─── Run a kavach scan ────────────────────────────────────────────────────────
function runScan(packageName, ecosystem) {
  return new Promise((resolve, reject) => {
    if (!kavachPath) {
      reject(new Error("KAVACH CLI not found. Run the KAVACH installer first."));
      return;
    }

    outputChannel.appendLine(`\n${"─".repeat(60)}`);
    outputChannel.appendLine(`🛡️  Scanning: ${packageName} (${ecosystem})`);
    outputChannel.appendLine(`${"─".repeat(60)}`);

    const args = ["scan", packageName, "--ecosystem", ecosystem];
    const env = {
      ...process.env,
      KAVACH_MODELS_DIR: path.join(KAVACH_DIR, "models"),
      PATH: `${KAVACH_BIN}${path.delimiter}${process.env.PATH}`,
    };

    const child = spawn(kavachPath, args, { env, shell: IS_WINDOWS });
    let stdout = "";

    child.stdout.on("data", (d) => { stdout += d; outputChannel.append(d.toString()); });
    child.stderr.on("data", (d) => { outputChannel.append(d.toString()); });

    child.on("close", () => {
      const upper = stdout.toUpperCase();
      let tier = "UNKNOWN";
      if (upper.includes("CRITICAL")) tier = "CRITICAL";
      else if (upper.includes("HIGH RISK") || upper.includes("HIGH\n") || upper.includes("HIGH ")) tier = "HIGH";
      else if (upper.includes("CAUTION")) tier = "CAUTION";
      else if (upper.includes("SAFE")) tier = "SAFE";

      const blocked = upper.includes("BLOCKED") || upper.includes("INSTALL BLOCKED");
      const scoreMatch = stdout.match(/(?:Risk Score|score)[:\s]+([0-9.]+)/i);
      const score = scoreMatch ? parseFloat(scoreMatch[1]) : null;

      resolve({ tier, score, blocked });
    });

    child.on("error", reject);
    setTimeout(() => { child.kill(); reject(new Error("Scan timed out")); }, 90000);
  });
}

// ─── Status bar ───────────────────────────────────────────────────────────────
function updateStatusBar() {
  if (!statusBarItem) return;
  if (!kavachPath) {
    statusBarItem.text = "$(shield) KAVACH: Not installed";
    statusBarItem.tooltip = "Click to install KAVACH";
    statusBarItem.command = "kavach.showInstallGuide";
    statusBarItem.backgroundColor = new vscode.ThemeColor("statusBarItem.warningBackground");
  } else if (!kavachEnabled) {
    statusBarItem.text = "$(shield) KAVACH: Disabled";
    statusBarItem.tooltip = "KAVACH is disabled — click to enable";
    statusBarItem.command = "kavach.enable";
    statusBarItem.backgroundColor = new vscode.ThemeColor("statusBarItem.warningBackground");
  } else {
    statusBarItem.text = "$(shield) KAVACH";
    statusBarItem.tooltip = "KAVACH active — click to scan a package";
    statusBarItem.command = "kavach.scanPackage";
    statusBarItem.backgroundColor = undefined;
  }
  statusBarItem.show();
}

// ─── Scan command ─────────────────────────────────────────────────────────────
async function cmdScanPackage() {
  if (!kavachPath) {
    vscode.window.showErrorMessage("KAVACH not installed", "View Install Guide")
      .then(s => s && vscode.commands.executeCommand("kavach.showInstallGuide"));
    return;
  }

  const editor = vscode.window.activeTextEditor;
  let pkgName = "";
  if (editor) {
    const sel = editor.document.getText(editor.selection).trim().replace(/['"]/g, "");
    if (sel && !sel.includes(" ") && sel.length < 100) pkgName = sel;
  }

  if (!pkgName) {
    pkgName = await vscode.window.showInputBox({
      prompt: "Package name to scan",
      placeHolder: "e.g. axios, requests, lodash",
    });
  }
  if (!pkgName) return;

  const ecoItem = await vscode.window.showQuickPick([
    { label: "npm", description: "Node.js / JavaScript packages" },
    { label: "pypi", description: "Python packages" },
  ], { placeHolder: "Select ecosystem" });
  if (!ecoItem) return;

  outputChannel.show(true);
  statusBarItem.text = `$(sync~spin) Scanning ${pkgName}...`;
  statusBarItem.backgroundColor = undefined;

  try {
    const result = await runScan(pkgName, ecoItem.label);
    const icons = { SAFE: "✅", CAUTION: "⚠️", HIGH: "🔴", CRITICAL: "🚨", UNKNOWN: "❓" };

    outputChannel.appendLine(`\n${icons[result.tier]} Result: ${result.tier}${result.score != null ? ` (${result.score.toFixed(2)})` : ""}`);
    outputChannel.appendLine(result.blocked ? "🚫 INSTALL BLOCKED" : "✅ INSTALL ALLOWED");

    if (result.blocked || result.tier === "HIGH" || result.tier === "CRITICAL") {
      vscode.window.showErrorMessage(
        `KAVACH: ${pkgName} is ${result.tier} RISK — Do not install`,
        "View Details"
      ).then(s => s && outputChannel.show());
    } else if (result.tier === "CAUTION") {
      vscode.window.showWarningMessage(
        `KAVACH: ${pkgName} flagged as CAUTION — Review before installing`,
        "View Details"
      ).then(s => s && outputChannel.show());
    } else {
      vscode.window.showInformationMessage(`KAVACH: ${pkgName} is SAFE ✅`);
    }
  } catch (err) {
    outputChannel.appendLine(`\n❌ Error: ${err.message}`);
    vscode.window.showErrorMessage(`KAVACH scan failed: ${err.message}`);
  }

  updateStatusBar();
}

// ─── Enable/Disable ───────────────────────────────────────────────────────────
async function cmdDisable() {
  const confirm = await vscode.window.showWarningMessage(
    "Disable KAVACH supply chain protection?",
    { modal: true },
    "Disable"
  );
  if (confirm !== "Disable") return;

  kavachEnabled = false;

  // Run kavach-disable script
  const disableScript = IS_WINDOWS
    ? path.join(KAVACH_BIN, "kavach-disable.bat")
    : path.join(KAVACH_BIN, "kavach-disable");

  if (fs.existsSync(disableScript)) {
    exec(`"${disableScript}"`, (err, stdout) => {
      outputChannel.appendLine(stdout);
    });
  }

  updateStatusBar();
  vscode.window.showWarningMessage(
    "KAVACH disabled. Restart your terminal to apply. Shell intercepts removed.",
    "Re-enable"
  ).then(s => s && cmdEnable());
}

async function cmdEnable() {
  kavachEnabled = true;

  const enableScript = IS_WINDOWS
    ? path.join(KAVACH_BIN, "kavach-enable.bat")
    : path.join(KAVACH_BIN, "kavach-enable");

  if (fs.existsSync(enableScript)) {
    exec(`"${enableScript}"`, (err, stdout) => {
      outputChannel.appendLine(stdout);
    });
  }

  updateStatusBar();
  vscode.window.showInformationMessage("KAVACH re-enabled. Restart your terminal to apply.");
}

// ─── Install guide webview ────────────────────────────────────────────────────
function cmdShowInstallGuide() {
  const panel = vscode.window.createWebviewPanel("kavachInstall", "Install KAVACH", vscode.ViewColumn.One, {});

  const macCmd = `curl -sSL https://raw.githubusercontent.com/kavach-security/kavach/main/install.sh | bash`;
  const winCmd = `iex (irm https://raw.githubusercontent.com/kavach-security/kavach/main/install.ps1)`;
  const manualCmd = IS_WINDOWS
    ? `git clone https://github.com/kavach-security/kavach\ncd kavach\\vscode-extension\\installer\npowershell -ExecutionPolicy Bypass -File install.ps1`
    : `git clone https://github.com/kavach-security/kavach\nbash kavach/installer/install.sh`;

  panel.webview.html = `<!DOCTYPE html>
<html>
<head>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; padding: 32px; max-width: 680px; color: #1F2937; background: #F9FAFB; }
  h1 { color: #6D28D9; font-size: 24px; margin-bottom: 4px; }
  h2 { color: #374151; font-size: 16px; margin-top: 28px; }
  p  { color: #6B7280; line-height: 1.6; }
  pre { background: #111827; color: #E5E7EB; padding: 16px; border-radius: 10px; font-family: 'SF Mono', 'Fira Code', monospace; font-size: 13px; overflow-x: auto; }
  .step { background: white; border: 1px solid #E5E7EB; border-radius: 12px; padding: 18px; margin: 12px 0; }
  .num  { background: #6D28D9; color: white; border-radius: 50%; width: 26px; height: 26px; display: inline-flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 700; margin-right: 10px; vertical-align: middle; }
  .badge { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 11px; font-weight: 700; margin: 3px; }
  .mac { background: #EDE9FE; color: #6D28D9; }
  .win { background: #DBEAFE; color: #1D4ED8; }
  .safe { background: #ECFDF5; color: #059669; }
  .warn { background: #FEF3C7; color: #B45309; }
  code { background: #F3F4F6; padding: 2px 6px; border-radius: 4px; font-family: monospace; font-size: 12px; }
</style>
</head>
<body>
  <h1>🛡️ Install KAVACH</h1>
  <p>KAVACH intercepts every <code>npm install</code> and <code>pip install</code> command and scans it with 5 AI agents before any code executes.</p>

  <h2>Quick Install</h2>

  <div class="step">
    <span class="num">1</span> <strong>macOS / Linux</strong> <span class="badge mac">macOS</span> <span class="badge mac">Linux</span>
    <pre>${macCmd}</pre>
  </div>

  <div class="step">
    <span class="num">2</span> <strong>Windows</strong> <span class="badge win">Windows 10/11</span>
    <p style="font-size:12px;color:#6B7280">Open PowerShell as Administrator, then run:</p>
    <pre>${winCmd}</pre>
  </div>

  <div class="step">
    <span class="num">3</span> <strong>Manual install (if above fails)</strong>
    <pre>${manualCmd}</pre>
  </div>

  <h2>After Installing</h2>
  <div class="step">
    <p>1. <strong>Restart your terminal</strong> — shell intercepts activate automatically</p>
    <p>2. Test it:</p>
    <pre>npm install lodash       # Should show ✅ SAFE
npm install yoshi-base   # Should show 🚨 CRITICAL — BLOCKED</pre>
    <p>3. <strong>Reload VS Code</strong> — the KAVACH status bar will turn active</p>
  </div>

  <h2>Enable / Disable</h2>
  <div class="step">
    <p>In VS Code Command Palette (<code>Cmd+Shift+P</code> / <code>Ctrl+Shift+P</code>):</p>
    <p><strong>KAVACH: Disable Protection</strong> — removes shell intercepts</p>
    <p><strong>KAVACH: Enable Protection</strong> — restores shell intercepts</p>
    <p>Or from terminal: <code>kavach-disable</code> / <code>kavach-enable</code></p>
  </div>
</body>
</html>`;
}

// ─── Activate ─────────────────────────────────────────────────────────────────
async function activate(context) {
  outputChannel = vscode.window.createOutputChannel("KAVACH Security");
  statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
  context.subscriptions.push(statusBarItem, outputChannel);

  outputChannel.appendLine("🛡️  KAVACH Security Extension starting...");
  outputChannel.appendLine(`Platform: ${process.platform}`);
  outputChannel.appendLine(`Looking for kavach-standalone...`);

  kavachPath = await findKavach();

  if (kavachPath) {
    outputChannel.appendLine(`✅ Found: ${kavachPath}`);
  } else {
    outputChannel.appendLine("⚠️  kavach-standalone not found. Use Command Palette → KAVACH: Show Install Guide");
  }

  updateStatusBar();

  context.subscriptions.push(
    vscode.commands.registerCommand("kavach.scanPackage",    cmdScanPackage),
    vscode.commands.registerCommand("kavach.disable",        cmdDisable),
    vscode.commands.registerCommand("kavach.enable",         cmdEnable),
    vscode.commands.registerCommand("kavach.showInstallGuide", cmdShowInstallGuide),
    vscode.commands.registerCommand("kavach.openDashboard",  () => {
      vscode.env.openExternal(vscode.Uri.parse("http://localhost:3000"));
    }),
  );

  // If kavach not found, prompt to install
  if (!kavachPath) {
    vscode.window.showWarningMessage(
      "KAVACH CLI not found on this machine.",
      "View Install Guide"
    ).then(s => s && cmdShowInstallGuide());
  }
}

function deactivate() {
  statusBarItem?.dispose();
  outputChannel?.dispose();
}

module.exports = { activate, deactivate };
