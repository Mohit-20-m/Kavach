#!/usr/bin/env python3
"""
KAVACH CLI Wrapper
==================
Intercepts npm install / pip install commands before execution.
Scans the package, shows verdict, blocks if critical/high risk.

Usage:
  kavach-npm install <package>     (replaces npm install)
  kavach-pip install <package>     (replaces pip install)
  kavach scan <package>            (manual scan)
  kavach setup                     (setup shell aliases)
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from typing import Optional

import httpx
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text
from rich import box

app = typer.Typer(
    name="kavach",
    help="🛡️  KAVACH — Intelligent Supply Chain Security",
    add_completion=False,
)
console = Console()

KAVACH_API = os.getenv("KAVACH_API_URL", "http://localhost:8000")

# Risk tier colors
TIER_COLORS = {
    "SAFE": "bold green",
    "CAUTION": "bold yellow",
    "HIGH": "bold red",
    "CRITICAL": "bold red on white",
}

TIER_ICONS = {
    "SAFE": "✅",
    "CAUTION": "⚠️ ",
    "HIGH": "🔴",
    "CRITICAL": "🚨",
}


# ─── Core scan function ───────────────────────────────────────────────────────

async def _scan_package(
    package_name: str,
    ecosystem: str,
    show_details: bool = True,
) -> Optional[dict]:
    """Call KAVACH backend API to scan a package."""

    with Progress(
        SpinnerColumn(),
        TextColumn(f"[cyan]KAVACH scanning [bold]{package_name}[/bold]..."),
        transient=True,
        console=console,
    ) as progress:
        progress.add_task("scan", total=None)

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{KAVACH_API}/api/v1/scan/",
                    json={
                        "package_name": package_name,
                        "ecosystem": ecosystem,
                        "source": "cli",
                    },
                )

            if resp.status_code != 200:
                console.print(
                    f"[yellow]⚠️  KAVACH API error ({resp.status_code}) — "
                    f"proceeding without scan[/yellow]"
                )
                return None

            return resp.json()

        except httpx.ConnectError:
            console.print(
                "[yellow]⚠️  KAVACH backend not running — "
                "install without scanning (run: docker-compose up)[/yellow]"
            )
            return None
        except Exception as e:
            console.print(f"[yellow]⚠️  Scan error: {e}[/yellow]")
            return None


def _display_verdict(result: dict, package_name: str):
    """Display a rich, formatted verdict in the terminal."""
    tier = result.get("risk_tier", "SAFE")
    score = result.get("risk_score", 0.0)
    blocked = result.get("install_blocked", False)
    color = TIER_COLORS.get(tier, "white")
    icon = TIER_ICONS.get(tier, "?")
    exec_ms = result.get("execution_time_ms", 0)

    # ── Header ────────────────────────────────────────────────────────────────
    console.print()
    console.rule(f"[cyan]🛡️  KAVACH Security Scan — {package_name}[/cyan]")

    # ── Risk verdict ──────────────────────────────────────────────────────────
    verdict_text = Text()
    verdict_text.append(f"\n  {icon} Risk Tier: ", style="bold white")
    verdict_text.append(f"{tier}", style=color)
    verdict_text.append(f"   Score: ", style="bold white")
    verdict_text.append(f"{score:.2f}/1.0", style=color)
    verdict_text.append(f"   Analysis: {exec_ms:.0f}ms\n", style="dim")

    if blocked:
        panel_style = "red"
        title = f"[bold red] {icon} INSTALL BLOCKED — {tier} RISK DETECTED[/bold red]"
    elif tier == "CAUTION":
        panel_style = "yellow"
        title = f"[bold yellow]{icon} CAUTION — Review before installing[/bold yellow]"
    else:
        panel_style = "green"
        title = f"[bold green]{icon} SAFE — Package cleared all checks[/bold green]"

    console.print(Panel(verdict_text, title=title, border_style=panel_style))

    # ── Agent breakdown ───────────────────────────────────────────────────────
    agent_scores = result.get("agent_scores", {})
    if agent_scores:
        table = Table(
            title="Agent Analysis",
            box=box.ROUNDED,
            border_style="cyan",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Agent", style="white", min_width=25)
        table.add_column("Risk Score", justify="center", min_width=12)
        table.add_column("Confidence", justify="center", min_width=12)
        table.add_column("Time", justify="right", min_width=10)

        agent_display = {
            "code_archaeologist": "🔬 Code Archaeologist",
            "dependency_graph": "🕸️  Dependency Graph",
            "maintainer_trust": "👤 Maintainer Trust",
            "behavioral_anomaly": "📈 Behavioral Anomaly",
            "semantic_intent": "🧠 Semantic Intent",
        }

        for agent_key, agent_data in agent_scores.items():
            a_score = agent_data.get("risk_score", 0)
            a_conf = agent_data.get("confidence", 0)
            a_time = agent_data.get("execution_time_ms", 0)

            # Color score
            if a_score > 0.7:
                score_str = f"[bold red]{a_score:.2f}[/bold red]"
            elif a_score > 0.4:
                score_str = f"[yellow]{a_score:.2f}[/yellow]"
            else:
                score_str = f"[green]{a_score:.2f}[/green]"

            table.add_row(
                agent_display.get(agent_key, agent_key),
                score_str,
                f"{a_conf:.2f}",
                f"{a_time:.0f}ms",
            )

        console.print(table)

    # ── Plain English Explanation ─────────────────────────────────────────────
    explanation = result.get("plain_english_explanation", "")
    if explanation:
        console.print(
            Panel(
                explanation,
                title="[bold white]🔍 Security Analysis[/bold white]",
                border_style="blue",
                padding=(1, 2),
            )
        )

    # ── Top Evidence ─────────────────────────────────────────────────────────
    evidence = result.get("evidence_summary", [])
    critical_evidence = [e for e in evidence if e.get("severity") == "critical"]
    if critical_evidence:
        console.print("\n[bold red]🚨 Critical Findings:[/bold red]")
        for ev in critical_evidence[:5]:
            console.print(f"  [red]•[/red] {ev.get('description', '')}")

    # ── Safe Alternatives ─────────────────────────────────────────────────────
    alternatives = result.get("safe_alternatives", [])
    if alternatives and blocked:
        console.print("\n[bold green]✅ Safe Alternatives:[/bold green]")
        for alt in alternatives[:3]:
            score_badge = f"[green]score: {alt.get('score', '?')}[/green]"
            console.print(
                f"  [green]→[/green] [bold]{alt['name']}[/bold] "
                f"— {alt.get('description', '')} ({score_badge})"
            )

    # ── Similar Historical Attacks ────────────────────────────────────────────
    attacks = result.get("similar_attacks", [])
    if attacks and tier in ("HIGH", "CRITICAL"):
        console.print("\n[bold yellow]⚠️  Similar Historical Attacks:[/bold yellow]")
        for attack in attacks[:2]:
            similarity = attack.get("similarity", 0)
            console.print(
                f"  [yellow]~[/yellow] {attack['title']} ({attack['year']}) "
                f"— {similarity:.0%} pattern match"
            )

    console.print()


def _execute_install(
    package_name: str, ecosystem: str, extra_args: list[str]
):
    """Execute the real package manager install command."""
    if ecosystem == "npm":
        cmd = ["npm", "install", package_name] + extra_args
    else:
        cmd = ["pip", "install", package_name] + extra_args

    console.print(f"[dim]→ Executing: {' '.join(cmd)}[/dim]")
    result = subprocess.run(cmd)
    return result.returncode


# ─── CLI Commands ─────────────────────────────────────────────────────────────

@app.command("npm")
def kavach_npm(
    args: list[str] = typer.Argument(..., help="npm arguments"),
):
    """
    KAVACH-wrapped npm command.
    Intercepts 'install' and 'i' subcommands for security scanning.
    """
    if not args:
        subprocess.run(["npm"] + list(args))
        return

    subcommand = args[0]

    # Only intercept install commands
    if subcommand not in ("install", "i", "add"):
        subprocess.run(["npm"] + list(args))
        return

    # Extract package names (skip flags like --save-dev)
    packages = [a for a in args[1:] if not a.startswith("-")]

    if not packages:
        # npm install without package — install from package.json, skip scan
        subprocess.run(["npm"] + list(args))
        return

    extra_flags = [a for a in args[1:] if a.startswith("-")]

    all_blocked = False

    for package in packages:
        # Clean version from package name (e.g., package@1.0.0 → package)
        pkg_name = package.split("@")[0] if "@" in package and not package.startswith("@") else package

        result = asyncio.run(_scan_package(pkg_name, "npm"))

        if result:
            _display_verdict(result, pkg_name)

            if result.get("install_blocked"):
                all_blocked = True
                console.print(
                    f"[bold red]🚫 Installation of '{pkg_name}' has been blocked by KAVACH.[/bold red]\n"
                )
            else:
                # Proceed with install
                returncode = _execute_install(package, "npm", extra_flags)
                if returncode != 0:
                    sys.exit(returncode)
        else:
            # API unavailable — proceed with install
            _execute_install(package, "npm", extra_flags)

    if all_blocked:
        sys.exit(1)


@app.command("pip")
def kavach_pip(
    args: list[str] = typer.Argument(..., help="pip arguments"),
):
    """KAVACH-wrapped pip command."""
    if not args or args[0] != "install":
        subprocess.run(["pip"] + list(args))
        return

    packages = [a for a in args[1:] if not a.startswith("-")]
    extra_flags = [a for a in args[1:] if a.startswith("-")]

    all_blocked = False

    for package in packages:
        pkg_name = package.split("==")[0].split(">=")[0].split("<=")[0].strip()

        result = asyncio.run(_scan_package(pkg_name, "pypi"))

        if result:
            _display_verdict(result, pkg_name)

            if result.get("install_blocked"):
                all_blocked = True
                console.print(
                    f"[bold red]🚫 Installation of '{pkg_name}' blocked by KAVACH.[/bold red]\n"
                )
            else:
                _execute_install(package, "pypi", extra_flags)
        else:
            _execute_install(package, "pypi", extra_flags)

    if all_blocked:
        sys.exit(1)


@app.command("scan")
def manual_scan(
    package: str = typer.Argument(..., help="Package name to scan"),
    ecosystem: str = typer.Option("npm", "--ecosystem", "-e", help="npm or pypi"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Manually scan a package without installing it."""
    result = asyncio.run(_scan_package(package, ecosystem))

    if result is None:
        console.print("[red]Scan failed — is KAVACH backend running?[/red]")
        raise typer.Exit(1)

    if json_output:
        print(json.dumps(result, indent=2))
    else:
        _display_verdict(result, package)

    tier = result.get("risk_tier", "SAFE")
    if tier in ("HIGH", "CRITICAL"):
        raise typer.Exit(1)


@app.command("setup")
def setup_aliases():
    """
    Setup shell aliases so 'npm' and 'pip' automatically route through KAVACH.
    Adds aliases to ~/.bashrc and ~/.zshrc.
    """
    alias_lines = [
        "\n# KAVACH Supply Chain Security",
        "alias npm='kavach npm'",
        "alias pip='kavach pip'",
        "alias pip3='kavach pip'",
    ]

    shells = []
    home = os.path.expanduser("~")

    for rc_file in [".bashrc", ".zshrc", ".bash_profile"]:
        rc_path = os.path.join(home, rc_file)
        if os.path.exists(rc_path):
            shells.append(rc_path)

    for shell_rc in shells:
        with open(shell_rc, "r") as f:
            content = f.read()

        if "KAVACH Supply Chain Security" not in content:
            with open(shell_rc, "a") as f:
                f.write("\n".join(alias_lines) + "\n")
            console.print(f"[green]✅ Added KAVACH aliases to {shell_rc}[/green]")
        else:
            console.print(f"[dim]Aliases already in {shell_rc}[/dim]")

    console.print(
        Panel(
            "[bold green]KAVACH is now active![/bold green]\n\n"
            "Restart your terminal or run:\n"
            "  [cyan]source ~/.bashrc[/cyan]  (or ~/.zshrc)\n\n"
            "Now all npm/pip installs are automatically scanned.",
            title="🛡️  Setup Complete",
            border_style="green",
        )
    )


@app.command("status")
def status():
    """Check KAVACH backend status."""
    try:
        result = asyncio.run(
            httpx.AsyncClient().get(f"{KAVACH_API}/api/v1/health/")
        )
        if result.status_code == 200:
            console.print("[green]✅ KAVACH backend is running[/green]")
        else:
            console.print(f"[red]❌ Backend returned {result.status_code}[/red]")
    except Exception:
        console.print(
            "[red]❌ KAVACH backend is not running[/red]\n"
            "[dim]Start it with: docker-compose up[/dim]"
        )


if __name__ == "__main__":
    app()
