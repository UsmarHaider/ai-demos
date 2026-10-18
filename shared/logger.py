"""Teaching logger.

Everything these demos do is printed to the terminal in colour AND written to a
log file in demo_project/logs/. The goal is that a student watching the screen
can answer: what was loaded, what was sent to the model, which tool ran, with
what arguments, how long it took, and how many tokens it cost.

Nothing here is magic - it is just `rich` printing plus Python's `logging`.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


class DemoLogger:
    """Pretty console output for live demos + a plain-text log file for later."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.console = Console(highlight=False)
        self._step_no = 0
        self._t0 = time.perf_counter()

        logfile = LOG_DIR / f"{name.lower().replace(' ', '_')}_{datetime.now():%Y-%m-%d}.log"
        self.logfile = logfile
        self._file_log = logging.getLogger(f"demo.{name}")
        self._file_log.setLevel(logging.DEBUG)
        self._file_log.propagate = False
        if not self._file_log.handlers:
            handler = logging.FileHandler(logfile, encoding="utf-8")
            handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
            )
            self._file_log.addHandler(handler)

    # ---------------------------------------------------------------- plumbing
    def _file(self, level: int, msg: str) -> None:
        self._file_log.log(level, msg.replace("\n", " ⏎ "))

    # ------------------------------------------------------------------ layout
    def banner(self, title: str, subtitle: str = "") -> None:
        body = Text(title, style="bold white")
        if subtitle:
            body.append("\n" + subtitle, style="dim white")
        self.console.print()
        self.console.print(Panel(body, border_style="bright_blue", padding=(1, 4)))
        self._file(logging.INFO, f"=== {title} === {subtitle}")

    def step(self, title: str) -> None:
        """A numbered pipeline stage. Students follow these like chapter headings."""
        self._step_no += 1
        self.console.print()
        self.console.print(
            Rule(f"[bold cyan]STEP {self._step_no}: {escape(title)}[/]", style="cyan", align="left")
        )
        self._file(logging.INFO, f"STEP {self._step_no}: {title}")

    def substep(self, text: str) -> None:
        self.console.print(f"  [cyan]›[/] {escape(text)}")
        self._file(logging.DEBUG, f"  > {text}")

    # ------------------------------------------------------------------- facts
    def kv(self, label: str, value: Any, style: str = "white") -> None:
        self.console.print(f"  [dim]{escape(label):.<28}[/] [{style}]{escape(str(value))}[/]")
        self._file(logging.INFO, f"  {label}: {value}")

    def ok(self, text: str) -> None:
        self.console.print(f"  [bold green]✓[/] {escape(text)}")
        self._file(logging.INFO, f"OK  {text}")

    def warn(self, text: str) -> None:
        self.console.print(f"  [bold yellow]![/] {escape(text)}")
        self._file(logging.WARNING, text)

    def error(self, text: str) -> None:
        self.console.print(f"  [bold red]✗[/] [red]{escape(text)}[/]")
        self._file(logging.ERROR, text)

    def note(self, text: str) -> None:
        """A teaching aside - explains WHY the previous line happened."""
        self.console.print(f"  [dim italic]{escape(text)}[/]")
        self._file(logging.DEBUG, f"note: {text}")

    # --------------------------------------------------------------------- LLM
    def llm_request(self, model: str, prompt: str, *, show_prompt: bool = True) -> None:
        self.console.print(f"  [magenta]⟶ LLM REQUEST[/] [dim]model=[/]{escape(model)}")
        self._file(logging.INFO, f"LLM REQUEST model={model}\n{prompt}")
        if show_prompt:
            self.console.print(
                Panel(
                    escape(prompt),
                    title="[magenta]exact prompt sent to the model[/]",
                    border_style="magenta",
                    padding=(0, 1),
                )
            )

    def llm_response(self, text: str, *, seconds: float | None = None, usage: dict | None = None) -> None:
        bits = []
        if seconds is not None:
            bits.append(f"{seconds:.2f}s")
        if usage:
            bits.append(
                f"in={usage.get('input_tokens', '?')} out={usage.get('output_tokens', '?')} "
                f"total={usage.get('total_tokens', '?')} tokens"
            )
        meta = ("  [dim]" + " · ".join(bits) + "[/]") if bits else ""
        self.console.print(f"  [magenta]⟵ LLM RESPONSE[/]{meta}")
        self._file(logging.INFO, f"LLM RESPONSE ({' '.join(bits)})\n{text}")

    def answer(self, text: str, title: str = "ANSWER") -> None:
        self.console.print()
        self.console.print(
            Panel(escape(text), title=f"[bold green]{title}[/]", border_style="green", padding=(1, 2))
        )
        self._file(logging.INFO, f"{title}: {text}")

    # ------------------------------------------------------------------- tools
    def tool_call(self, name: str, args: dict) -> None:
        self.console.print(
            f"  [bold yellow]🔧 TOOL CALL[/] [yellow]{escape(name)}[/]"
            f"({escape(json.dumps(args, default=str))})"
        )
        self._file(logging.INFO, f"TOOL CALL {name}({json.dumps(args, default=str)})")

    def tool_result(self, name: str, result: Any, seconds: float) -> None:
        self.console.print(
            f"  [bold yellow]🔧 TOOL RESULT[/] [yellow]{escape(name)}[/] "
            f"[dim]({seconds * 1000:.0f} ms)[/] → {escape(str(result))}"
        )
        self._file(logging.INFO, f"TOOL RESULT {name} ({seconds*1000:.0f}ms) -> {result}")

    # ------------------------------------------------------------------ tables
    def table(self, title: str, columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
        t = Table(title=title, title_style="bold green", header_style="bold", box=None, pad_edge=False)
        for c in columns:
            t.add_column(c)
        for r in rows:
            t.add_row(*[escape(str(c)) for c in r])
        self.console.print(t)
        self._file(logging.INFO, f"TABLE {title}: cols={list(columns)} rows={[list(map(str, r)) for r in rows]}")

    def code(self, source: str, language: str = "python", title: str = "") -> None:
        self.console.print(Panel(Syntax(source, language, theme="monokai", word_wrap=True), title=title))

    # ------------------------------------------------------------------ timing
    @contextmanager
    def timed(self, label: str):
        """`with log.timed("Embedding"):` - prints how long the block took."""
        start = time.perf_counter()
        self.console.print(f"  [blue]⏱  {escape(label)}…[/]")
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.console.print(f"  [blue]⏱  {escape(label)} finished in {elapsed:.2f}s[/]")
            self._file(logging.INFO, f"TIMING {label}: {elapsed:.3f}s")

    def summary(self, rows: dict[str, Any]) -> None:
        t = Table(box=None, show_header=False, pad_edge=False)
        t.add_column(style="dim")
        t.add_column(style="bold white")
        for k, v in rows.items():
            t.add_row(k, str(v))
        self.console.print()
        self.console.print(Panel(t, title="[bold blue]RUN SUMMARY[/]", border_style="blue"))
        self._file(logging.INFO, f"SUMMARY {rows}")
        self.console.print(f"[dim]Full log written to: {self.logfile}[/]")

    def elapsed(self) -> float:
        return time.perf_counter() - self._t0


def get_logger(name: str) -> DemoLogger:
    return DemoLogger(name)
