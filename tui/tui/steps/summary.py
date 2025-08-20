# Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
# Licensed under the MIT License

from __future__ import annotations
import json
import asyncio
import os
import stat
import re
from pathlib import Path
from time import sleep
from textual.widgets import Button, Static, Header, Footer, RichLog, Label, Input
from textual.containers import Horizontal, Vertical
from textual.app import ComposeResult
from tui.steps.base import StepScreen
from textual.screen import Screen
from config_loader import BINE_DIR
from typing import Any, Dict, Iterable, List, Tuple, Union

JsonLike = Union[Dict[str, Any], str, Path]


# TODO: Temporary, thanks GPT :)
def json_to_exports(config: JsonLike, sh_path: Union[str, Path]) -> str:
    """
    Parse the given JSON config (dict or file path) and write a shell script that exports variables
    per the agreed specification. Returns the output .sh path as a string.

    Key points:
    - Shebang included; no header block (only # skipped: comments when something's missing).
    - Strings => double-quoted; numbers => unquoted; booleans => "yes"/"no".
    - environment.other_var: numeric values unquoted; string values double-quoted.
    - MODULES is a single consolidated line ordered as: lib_load (if any), gpu_load (if GPU awareness is yes), python_module.
    - GPU awareness = "yes" iff any non-zero value exists in libraries[0].tests.gpu.
      Only then export GPU_LIB/GPU_LIB_VERSION and include the GPU module.
    - QOS exported only if qos.is_required is true.
    - TASKS_PER_NODE omitted if empty/missing; GPU_PER_NODE is "0" if empty/missing; GPU_AWARENESS "yes"/"no".
    - Collectives: export COLLECTIVES and, per collective, *_ALGORITHMS and *_ALGORITHMS_SKIP (yes if any constraint has key == "count").
    - UTF-8 I/O; overwrite output; mark executable.
    """
    # --- Load JSON ---
    data: Dict[str, Any]
    if isinstance(config, (str, Path)):
        with open(config, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif isinstance(config, dict):
        data = config
    else:
        raise TypeError("config must be a dict, str path, or Path")

    out_path = Path(sh_path)
    lines: List[str] = []

    # helpers -------------
    def is_number_like(v: Any) -> bool:
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return True
        if isinstance(v, str):
            return bool(re.fullmatch(r"[+-]?\d+(?:\.\d+)?", v))
        return False

    def bash_escape_double_quoted(s: str) -> str:
        # Escape backslash, dollar, and double quote to be safe inside "..."
        return s.replace("\\", "\\\\").replace("$", "\\$").replace('"', '\\"')

    def dq(s: str) -> str:
        return f"\"{bash_escape_double_quoted(str(s))}\""

    def yesno(b: Any) -> str:
        return "yes" if bool(b) else "no"

    def csv(values: Iterable[Any]) -> str:
        # Join without spaces
        return ",".join(str(v) for v in values)

    def write_export(name: str, value: Any, *, quote: bool) -> None:
        if quote:
            lines.append(f'export {name}={dq(value)}')
        else:
            lines.append(f'export {name}={value}')

    def mpi_lib_tag(lib_type: str) -> str:
        lt = lib_type.strip().lower()
        if "open" in lt and "mpi" in lt:
            return "OMPI"
        if "cray" in lt:
            return "CRAY_MPICH"
        if "mpich" in lt and "cray" not in lt:
            return "MPICH"
        # Generic fallback: uppercase and normalize separators
        tag = re.sub(r"[^A-Za-z0-9]+", "_", lib_type).upper().strip("_")
        return tag

    def parse_module_name_version(mod: str) -> Tuple[str, str]:
        # Expect "<name>/<version>" or just "<name>"
        parts = str(mod).split("/", 1)
        name = parts[0]
        version = parts[1] if len(parts) > 1 else ""
        return name, version

    # --- Begin script (shebang only) ---
    lines.append("#!/bin/bash")

    # We'll assemble MODULES parts then write one consolidated export later
    python_module: str | None = None
    lib_module: str | None = None
    gpu_module: str | None = None  # only filled when GPU awareness is yes

    # ========== environment ==========
    env = data.get("environment")
    if not isinstance(env, dict):
        lines.append("# skipped: environment section not found")
    else:
        # LOCATION
        if "name" in env:
            write_export("LOCATION", env["name"], quote=True)
        else:
            lines.append("# skipped: environment.name missing")

        # RUN
        write_export("RUN", "srun" if env.get("slurm") else "mpirun", quote=False)

        # python_module (tail of MODULES)
        py_mod = env.get("python_module")
        if py_mod:
            python_module = str(py_mod)
        else:
            lines.append("# skipped: environment.python_module missing (MODULES may be partial)")

        # other_var (export all key-value pairs)
        other_var = env.get("other_var")
        if isinstance(other_var, dict):
            for k, v in other_var.items():
                if is_number_like(v):
                    write_export(str(k), v, quote=False)
                else:
                    write_export(str(k), v, quote=True)

        # PARTITION and optional QOS
        part = env.get("partition")
        if isinstance(part, dict):
            if "name" in part:
                write_export("PARTITION", part["name"], quote=True)
            else:
                lines.append("# skipped: environment.partition.name missing")
            qos = part.get("qos")
            if isinstance(qos, dict):
                if qos.get("is_required", False):
                    qname = qos.get("name")
                    if qname:
                        write_export("QOS", qname, quote=True)
                    else:
                        lines.append("# skipped: QOS required but name missing")
                    extra_reqs = qos.get("extra_requirements")
                    if isinstance(extra_reqs, dict):
                        for k, v in extra_reqs.items():
                            if is_number_like(v):
                                write_export(f"QOS_{k.upper()}", v, quote=False)
                            else:
                                write_export(f"QOS_{k.upper()}", v, quote=True)
        else:
            lines.append("# skipped: environment.partition missing")

    # ========== test ==========
    test = data.get("test")
    if not isinstance(test, dict):
        lines.append("# skipped: test section not found")
    else:
        # booleans
        for src_key, out_name in [
            ("compile_only", "COMPILE_ONLY"),
            ("debug_mode", "DEBUG_MODE"),
            ("dry_run", "DRY_RUN"),
            ("delete", "DELETE"),
            ("compress", "COMPRESS"),
        ]:
            if src_key in test:
                write_export(out_name, yesno(test[src_key]), quote=True)
            else:
                lines.append(f"# skipped: test.{src_key} missing")

        # numbers / strings
        if "number_of_nodes" in test:
            write_export("N_NODES", test["number_of_nodes"], quote=False)
        else:
            lines.append("# skipped: test.number_of_nodes missing")

        if "output_level" in test:
            write_export("OUTPUT_LEVEL", test["output_level"], quote=True)
        else:
            lines.append("# skipped: test.output_level missing")

        if "test_time" in test:
            write_export("TEST_TIME", test["test_time"], quote=True)
        else:
            lines.append("# skipped: test.test_time missing")

        # dimensions
        dims = test.get("dimensions")
        if not isinstance(dims, dict):
            lines.append("# skipped: test.dimensions missing")
        else:
            # TYPES
            if "dtype" in dims:
                write_export("TYPES", dims["dtype"], quote=True)
            else:
                lines.append("# skipped: test.dimensions.dtype missing")

            # SIZES from sizes_elements (CSV as string)
            if "sizes_elements" in dims and isinstance(dims["sizes_elements"], list):
                sizes_csv = csv(dims["sizes_elements"])
                write_export("SIZES", sizes_csv, quote=True)
            else:
                lines.append("# skipped: test.dimensions.sizes_elements missing or not a list")

            # SEGMENT_SIZES from segsizes_bytes (CSV as string)
            if "segsizes_bytes" in dims and isinstance(dims["segsizes_bytes"], list):
                segs_csv = csv(dims["segsizes_bytes"])
                write_export("SEGMENT_SIZES", segs_csv, quote=True)
            else:
                lines.append("# skipped: test.dimensions.segsizes_bytes missing or not a list")

    # ========== libraries (assume first) ==========
    libs = data.get("libraries")
    first_lib = libs[0] if isinstance(libs, list) and libs else None
    gpu_awareness_yes = False
    lib_type = ''
    if not isinstance(first_lib, dict):
        lines.append("# skipped: libraries[0] not found")
    else:
        # MPI_LIB from lib_type
        lib_type = first_lib.get("lib_type")
        if lib_type:
            write_export("MPI_LIB", mpi_lib_tag(str(lib_type)), quote=True)
        else:
            lines.append("# skipped: libraries[0].lib_type missing")

        # PICOCC from compiler
        comp = first_lib.get("compiler")
        if comp:
            write_export("PICOCC", comp, quote=True)
        else:
            lines.append("# skipped: libraries[0].compiler missing")

        # MPI_LIB_VERSION
        ver = first_lib.get("version")
        if ver:
            write_export("MPI_LIB_VERSION", ver, quote=True)
        else:
            lines.append("# skipped: libraries[0].version missing")

        # TASKS_PER_NODE from tests.cpu
        tests_lib = first_lib.get("tests", {})
        cpu_list = tests_lib.get("cpu")
        if isinstance(cpu_list, list) and len(cpu_list) > 0:
            write_export("TASKS_PER_NODE", csv(cpu_list), quote=True)
        elif cpu_list is None:
            lines.append("# skipped: libraries[0].tests.cpu missing (TASKS_PER_NODE not exported)")
        else:
            lines.append("# skipped: libraries[0].tests.cpu empty (TASKS_PER_NODE not exported)")

        # GPU_PER_NODE from tests.gpu (default "0" if empty/missing)
        gpu_list = tests_lib.get("gpu")
        if isinstance(gpu_list, list) and len(gpu_list) > 0:
            gpu_per_node_csv = csv(gpu_list)
            write_export("GPU_PER_NODE", gpu_per_node_csv, quote=True)
            # GPU awareness = yes if any non-zero value exists
            gpu_awareness_yes = any(int(v) != 0 for v in gpu_list if is_number_like(v))
            write_export("GPU_AWARENESS", "yes" if gpu_awareness_yes else "no", quote=True)

        # lib_load module candidate
        lib_load = first_lib.get("lib_load")
        if isinstance(lib_load, dict) and lib_load.get("type") == "module" and lib_load.get("module"):
            lib_module = str(lib_load["module"])

        # GPU module & GPU_LIB vars only when awareness is yes
        if gpu_awareness_yes:
            gpu_support = first_lib.get("gpu_support", {})
            gpu_load = gpu_support.get("gpu_load", {}) if isinstance(gpu_support, dict) else {}
            if isinstance(gpu_load, dict) and gpu_load.get("type") == "module" and gpu_load.get("module"):
                mod = str(gpu_load["module"])
                name, version = parse_module_name_version(mod)
                write_export("GPU_LIB", name, quote=True)
                if version:
                    write_export("GPU_LIB_VERSION", version, quote=True)
                gpu_module = mod
            else:
                lines.append("# skipped: gpu module load not available (GPU_LIB/GPU_LIB_VERSION not exported)")

    # After we've gathered modules, build final ordered MODULES:
    # order = [lib_module (if any), gpu_module (iff awareness yes), python_module (if any)]
    modules_ordered: List[str] = []
    if lib_module:
        modules_ordered.append(lib_module)
    if gpu_module:
        modules_ordered.append(gpu_module)
    if python_module:
        modules_ordered.append(python_module)

    if modules_ordered:
        write_export("MODULES", csv(modules_ordered), quote=True)
    else:
        lines.append("# skipped: MODULES not written (no modules determined)")

    # ========== collectives & algorithms ==========
    if isinstance(first_lib, dict):
        algos = first_lib.get("algorithms")
        if not isinstance(algos, dict):
            lines.append("# skipped: libraries[0].algorithms missing")
        else:
            # preserve original key order
            collective_keys = list(algos.keys())
            if collective_keys:
                write_export("COLLECTIVES", csv(collective_keys), quote=True)
            else:
                lines.append("# skipped: algorithms empty (COLLECTIVES not exported)")

            for coll_key in collective_keys:
                entries = algos.get(coll_key)
                if not isinstance(entries, list) or not entries:
                    lines.append(f"# skipped: algorithms.{coll_key} empty")
                    continue

                # names (CSV)
                names = [str(e.get("name", "")) for e in entries]
                write_export(f"{coll_key.upper()}_ALGORITHMS", csv(names), quote=True)

                # skip flags (yes if any constraint has key == "count")
                skips: List[str] = []
                for e in entries:
                    cons = e.get("constraints", [])
                    has_count = False
                    if isinstance(cons, list):
                        for c in cons:
                            if isinstance(c, dict) and c.get("key") == "count":
                                has_count = True
                                break
                    if has_count:
                        skips.append(str(e.get("name", "")))
                lines.append(f'export {coll_key.upper()}_ALGORITHMS_SKIP="{",".join(skips)}"')

                # is segmented flags (yes if any constraint has segmented in tags
                seg: List[str] = []
                for e in entries:
                    tags = e.get("tags", [])
                    seg.append("no" if "is_segmented" not in tags else "yes")
                lines.append(f'export {coll_key.upper()}_ALGORITHMS_IS_SEGMENTED="{",".join(seg)}"')

                if lib_type and "mpich" in lib_type.lower():
                    cvar: List[str] = []
                    for e in entries:
                        var = e.get("selection", "auto")
                        cvar.append(var if var != "pico" else "auto")
                    lines.append(f'export {coll_key.upper()}_ALGORITHMS_CVARS="{",".join(cvar)}"')



    # --- Write file and chmod +x ---
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")

    # Make executable
    mode = os.stat(out_path).st_mode
    os.chmod(out_path, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    return str(out_path)


SAVE_MSG =  "███████╗ █████╗ ██╗   ██╗███████╗  ██████╗ \n"\
            "██╔════╝██╔══██╗██║   ██║██╔════╝  ╚════██╗\n"\
            "███████╗███████║██║   ██║█████╗      ▄███╔╝\n"\
            "╚════██║██╔══██║╚██╗ ██╔╝██╔══╝      ▀▀══╝ \n"\
            "███████║██║  ██║ ╚████╔╝ ███████╗    ██╗   \n"\
            "╚══════╝╚═╝  ╚═╝  ╚═══╝  ╚══════╝    ╚═╝   \n"

TEST_DIR = BINE_DIR / "tests"

class SaveScreen(Screen):
    BINDINGS = [
        ("Tab", "focus_next", "Focus Next"),
        ("Shift+Tab", "focus_previous", "Focus Previous"),
        ("Enter", "select_item", "Select Item"),
        ("q", "request_quit", "Quit")
    ]

    __data: dict

    def __init__(self, json: dict) -> None:
        super().__init__()
        self.__data = json

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Vertical(
            Label(SAVE_MSG, id="question", classes="save-label"),
            Static("Files will be saved in `./tests` directory.", classes="field-label"),
            Input(placeholder="Enter filename to save as...", id="filename-input"),
            Label("", id="path-error", classes="error"),
            Horizontal(
                Button("Save", id="save", disabled=True),
                Button("Cancel", id="cancel"),
                classes="quit-button-row"
            ),
            id="save-dialog",
        )
        yield Footer()


    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save":
            raw = self.query_one("#filename-input", Input).value.strip()
            base_name = Path(raw).name
            stem = Path(base_name).stem
            json_name = stem + ".json" if not base_name.lower().endswith(".json") else base_name
            sh_name   = stem + ".sh"
            TEST_DIR.mkdir(exist_ok=True)

            suffix_id = 0
            while True:
                name = f"{Path(json_name).stem}_{suffix_id}.json" if suffix_id else json_name
                shn  = f"{Path(sh_name).stem}_{suffix_id}.sh"     if suffix_id else sh_name

                target = (TEST_DIR / name).resolve()
                sh_target = (TEST_DIR / shn).resolve()

                if TEST_DIR.resolve() in target.parents and TEST_DIR.resolve() in sh_target.parents:
                    if not target.exists() and not sh_target.exists():
                        break
                suffix_id += 1

            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                await asyncio.to_thread(
                    lambda: target.write_text(
                        json.dumps(self.__data, indent=2, ensure_ascii=False),
                        encoding="utf-8"
                    )
                )
                await asyncio.to_thread(json_to_exports, self.__data, sh_target)

            except Exception as e:
                self.query_one("#path-error", Label).update(f"Error saving file: {e}")
                return  # stay on screen so user can fix filename / retry

            await asyncio.sleep(2)
            self.app.exit()

        elif event.button.id == "cancel":
            self.app.pop_screen()


    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "filename-input":
            return

        filename = event.value.strip()
        save_btn = self.query_one("#save", Button)
        error = self.query_one("#path-error", Label)

        if not filename:
            save_btn.disabled = True
            error.update("Filename cannot be empty.")
            return

        if filename.count('.') > 1:
            save_btn.disabled = True
            error.update("Filename cannot contain multiple dots.")
            return

        if '.' in filename and not filename.lower().endswith(".json"):
            save_btn.disabled = True
            error.update("Only .json extension is allowed.")
            return

        save_btn.disabled = False
        error.update("")


    def action_request_quit(self) -> None:
        self.app.pop_screen()


class SummaryStep(StepScreen):
    __json: dict
    __summary: str

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        self.__json = self.session.to_dict()
        self.__summary = self.session.get_summary()
        json_log = RichLog(markup=False, classes="summary-box", id="json-log", wrap=True, auto_scroll=False)
        summary_log = RichLog(markup=False, classes="summary-box", id="summary-log", wrap=True, auto_scroll=False)
        json_log.write(json.dumps(self.__json, indent=2))
        summary_log.write(self.__summary)

        yield Horizontal(
            Vertical(
                Static("Generated Test JSON", classes="field-label"),
                json_log,
                classes="summary-container"
            ),
            Vertical(
                Static("Short Summary", classes="field-label"),
                summary_log,
                classes="summary-container"
            ),
            classes="full"
        )

        yield Horizontal(
            Button("Prev", id="prev"),
            Button("Save", id="next"),
            classes="button-row"
        )

        yield Footer()


    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "prev":
            from tui.steps.algorithms import AlgorithmsStep
            self.prev(AlgorithmsStep)
        elif event.button.id == "next":
            self.app.push_screen(SaveScreen(self.__json))


    # TODO:
    def get_help_desc(self):
        return "a","b"
