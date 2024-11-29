from pathlib import Path

import typer
from typing_extensions import Annotated

from .cli_utils import CliUtils

app = typer.Typer(
    no_args_is_help=True,
    rich_markup_mode="rich",
    add_completion=False,
    pretty_exceptions_enable=False,
)


@app.command(
    no_args_is_help=True,
    rich_help_panel="Primary Functions",
    help="Blah",
)
def whispergg(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            help="blah",
            show_default=False,
            callback=CliUtils.validate_path_exists,
            rich_help_panel="Required",
        ),
    ],
):
    from .transcriber import transcribe

    transcribe(input_path, compute_type="float16")


if __name__ == "__main__":
    app()
