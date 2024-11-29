from pathlib import Path

from .alignment import align
from .asr import asr
from .cli_utils import CliUtils


def transcribe(
    media_path: Path,
    model_size: str = "turbo",
    language: str = "en",
    batch_size: int = 8,
    compute_type: str = "int8",
    device: str = "cuda",
):
    CliUtils.print_status("ASR")
    asr_results = asr(
        media_path=media_path,
        model_size=model_size,
        language=language,
        batch_size=batch_size,
        compute_type=compute_type,
        device=device,
    )

    test_path = Path("test") / Path("test.txt")
    with test_path.open("w") as f:
        for segement in asr_results["segments"]:
            f.write(f"{segement["text"]}\n")

    CliUtils.print_status("Alignment")
    align_results = align(
        media_path=media_path, asr_results=asr_results, device=device, batch_size=batch_size
    )

    CliUtils.print_status("Results")
    combined_results = asr_results
    combined_results["words"] = align_results

    from .utils import write_to_json

    write_to_json(combined_results, Path("test") / "combined_results.json")
