from pathlib import Path

from faster_whisper import BatchedInferencePipeline, WhisperModel

from .cli_utils import CliUtils


def asr(
    media_path: Path,
    model_size: str = "turbo",
    language: str = "en",
    batch_size: int = 8,
    compute_type: str = "int8",
    device: str = "cuda",
) -> dict:
    input_path_str = str(media_path.resolve())

    CliUtils.print_status("Loading model")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    batched_model = BatchedInferencePipeline(model=model)
    segments, _ = batched_model.transcribe(
        input_path_str,
        batch_size=batch_size,
        language=language,
    )

    CliUtils.print_status("Transcribing")

    result_list: list[dict] = []

    for segment in segments:
        inner_dict: dict = {
            "text": segment.text,
            "start": segment.start,
            "end": segment.end,
        }

        result_list.append(inner_dict)

    # Output in whisperx format for compatibility with other tools
    whisper_x_format = {"segments": result_list, "language": language}

    return whisper_x_format
