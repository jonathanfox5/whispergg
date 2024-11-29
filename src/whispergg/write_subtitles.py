from pathlib import Path

from .cli_utils import CliUtils
from .whisperx.subtitles_processor import SubtitlesProcessor


def write_subtitles_split(
    stage2_results: list[dict],
    output_directory: Path,
    subtitle_format: str,
    max_line_length: int,
    sub_split_threshold: int,
) -> list:
    is_vtt = False
    if subtitle_format.lower() == "vtt":
        is_vtt = True

    output_paths: list[Path] = []
    for result_dict in stage2_results:
        media_path: Path = result_dict["path"]
        output_path = output_directory / f"{media_path.stem}.{subtitle_format}"

        if output_path.exists():
            output_path.unlink(missing_ok=True)

        CliUtils.print_plain(f"Writing split (normal) subtitles: {output_path}")

        subtitles_proccessor = SubtitlesProcessor(
            result_dict["stage2_output"]["segments"],
            result_dict["language"],
            max_line_length=max_line_length,
            min_char_length_splitter=sub_split_threshold,
            is_vtt=is_vtt,
        )

        subtitles_proccessor.save(output_path, advanced_splitting=True)

        output_paths.append(output_path)

    return output_paths
