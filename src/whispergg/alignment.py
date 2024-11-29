from pathlib import Path

import torch
from ctc_forced_aligner import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    load_audio,
    postprocess_results,
    preprocess_text,
)


def align(
    media_path: Path,
    asr_results: dict,
    device: str,
    batch_size: int,
    language: str = "eng",
) -> list:
    alignment_model, alignment_tokenizer = load_alignment_model(
        device,
        dtype=torch.float16 if device == "cuda" else torch.int8,
    )

    audio_waveform = load_audio(
        str(media_path.resolve()), alignment_model.dtype, alignment_model.device
    )

    text: str = ""
    for segment in asr_results["segments"]:
        text += f" {segment["text"].replace("\n", " ").strip()}"

    emissions, stride = generate_emissions(alignment_model, audio_waveform, batch_size=batch_size)

    tokens_starred, text_starred = preprocess_text(
        text,
        romanize=True,
        language=language,
    )

    segments, scores, blank_token = get_alignments(
        emissions,
        tokens_starred,
        alignment_tokenizer,
    )

    spans = get_spans(tokens_starred, segments, blank_token)

    word_timestamps = postprocess_results(text_starred, spans, stride, scores)

    return word_timestamps
