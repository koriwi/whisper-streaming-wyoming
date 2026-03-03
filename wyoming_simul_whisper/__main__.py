"""CLI entry point for the Wyoming SimulStreaming Whisper server."""

import argparse
import asyncio
import logging
from functools import partial

from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer

try:
    from . import __version__
    from .engine import StreamingEngine
    from .handler import WhisperEventHandler
except ImportError:
    # Running as a script: python __main__.py
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from wyoming_simul_whisper import __version__
    from wyoming_simul_whisper.engine import StreamingEngine
    from wyoming_simul_whisper.handler import WhisperEventHandler


def _build_info(args: argparse.Namespace) -> Info:
    return Info(
        asr=[
            AsrProgram(
                name="simul-whisper",
                description="Streaming Whisper via SimulStreaming / AlignAtt",
                attribution=Attribution(
                    name="ufal/SimulStreaming",
                    url="https://github.com/ufal/SimulStreaming",
                ),
                installed=True,
                version=__version__,

                models=[
                    AsrModel(
                        name=args.model,
                        description=f"Whisper {args.model}",
                        attribution=Attribution(
                            name="OpenAI Whisper",
                            url="https://github.com/openai/whisper",
                        ),
                        installed=True,
                        version=__version__,
                        languages=[args.language] if args.language != "auto" else [],
                    )
                ],
            )
        ]
    )


async def _main(args: argparse.Namespace) -> None:
    engine = StreamingEngine(
        model_path=args.model,
        language=args.language,
        task=args.task,
        frame_threshold=args.frame_threshold,
        beam_size=args.beams,
        audio_max_len=args.audio_max_len,
        audio_min_len=args.audio_min_len,
        cif_ckpt_path=args.cif_ckpt_path,
        never_fire=args.never_fire,
        logdir=args.logdir,
    )

    if args.warmup:
        engine.warmup()

    wyoming_info = _build_info(args)

    server = AsyncServer.from_uri(args.uri)
    logging.info("Listening on %s", args.uri)

    await server.run(
        partial(
            WhisperEventHandler,
            wyoming_info,
            engine,
            args.min_chunk_size,
        )
    )


def run() -> None:
    parser = argparse.ArgumentParser(
        description="Wyoming SimulStreaming Whisper ASR server",
    )
    parser.add_argument(
        "--uri",
        required=True,
        help="Wyoming server URI, e.g. tcp://0.0.0.0:10250 or unix:///tmp/whisper.sock",
    )
    parser.add_argument(
        "--model",
        default="large-v3",
        help="Whisper model name or path to .pt file (default: large-v3)",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language code, or 'auto' for automatic detection (default: en)",
    )
    parser.add_argument(
        "--task",
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Task: transcribe or translate (default: transcribe)",
    )
    parser.add_argument(
        "--min-chunk-size",
        type=float,
        default=1.2,
        dest="min_chunk_size",
        help="Seconds between streaming inference runs (default: 1.2)",
    )
    parser.add_argument(
        "--frame-threshold",
        type=int,
        default=4,
        dest="frame_threshold",
        help="AlignAtt frame threshold for emission (default: 4)",
    )
    parser.add_argument(
        "--beams",
        type=int,
        default=1,
        help="Beam search width; 1 = greedy (default: 1)",
    )
    parser.add_argument(
        "--audio-max-len",
        type=float,
        default=30.0,
        dest="audio_max_len",
        help="Maximum audio buffer length in seconds (default: 30.0)",
    )
    parser.add_argument(
        "--audio-min-len",
        type=float,
        default=1.0,
        dest="audio_min_len",
        help="Minimum audio length before inference starts (default: 1.0)",
    )
    parser.add_argument(
        "--cif-ckpt-path",
        default="",
        dest="cif_ckpt_path",
        help="Path to CIF checkpoint for end-of-word detection (optional)",
    )
    parser.add_argument(
        "--never-fire",
        action="store_true",
        dest="never_fire",
        help="Disable CIF end-of-word boundary firing",
    )
    parser.add_argument(
        "--logdir",
        default=None,
        help="Directory for per-segment debug audio/hypothesis logs (optional)",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        default=True,
        help="Run a warmup inference after loading the model (default: True)",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_false",
        dest="warmup",
        help="Skip warmup inference",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    asyncio.run(_main(args))


if __name__ == "__main__":
    run()
