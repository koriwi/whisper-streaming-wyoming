"""Wyoming AsyncEventHandler for streaming Whisper transcription."""

import asyncio
import logging
from typing import Optional

from wyoming.asr import Transcribe, Transcript, TranscriptChunk
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

from .engine import StreamingEngine

logger = logging.getLogger(__name__)


class WhisperEventHandler(AsyncEventHandler):
    """Handles one Wyoming client connection, streaming partial transcripts."""

    def __init__(
        self,
        wyoming_info: Info,
        engine: StreamingEngine,
        min_chunk_size: float = 1.2,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._info = wyoming_info
        self._engine = engine
        self._min_chunk_size = min_chunk_size

        # State for the current utterance
        self._language: Optional[str] = None
        self._inference_task: Optional[asyncio.Task] = None
        self._partial_texts: list[str] = []

        # Normalise incoming audio to 16 kHz / 16-bit / mono
        self._converter = AudioChunkConverter(rate=16000, width=2, channels=1)

    # ------------------------------------------------------------------
    # Wyoming protocol event handlers
    # ------------------------------------------------------------------

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self._info.event())
            return True

        if Transcribe.is_type(event.type):
            transcribe = Transcribe.from_event(event)
            self._language = transcribe.language
            logger.debug("Transcribe requested, language=%s", self._language)
            return True

        if AudioStart.is_type(event.type):
            await self._on_audio_start()
            return True

        if AudioChunk.is_type(event.type):
            await self._on_audio_chunk(AudioChunk.from_event(event))
            return True

        if AudioStop.is_type(event.type):
            await self._on_audio_stop()
            return False  # Signal end of session

        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _on_audio_start(self) -> None:
        logger.debug("AudioStart received — resetting engine")
        self._partial_texts = []
        await asyncio.to_thread(self._engine.reset)

        # Start background inference loop
        self._inference_task = asyncio.create_task(self._inference_loop())

    async def _on_audio_chunk(self, chunk: AudioChunk) -> None:
        converted = self._converter.convert(chunk)
        await asyncio.to_thread(
            self._engine.insert_audio,
            converted.audio,
            converted.rate,
            converted.width,
            converted.channels,
        )

    async def _on_audio_stop(self) -> None:
        logger.debug("AudioStop received — finalising")

        # Stop the background inference loop
        if self._inference_task is not None:
            self._inference_task.cancel()
            try:
                await self._inference_task
            except asyncio.CancelledError:
                pass
            self._inference_task = None

        # Final inference
        final_text = await asyncio.to_thread(self._engine.finish)
        logger.info("Final transcript: %r", final_text)

        await self.write_event(Transcript(text=final_text).event())

    async def _inference_loop(self) -> None:
        """Periodically runs streaming inference and emits TranscriptChunk events."""
        try:
            while True:
                await asyncio.sleep(self._min_chunk_size)

                partial = await asyncio.to_thread(self._engine.process_iter)
                if partial:
                    logger.debug("Partial: %r", partial)
                    self._partial_texts.append(partial)
                    await self.write_event(TranscriptChunk(text=partial).event())
        except asyncio.CancelledError:
            pass
