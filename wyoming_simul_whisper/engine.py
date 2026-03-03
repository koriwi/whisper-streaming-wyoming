"""Thread-safe wrapper around SimulStreaming's PaddedAlignAttWhisper."""

import logging
import threading
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class StreamingEngine:
    """Thread-safe streaming ASR engine wrapping PaddedAlignAttWhisper.

    All public methods are protected by a threading.Lock because the model
    holds mutable GPU state (KV cache, attention hooks) that is not
    concurrent-safe.
    """

    def __init__(
        self,
        model_path: str,
        language: str = "en",
        task: str = "transcribe",
        frame_threshold: int = 4,
        beam_size: int = 1,
        audio_max_len: float = 30.0,
        audio_min_len: float = 1.0,
        cif_ckpt_path: str = "",
        never_fire: bool = False,
        logdir: Optional[str] = None,
    ) -> None:
        # Import here so startup errors surface clearly
        from simulstreaming.whisper.simul_whisper.config import AlignAttConfig
        from simulstreaming.whisper.simul_whisper.simul_whisper import PaddedAlignAttWhisper

        self._lock = threading.Lock()

        cfg = AlignAttConfig(
            model_path=model_path,
            language=language,
            task=task,
            frame_threshold=frame_threshold,
            beam_size=beam_size,
            decoder_type="beam" if beam_size > 1 else "greedy",
            audio_max_len=audio_max_len,
            audio_min_len=audio_min_len,
            cif_ckpt_path=cif_ckpt_path,
            never_fire=never_fire,
            logdir=logdir,
        )

        logger.info("Loading model from %s …", model_path)
        self._model = PaddedAlignAttWhisper(cfg)
        logger.info("Model loaded.")

        # Queued raw float32 audio chunks (16 kHz, mono)
        self._audio_queue: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all audio buffers and model state for a new utterance."""
        with self._lock:
            self._audio_queue.clear()
            self._model.segments = []
            self._model.refresh_segment(complete=True)

    def insert_audio(
        self,
        pcm_bytes: bytes,
        rate: int,
        width: int,
        channels: int,
    ) -> None:
        """Accept raw PCM audio and queue it for the next inference pass.

        Wyoming delivers 16 kHz / 16-bit / mono after AudioChunkConverter
        normalises the stream, so conversion is straightforward.

        Args:
            pcm_bytes: Raw PCM bytes.
            rate: Sample rate (expected 16000).
            width: Bytes per sample (expected 2 for int16).
            channels: Channel count (expected 1).
        """
        if not pcm_bytes:
            return

        # Convert int16 PCM → float32 in [-1, 1]
        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        if channels > 1:
            audio_int16 = audio_int16.reshape(-1, channels).mean(axis=1).astype(np.int16)
        audio_f32 = audio_int16.astype(np.float32) / 32768.0

        with self._lock:
            self._audio_queue.append(audio_f32)

    def process_iter(self) -> str:
        """Run one streaming inference step and return any new partial text.

        Intended to be called periodically (e.g. every 1.2 s) from a
        background thread/task while audio is still being received.

        Returns:
            Newly recognised text since the last call, or empty string if
            there is nothing new yet.
        """
        with self._lock:
            if not self._audio_queue:
                return ""

            # Drain queue and feed to model buffer
            combined = np.concatenate(self._audio_queue)
            self._audio_queue.clear()
            segment = torch.from_numpy(combined)
            self._model.insert_audio(segment)

            new_tokens, _ = self._model.infer(is_last=False)
            if not new_tokens:
                return ""
            return self._model.tokenizer.decode(new_tokens)

    def finish(self) -> str:
        """Flush remaining audio and return the final complete transcript.

        Also resets the model state so the engine is ready for the next
        utterance.

        Returns:
            The complete transcript for the current utterance.
        """
        with self._lock:
            # Feed any remaining queued audio
            if self._audio_queue:
                combined = np.concatenate(self._audio_queue)
                self._audio_queue.clear()
                segment = torch.from_numpy(combined)
                self._model.insert_audio(segment)

            # Final inference pass
            final_tokens, _ = self._model.infer(is_last=True)

            # Collect all committed tokens from the model's token list
            # tokens[0] is the initial/forced tokens; tokens[1:] are hypotheses
            all_tokens: list[int] = []
            for t in self._model.tokens[1:]:
                all_tokens.extend(t[0].tolist())
            if final_tokens:
                # final_tokens are already appended to self._model.tokens by infer()
                pass

            # Decode everything committed so far (from model state)
            committed = [tok for tlist in self._model.tokens[1:] for tok in tlist[0].tolist()]
            text = self._model.tokenizer.decode(committed) if committed else ""

            # Reset for next utterance
            self._model.segments = []
            self._model.refresh_segment(complete=True)

            return text.strip()

    def warmup(self) -> None:
        """Run a short dummy inference to initialise GPU buffers."""
        logger.info("Warming up model …")
        dummy = torch.zeros(16000, dtype=torch.float32)
        with self._lock:
            self._model.insert_audio(dummy)
            try:
                self._model.infer(is_last=True)
            except Exception:
                pass
            self._model.segments = []
            self._model.refresh_segment(complete=True)
        logger.info("Warmup complete.")
