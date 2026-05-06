from __future__ import annotations

import numpy as np
import pytest

from voice_lan_stt.recorder import VadOptions, rms_for_audio


def test_rms_for_audio_normalizes_int16_audio() -> None:
    audio = np.array([[0], [16384], [-16384]], dtype=np.int16)

    assert rms_for_audio(audio) == pytest.approx(0.408248, rel=1e-5)


def test_vad_options_validate_rejects_invalid_values() -> None:
    options = VadOptions(sample_rate=16000, threshold=0)

    with pytest.raises(ValueError, match="threshold"):
        options.validate()
