# Third-party notices

## CLAP checkpoint converter

`tools/convert_clap_checkpoint.py` contains the key-renaming function and mapping adapted from [Transformers v4.35.0](https://github.com/huggingface/transformers/blob/v4.35.0/src/transformers/models/clap/convert_clap_original_pytorch_to_hf.py).

Copyright 2023 The HuggingFace Inc. team. All rights reserved.

Licensed under [Apache License 2.0](LICENSES/Apache-2.0.txt). The converter was modified to accept the official music checkpoint directly, strip DDP prefixes, validate loaded keys, save Transformers model/processor files, and check text-embedding contrasts. The upstream release has no root NOTICE file.

## Classifier weights and ASAP

`gangqinpu/classifier/model.joblib` was trained on Gangqinpu and ASAP-derived features and is distributed under [CC BY-NC-SA 4.0](LICENSES/CC-BY-NC-SA-4.0.txt). ASAP: Francesco Foscarin, Andrew McLeod, Philippe Rigaux, Florent Jacquemard, and Masahiro Sakai, “ASAP: A Dataset of Aligned Scores and Performances for Piano Transcription,” ISMIR 2020; [source and terms](https://github.com/fosfrancesco/asap-dataset). No ASAP MIDI or audio files are bundled.

Other libraries and the separately downloaded Transkun/CLAP checkpoints retain their own licenses. They are not relicensed by this repository. The external CCMZ converter is not bundled.
