# Handwritten Mathematical Equation Transcription and Correction

## Part 1: High-Level Solution (02/09/25)

The goal is to develop a model that transcribes handwritten mathematical equations into syntactically and semantically correct LaTeX code. The pipeline involves two stages:

1. **Stage 1: RNN/Transformer Transcription**

Since we are working with sequential data (InkML format), we will experiment with RNN and Transformers for transcribing handwritten equations into raw LaTex. The model will be built from scratch to allow completer control over architecture and training. We believe the main challenge is to how to take advantages of sequential data to extracts relevant features and maps them to a sequence of tokens corresponding to LaTex commands (e.g., `\frac`, `\sqrt`, `^`, etc.)

2. **Stage 2: LLM Correction**

The raw LaTex output from the transcription model is processed by an LLM. The LLM refines the transcription by resolving ambiguities, fixing syntax errors, and ensuring structural consistency.

Example: `\cos(0) = 1` might be incorrectly transcribed but fixed at this stage.


