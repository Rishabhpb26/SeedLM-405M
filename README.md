# SeedLM-405M : A Large Language Model Which Was Trained On A Consumer GPU

SeedLM-405M is an experimental large language model built to demonstrate what a single independent developer can achieve with limited hardware.

The main goal of this project is to show small developers and researchers that building and training an LLM from scratch is possible even on consumer hardware. Many people believe that only large companies with massive GPU clusters can create language models. This project challenges that assumption.

SeedLM-405M is a **405M parameter transformer-based language model** trained entirely from scratch.

SeedLM-405M was trained on **825 million tokens**. While that number may sound small compared to modern large-scale models, it represents an enormous amount of text. To put it into perspective, reading that much data would take a human roughly **18 years**.

The model was trained entirely on a **single RTX 3060 (12GB VRAM)**. Training reached this point after roughly **60 hours of GPU compute**, which was likely near the practical limit of the hardware. During the final stages of training, the GPU even began showing signs of stress with unusual behavior, reminding us how demanding large-scale model training can be on consumer systems.

This project exists to answer a simple question:

**What is the maximum a small developer can realistically achieve when building an LLM from scratch using consumer hardware?**

SeedLM-405M is the result of that experiment.

The repository includes the inference code, model configuration, tokenizer, and trained weights so others can explore how a small-scale independently trained LLM behaves and what can realistically be achieved outside of large research labs.

## Fine-Tuning and Chatbot Potential

SeedLM-405M is **fine-tunable**, which means developers can further train the model for specific tasks.

By applying **Supervised Fine-Tuning (SFT)** on instruction or conversation datasets, SeedLM-405M can be adapted into a **custom chatbot or assistant**. This allows developers to experiment with instruction following, dialogue systems, and task-specific behavior.

Fine-tuning the model can help reveal both:

* the **true potential** of the base model
* the **practical limitations** of a 405M parameter model trained on consumer hardware

This makes SeedLM-405M a useful experimental platform for developers who want to understand how smaller independently trained language models behave when adapted for real-world tasks.

If you're an independent developer curious about training or fine-tuning your own language model, this project is meant to inspire you

Credits:
SeedLM-405M was developed and trained by Rishabh Modi.
