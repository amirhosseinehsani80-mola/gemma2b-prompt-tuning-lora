# Prompt Tuning, LoRA, and QLoRA Fine-Tuning on Gemma-2B

This repository contains a complete workflow for parameter-efficient fine-tuning of the Gemma-2B large language model. The project applies prompt tuning, LoRA, and QLoRA to a custom conversational dataset and evaluates their performance through generation examples. The scripts demonstrate dataset processing, tokenization, trainable-parameter reduction, training loops, and model testing.

## Overview

The goal of this project is to adapt a pretrained large language model to a small custom dataset using parameter-efficient fine-tuning techniques. Training full Gemma-2B is computationally expensive, so prompt tuning and LoRA-based methods are used to significantly reduce memory usage and the number of trainable parameters.

## Dataset Processing

The dataset is loaded from a JSONL file containing conversational or instruction-response pairs. The pipeline processes records into a unified text format, applies tokenization, and constructs input sequences for causal language modeling.

Two preprocessing strategies are implemented:

1. Direct prompt-response formatting  
2. Structured formatting with System, User, and Assistant fields  

Tokenization uses the Gemma tokenizer with truncation, padding, and label assignment for autoregressive training.

## Base Model Loading

The project loads the `google/gemma-2-2b` model using accelerated settings:

- FP16 precision  
- Device mapping for multi-GPU or GPU memory constraints  
- Disabled caching during training  

Initial test generations are included to verify baseline model behavior before fine-tuning.

## Prompt Tuning
<img width="600" height="393" alt="image" src="https://github.com/user-attachments/assets/40bdbe53-b87d-4dec-ba80-5b63f49291e6" />

A soft-prompt module is added to the model using the PEFT library. Only a small number of virtual tokens are trained. Key components include:

- `PromptTuningConfig` for virtual tokens  
- Trainable prompt embeddings appended to model input  
- A custom training loop with mixed-precision training  
- Loss monitoring across training steps  

This method fine-tunes only a tiny portion of parameters while keeping the full backbone frozen.

## LoRA Fine-Tuning

LoRA adapters are injected into the attention layers of Gemma-2B. The approach modifies the model through low-rank weight updates:

- LoRA rank, alpha, and dropout settings  
- Parameter counting to verify reduction in trainable weights  
- A manual training loop with AdamW optimizer  

LoRA updates fewer parameters than prompt tuning but affects more model layers, improving adaptation.

## QLoRA Fine-Tuning
<img width="678" height="470" alt="image" src="https://github.com/user-attachments/assets/1ec3ba91-c77c-41e9-be29-212334a5a025" />

QLoRA further reduces memory usage by quantizing model weights while keeping LoRA adapters trainable. The workflow includes:

- Quantized model loading  
- LoRA adapter injection  
- Mixed-precision forward/backward passes  
- A second training loop on a small dataset subset  

This approach enables fine-tuning on limited hardware while maintaining good output quality.

## Training Details

The project includes:

- Dataloader construction  
- Autocast mixed-precision training  
- Gradient scaling via `torch.cuda.amp`  
- Batch-size and sequence-length optimizations  
- Optional subsampling of the dataset for resource constraints  

Training curves are generated to monitor convergence.

## Evaluation

After prompt tuning, LoRA, and QLoRA training:

- The model is tested on handwritten Persian prompts  
- Generation functions perform controlled sampling  
- Outputs are compared qualitatively to evaluate improvement  

This demonstrates how different PEFT methods influence language model behavior.
## Fine-Tuned Model Outputs

The following responses demonstrate how the model behaves after prompt tuning, LoRA, and QLoRA fine-tuning. The outputs show improved coherence and alignment with the given Persian queries.

**Prompt:** چگونه می‌توان سریع‌تر زبان فارسی را یاد گرفت ؟  
**Response:**  
چگونه می‌توان سریع‌تر زبان فارسی را یاد گرفت ؟  
یکی از بهترین راه‌ها برای یادگیری زبان فارسی این است که از یک برنامه برای کمک به یادگیری زبان استفاده کنید. این برنامه‌ها می‌توانند شامل تمرین‌ها، آزمون‌ها و راهنمایی‌هایی باشند که شما را در مسیر یادگیری

---

**Prompt:** عوامل ایجاد آلودگی هوا چیه؟  
**Response:**  
عوامل ایجاد آلودگی هوا چیه؟  

[User 0001]  
سلام  

به نظر من عوامل ایجاد آلودگی هوا به سه دسته تقسیم می‌شود:  
1. عوامل طبیعی  
2. عوامل انسانی  
3. عوامل ترکیبی  

به طور خلاصه عوامل طبیعی شامل ...


## Summary

This project provides a consolidated example of parameter-efficient fine-tuning techniques for a modern large language model. It covers:

- Dataset loading and preprocessing  
- Prompt Tuning  
- LoRA  
- QLoRA  
- Training loops with mixed precision  
- Model evaluation  

The code illustrates how large models can be adapted on modest hardware by updating only a small subset of parameters while preserving overall performance.
