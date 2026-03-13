# **Pre-Course Primer: LLM Internals & Performance Engineering**

This document provides foundational knowledge, architectural reasoning, and critical trade-offs aligned with your upcoming Module 2 syllabus. It distinguishes established mathematical facts from empirical architectural choices.

## **1\. Linear Classification & Optimization (Prep for Mar 18 & 25\)**

Before understanding a 100-billion parameter model, you must understand the optimization of a single matrix.

### **Linear Classification Models**

* **Mechanism:** A classifier maps an input vector ![][image1] to output logits ![][image2] via a weight matrix ![][image3] and bias ![][image4]: ![][image5].  
* **The LLM Head:** The final layer of an LLM (the "head") is essentially a massive linear classifier. It takes the final hidden state of the transformer and projects it across the entire vocabulary space to output logits. These logits are passed through a Softmax function to yield a probability distribution over the vocabulary.  
* **Fact:** Softmax normalizes logits into probabilities, but it is highly sensitive to large values (exponential scaling).

### **Gradient Descent & SGD Modifications**

* **Stochastic Gradient Descent (SGD):** Instead of calculating the gradient over the entire dataset (which is computationally impossible for LLMs), SGD calculates the gradient on a small "mini-batch."  
* **The Problem with Vanilla SGD:** It struggles with "ravines" in the loss landscape, oscillating across slopes rather than moving directly to the local minimum.  
* **Adam (Adaptive Moment Estimation):** The industry standard for training LLMs.  
  * *Mechanism:* It maintains two moving averages for each parameter: the first moment (mean of gradients, like momentum) and the second moment (uncentered variance of gradients).  
  * *Architectural Reasoning:* By dividing the learning rate by the square root of the second moment, Adam effectively gives infrequent parameters larger updates and frequent parameters smaller updates. This is critical for NLP, where word frequencies follow a long-tail distribution (Zipf's Law).

## **2\. Deep Learning Primitives & Sequences (Prep for Apr 1 & 8\)**

### **Fully Connected Neural Networks (FCNNs/MLPs)**

* **Mechanism:** FCNNs introduce non-linearity. A linear transformation followed by a non-linear activation function (like ReLU or GeLU).  
* **Fact (Universal Approximation Theorem):** A feed-forward network with a single hidden layer containing a finite number of neurons can approximate continuous functions on compact subsets of ![][image6], given a suitable activation function.  
* **Role in LLMs:** FCNNs act as the "memory" or "knowledge retrieval" blocks within transformer layers, processing the context aggregated by the attention mechanism.

### **Tokenization**

* **Mechanism:** LLMs do not read text; they read arrays of integers. Tokenization maps sub-words to these integers (e.g., Byte Pair Encoding (BPE)).  
* **Trade-off:** Vocabulary Size vs. Sequence Length.  
  * A smaller vocabulary (e.g., character-level) means longer sequences, straining the context window and compute.  
  * A larger vocabulary means shorter sequences, but drastically inflates the size of the embedding matrix and the final output head, requiring more memory.

### **The Problem with RNNs**

* **Mechanism:** Recurrent Neural Networks (RNNs) process tokens sequentially, maintaining a "hidden state" that updates with each new token.  
* **Architectural Flaw:** Sequential processing prevents parallelization across GPUs. Furthermore, through backpropagation through time (BPTT), gradients tend to vanish or explode, making it mathematically difficult for the network to retain information from token 1 by the time it reaches token 1000\.

## **3\. The Transformer Architecture (Prep for Apr 15 & 22\)**

The transformer solved the RNN's sequential bottleneck by processing all tokens simultaneously using Attention.

### **Scaled Dot-Product Attention**

This is the core mathematical engine of modern AI.

* ![][image7]**Internal Mechanism:**  
  1. **Q (Query), K (Key), V (Value):** For every token, linear projections create a Query vector (what I am looking for), a Key vector (what I contain), and a Value vector (my actual content).  
  2. ![][image8] **(Dot Product):** Taking the dot product of every Query with every Key computes a similarity score. High dot product \= high relevance.  
  3. **Scaling (![][image9]):** *Architectural reasoning:* As the dimension ![][image10] grows, the dot products grow large in magnitude. This pushes the softmax function into regions with extremely small gradients (saturation). Dividing by ![][image9] stabilizes the gradients.  
  4. **Multiplication with V:** The softmax scores are used as weights to sum up the Value vectors. The output for a token is a weighted average of all other relevant tokens' values.

### **Decoder-Only Architecture (The Standard LLM)**

* Most modern LLMs (GPT, Llama, Mistral) drop the encoder from the original Transformer. They are "autoregressive decoders."  
* **Masked Self-Attention:** They use a lower-triangular mask on the ![][image8] matrix to ensure a token can only attend to previous tokens, never future tokens (which would be cheating during training).

## **4\. Modern Architecture & Performance Engineering (Prep for Apr 29 & May 6\)**

This is where the theoretical meets the physical constraints of GPU memory and bandwidth.

### **Modern Components (Working Theories & Empirical Standards)**

* **RoPE (Rotary Position Embeddings):** Instead of adding absolute position vectors (like the original transformer), RoPE encodes relative position by rotating the Query and Key representations in the complex plane. *Benefit:* Better extrapolation to sequence lengths longer than seen in training.  
* **SwiGLU Activation:** Replaces standard ReLU in the FCNN layers. It is empirically proven to yield better performance per parameter, though the strict mathematical reason *why* it outperforms others is still a working theory in the literature.

### **Practical LLM Inference: The Memory Wall**

LLM inference is generally **memory bandwidth bound**, not compute bound. Generating a token requires loading the entire model's weights from HBM (High Bandwidth Memory) to the compute cores.

* **The KV Cache:** To avoid recomputing the Keys and Values for previous tokens during generation, they are cached in GPU memory.  
* **GQA (Grouped Query Attention):** *Trade-off resolution.* Multi-Head Attention (MHA) creates unique K and V heads for every Q head, leading to massive KV caches that exhaust VRAM. Multi-Query Attention (MQA) uses 1 K and V head for all Q heads (degrades quality). GQA groups multiple Q heads to share a single K/V head, striking the optimal balance between inference memory reduction and model capability.

### **Fine-Tuning and LoRA (Low-Rank Adaptation)**

* **The Problem:** Full fine-tuning of a 70B parameter model requires updating 70B weights, necessitating massive clusters of GPUs simply to hold the optimizer states (e.g., Adam's momentum variables take 2-3x the memory of the model weights themselves).  
* **LoRA Mechanism:** LoRA freezes the pre-trained weight matrix ![][image11]. Instead of updating ![][image11], it learns a smaller weight update ![][image12].  
* **The Math:** ![][image12] is constrained by low-rank decomposition. It is represented as the product of two smaller matrices, ![][image13] and ![][image14].![][image15]  
  If ![][image11] is ![][image16] (100M parameters), and rank ![][image17], ![][image13] is ![][image18] and ![][image14] is ![][image19] (160k parameters total).  
* **Performance Consideration:** LoRA reduces trainable parameters by up to 10,000x, drastically reducing VRAM requirements and allowing fine-tuning on consumer-grade GPUs, with the trade-off being a highly constrained update space that cannot learn entirely new foundational concepts, but is excellent for style transfer or specific format instruction.

## **5\. The Current Open/API Model Landscape**

Understanding the current ecosystem requires distinguishing between underlying architectures (Dense vs. Mixture-of-Experts) and the intended computational environment (Edge vs. Data Center).

### **DeepSeek (China)**

**Models:** DeepSeek-V3.2, DeepSeek-R1-0528, DeepSeek-V3-0324

* **Introduction:** DeepSeek leads the frontier of hyper-efficient AI. Their models utilize DeepSeek Sparse Attention (DSA) and advanced MoE (Mixture of Experts) architectures.  
* **Strengths:** Market-disrupting efficiency. DeepSeek-V3.2 currently matches or beats proprietary models (like GPT-5.2 and Claude Opus) on production coding and math benchmarks at a fraction of the inference cost. R1-series models excel in verifiable Chain-of-Thought (CoT) reasoning.  
* **Weaknesses:** While highly logical, the "reasoner" variants can struggle with context comprehension and consistency in creative writing or roleplay scenarios. There are also reported alignment and censorship constraints based on their origin.

### **Qwen / Alibaba Cloud (China)**

**Models:** Qwen3 series (Coder, Thinking, VL), Qwen2.5 series

* **Introduction:** Alibaba's flagship open-weight series. The Qwen3 family utilizes an MoE architecture spanning from 0.6B to massive 480B parameter configurations, featuring specialized "Thinking" modes for long-horizon planning.  
* **Strengths:** Exceptional cost-to-performance ratio and native multilingual capabilities (119 languages). Qwen3 MoE configurations maintain very low active parameters per token, making high-end coding and reasoning accessible on moderate hardware.  
* **Weaknesses:** The most glaring empirical weakness is a deficit in general "world knowledge" (pop culture, media, niche history), which leads to severe hallucinations even at low temperatures. It also exhibits high baseline alignment/caution which can hinder complex, unconstrained problem-solving.

### **OpenAI (USA)**

**Models:** gpt-oss-120b, gpt-oss-20b

* **Introduction:** OpenAI's pivot into the open-weights ecosystem. These are MoE models trained with reinforcement learning techniques derived from their internal o-series (o3/o4).  
* **Strengths:** State-of-the-art agentic tool use (web browsing, function calling) and reasoning. They are natively quantized in MXFP4, allowing the 120B model (117B total, 5.1B active parameters) to run efficiently on a single 80GB GPU.  
* **Weaknesses:** While the weights are open, the training data and precise pre-training methodologies remain a proprietary black box. Furthermore, "raw" output formatting requires explicit handling of harmony schemas to function correctly.

### **Moonshot AI (China)**

**Models:** Kimi-K2.5, Kimi-K2-Instruct, Kimi-K2-Thinking

* **Introduction:** Moonshot's native multimodal architecture is built around ultra-long context limits and subagent orchestration (swarms).  
* **Strengths:** A massive 256K native context window handled with high retrieval accuracy. It dominates in complex document extraction, visual OCR tasks, and executing multi-step agent workflows.  
* **Weaknesses:** Extreme verbosity. The model tends to generate significantly more output tokens than competitors (over-engineering code or explanations), which heavily inflates the effective API cost. Additionally, in highly complex agent swarms, subagents can suffer from conceptual drift.

### **Prime Intellect (USA)**

**Models:** INTELLECT-3

* **Introduction:** A 106B-parameter MoE model built by fine-tuning Zhipu's GLM-4.5-Air base model using massive asynchronous Reinforcement Learning (RL) via their custom "prime-rl" stack.  
* **Strengths:** A powerful demonstration of decentralized/open reinforcement learning. It performs exceptionally well on math and science benchmarks for its active weight class (12B active parameters).  
* **Weaknesses:** Dependent on the quality of open RL environments; it occasionally trails absolute frontier models (DeepSeek V3.2) in highly constrained, production-level coding workflows.

### **Zhipu AI / Z.ai (China)**

**Models:** GLM-5, GLM-4.7, GLM-4.5

* **Introduction:** Flagship bilingual foundation models with strong long-context and agent-oriented capabilities.  
* **Strengths:** Very intelligent bare-prompt reasoning. It can solve complex logic problems from scratch without relying heavily on strict prompting frameworks.  
* **Weaknesses:** In fully automated tasks (where the model orchestrates itself without user intervention), it can derail or hallucinate tool calls over long horizons. Official API endpoints have historically faced speed constraints compared to DeepSeek or Qwen.

### **Meta (USA)**

**Models:** Llama-3.3-70B-Instruct, Meta-Llama-3.1-8B, Meta-Llama-Guard-3

* **Introduction:** The anchor of the Western open-source ecosystem, relying on traditional dense transformer architectures rather than MoE.  
* **Strengths:** Extremely robust, predictable performance across chat, coding, and general knowledge. Massive community support guarantees immediate integration into fine-tuning frameworks like LoRA and execution engines like vLLM.  
* **Weaknesses:** Because they are dense models, inference requires calculating *every* parameter for *every* token. A dense 70B model requires significantly more compute throughput than an MoE model with 70B total but only 10B active parameters.

### **Google (USA)**

**Models:** Gemma-3-27b-it, Gemma-2-2b-it, Gemma-2-9b-it

* **Introduction:** Google's lightweight, open-weight models based on the architecture of their flagship Gemini models.  
* **Strengths:** Incredible instruction following and prose generation at small parameter counts. Highly optimized for on-device/edge inference.  
* **Weaknesses:** Empirically, they struggle with deep mathematical and logical reasoning compared to MoE counterparts like Qwen or DeepSeek at similar parameter scales.

### **NVIDIA (USA)**

**Models:** Nemotron-3-Super-120b, Nemotron-Nano-V2-12b, Llama-3\_1-Nemotron-Ultra

* **Introduction:** NVIDIA's enterprise-grade models, frequently fine-tuned variants of existing architectures or highly optimized MoEs built for their proprietary hardware.  
* **Strengths:** Unparalleled stability, robust safety guardrails, and highly deterministic behavior for enterprise data pipelines and RAG (Retrieval-Augmented Generation).  
* **Weaknesses:** Heavily aligned, meaning they may refuse benign requests if they trigger safety heuristics, and they generally lack the creative versatility seen in Hermes or standard Llama iterations.

### **NousResearch (USA)**

**Models:** Hermes-4-405B, Hermes-4-70B

* **Introduction:** Community-driven models heavily fine-tuned on verified Chain-of-Thought traces.  
* **Strengths:** Exceptionally uncensored and steerable. They will execute exactly what is prompted, making them ideal for highly customized programmatic workflows where alignment artifacts cause systemic failures.  
* **Weaknesses:** Performance is bounded by the base model (typically Llama), and the 405B variant requires institutional-grade hardware (multiple 80GB GPUs) simply to load into memory.

## **6\. Decoding Model Nomenclature & Tags**

Model names in provider platforms typically follow a strict taxonomy indicating their architecture, training stage, and specialized modality. Understanding these tags is crucial for calculating VRAM limits and deployment costs.

### **Parameter Counts (e.g., 8B, 70B, 120b)**

* **Meaning:** Indicates the total number of learned parameters (weights and biases) in the network. "B" stands for Billions.  
* **Performance Consideration:** In a dense model (like Llama), a 70B model requires calculating all 70B parameters for every single token generated. At standard 16-bit precision, you need roughly 2GB of VRAM for every 1B parameters just to load the weights (e.g., \~140GB for a 70B model).

### **Active Parameters in MoE (e.g., A3B, A12B, A35B)**

* **Meaning:** "A" denotes **Active Parameters**. This is specific to **Mixture of Experts (MoE)** architectures.  
* **Mechanism:** In an MoE model (like Qwen3-Coder-480B-A35B), the feed-forward networks (FCNNs) in the transformer layers are replaced by multiple specialized "expert" sub-networks. A routing algorithm determines which 1 or 2 experts are needed for a specific token.  
* **Performance Consideration:** This fundamentally decouples model capacity from inference cost. The model has 480B total parameters of knowledge (requiring massive VRAM to store), but it only activates 35B parameters per token. Compute speed (Tokens/sec) behaves like a 35B model, not a 480B model.

### **Tuning States (Base vs. Instruct / it)**

* **Base:** The foundational model resulting strictly from pre-training on massive corpora via next-token prediction. It does not know how to answer questions; it only knows how to complete documents.  
* **Instruct / it:** Instruction-tuned. The base model has undergone Supervised Fine-Tuning (SFT) and alignment (like RLHF or DPO) to understand conversational formats (User/Assistant roles) and respond directly to prompts.

### **Reasoning Modes (Thinking, R1)**

* **Meaning:** Designates models optimized for verifiability and complex logic through **Chain-of-Thought (CoT)** generation.  
* **Mechanism:** Trained via large-scale Reinforcement Learning (RL) to emit thousands of "thinking" tokens before emitting the final answer. The model learns to break down problems, correct its own mistakes, and test hypotheses internally.  
* **Performance Consideration:** Massive increase in **Time-to-First-Token (TTFT)**. Generating 2,000 hidden thinking tokens before outputting the actual response drastically increases the API cost per query compared to a standard instruct model.

### **Modality & Specialty Tags (VL, Vision, Coder)**

* **VL / Vision:** Vision-Language. Indicates the model architecture has been expanded to include a vision encoder (like SigLIP or CLIP). The encoder processes an image into continuous vector representations, which are then projected directly into the LLM's standard embedding space.  
* **Coder:** Indicates continuous pre-training on high-quality code repositories, git commits, and synthetic coding scenarios. These models sacrifice generalized prose ability in exchange for severe exactitude in syntax and tool-calling structures.

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAYCAYAAAAs7gcTAAAArUlEQVR4XmNgGAUDAhQUFDjQhJjR+AwM8vLyx4AKDwHxKiD7PxBPBbJ3AenVQPwAWeFSIHZC4oMUX0Fi/0dW/AfOgfD/Kysri0HZv4G4G1keDqSlpYVRTMIHQG7FqxgouUdOTs4GykZxo4qKCh/QoyvBHCAjAqpgHZLivzDFQPZbGJtBUVFRDyjwA2iyEoiWlZXVgWrQBOKXQHFruGIQgFqViCwG5Psj80cBOgAAeUwwukIdCUcAAAAASUVORK5CYII=>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAYCAYAAADDLGwtAAAAxklEQVR4XmNgGAW0AIzoAihAQUFhpry8/HYg/gbEp5HlgPz/QNwFUiQhJye3Ayr4AyQBUwQU1wLxZWVlTUGSz9B030Pib0bWCAZAk1ehC0I1fkMWgwnCFaqoqLBDxWqQ1cEUroXxge6rA4kZGxuzIquDKdyDxP+FbAMcAAUPgyQUFRX1YIqA+B26OjBQUlKSA0pmAT0mAFWYgqIAKgi3Bsp/gKSEgUFUVJQHaqUZiA/0hA3QxFsoimAAKDkdZioQH0GXH2AAANqVQZ1x0ZkgAAAAAElFTkSuQmCC>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABUAAAAYCAYAAAAVibZIAAABPElEQVR4Xu2SvUrEQBSFA6KFaJkmRX4gEEmnAd/CWgt7ayvFJ7BetPQZBBsfwEJQwd40oqigrLAqa6Xrd9fMMl5nYmWXA4fJPV/unclPEHT6FyVJMsQPuI7j+Ib1zWIDfJ2m6RXrLe4bVpblDPUjvsP3+NWwsQgO8Qhf/ADf7EAYG1YO1mvYumYBJ1lthp472JownYvIP3U2EXCxGfrsYJKPqqqa1owNN3Q2UVEU86bZzqn3sixblpwBCzaj3rZrpzxDB4bx3lYUe7drp/RQ+eLmkZuTblr3noRhOGdqr+yheZ6HXB8rti/XURTNssGZYa2yh7I+Odj4z2Ad2qxVZiiuKaccrI8v5aQ2a5UZyqMd+Rg+1axVNHxIo85F5C8+1iqaal+jfBjYjs7/FI27NC7pXMQ/uqWzTp1+6wt3dWy3XxoqBgAAAABJRU5ErkJggg==>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAZCAYAAAAMhW+1AAAAsUlEQVR4XmNgGCRATk7upLy8/H8oPoYuDwYKCgoWIAVAOgBdDgyAkq0gBejicACU/ElIAcj+6+jiYKClpcUGtT8SSE+BKu6CKwBKVEIF/0tJSXGBxIDsj0B8FawAykGxH6ipEy4G1f0MWQGQvwWsAEgoghgyMjK6aArAVoIYUejGAwEjVMEJkF0C6AqA/HUoYiCOoqKiPIgtKyurA+ID40cLrgAWDlD8QkVFhR0uOTIAAD1NPGQjkOK2AAAAAElFTkSuQmCC>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGQAAAAYCAYAAAAMAljuAAADiElEQVR4Xu1YS2hTQRStih9Q0U1IhXxePhKIoGBQNyL1v3DjTgqiLgRREFFXQhF0IeKniBSkKxfiFxWxCwVdqFWI1I0i4rcWxdqqKKJW0UU9N7mT3Ny8lCSvfUicA4c399w7M3fmznsvL01NFhYWFhYW/ymi0egQ+B7sBfvBb8L3AXwFvohEIm9w/WR8juPcgf0RfAcOgpuNb6zBcw4Tta8hgM09T4vDps/RPrPwWCy2RPvQb2elfmMNzLmikQuyttLixEkc5+L7qrVqEY/HI+i/X+vVAn0fg6e03hCgE+5WEBRqnSkIGJM+3DHz4D8mtVqAOeMeCzKM+R2ta5Sdon8NwWBwqtbS6fQkWiCdWqlzId7SFRu4UvukXStCodBsrwWhayaTmUj5lzhRqE4EXIvmX5A90seLOiQ1P4CcDmLel+AA2IcTvYrbx7G5G3Q857lMati0JGJvkw/jbTU62nvAFhFaM5BPCmPs03o1QL9WzvcH2A5mwZ/G2Yykr1Mb4i8KNB2hp8kOh8MLjKYB/2UevGrqMTRowxF32NiyH7efFqOLMVjLFmE/oCvGOsl92oUvv3gP8FIQzP9M7wPZyPUcNfqlCPYKu0t39ANyTrqlOa/d7BvUjybWKeYItVOp1HS0s6y3kQ+bd5VsLPq+22OvVngsCOVaVpASDYNfqBA0JDW/gflv6LzcwLl2c7vwLYJ1rWbfANuLjK9aoG+HC0+DPS56h+4vgUIGKR8cjPVS5xyL69RCMpmczFpbIcgF9EJCbKAW6jFGgs6rEjiOPvAeYtOnGB2P24QZA+ySfbyg3jvE4W8fqaE420nD9URB5IQviaC9pNEjoxDkAsSsAY/WQj2GBmLOOvwSphzAi8o/S9qsURzxlnKNF77Xylc36i0I9vUA5SI12H+0ZhZ0U9i/y4J8ABbZzLl8J5va0FqNH/aZYnQR3Mc135F89aLegiCP+TIXjNHC+ZX8QqTAbnJgorlRLgb4uSTIH+RONBJ1cH2CE3UF1y/0IUa/BiF36g4EzneX1gnkK3kcjALqLQghmv+8yCKnHZzbch2TA/8dsA0TzeQF+vaHmwQ9JpHDRjQnkI28ZsDeVPYBJYBc72nNAH0fac0rvBSEkEgkwsh5qdZz4M0v3EZs94kQC3fkDsyoIhAITKMCoOILycbtsxiVf67jLHwEPV/NXQLe1X4LCwsLCwsX/AVdg1Wb7eejOwAAAABJRU5ErkJggg==>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABcAAAAYCAYAAAARfGZ1AAABdUlEQVR4Xu2TzysEcRjGBytOrjiY2Z2dHDnsjXJYJ0UpF+Wi9kB+puQgBwdX5W9QjlylZPfoKjk5uSglJbWK1fq8er/j3SkxDkrtU0/P+z7vj/nud2Y9r4k/RxAEr7D+BZ/gAexNzgmy2eyki+kZs7UGUDwUZWDa9/1hjXvw79A19DnRf4rfKYegf1S9uu2JQaGiuhBFUYfzWTBveuIH0NNFvgUvTP3b5YtIxvm5XG4EDmht3/maV+GUxBxilvjB1mOY5UtIq/FL7pcQ3zhf8/ikxDVYhDO2xxUrotzfste4/FpUXh7ccz7IJJZ/xOjVZ4vCLocT5JvwHr7JIN54YsQrFArtNuf6Bm0ewy4Pw7CfU26r3/CV/AosKYu6a5FPUl6S1qq2NzWSy9V7VB2CO6Y9HRg+E9XlbWq34N9q/Zw77Xb9qcDwsShXMWf/RDxsg1pRe2rO/xEYKMEjlpzAVeJdeJnP5/tMzwtcp77irq+JJv4R3gFcPGWfp1ruqwAAAABJRU5ErkJggg==>

[image7]: <data:image/png;base64,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>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAYCAYAAACBbx+6AAACS0lEQVR4Xu2WQUgUcRTG10MSeOkQDAzszO56W8JDC+khoegigmc9dLHwsHgIDKFjF7sIXWQzu0W3QLxJqTfBS51EokBRWcsCCwRJg8i+5743vH3+Z0zn0Bz2B4+Z//u+/9tvd3ZmN5drcX7CMHyG2oupOmrU7vmvINBxuVxu52UbrZU2FgTBTVlngkKh0CPnCPdAB4bWJ+eJYNM8bUR9Q+3QOYaNWJ9QKpUCeA5QX1Gf+bhXLBZviCefz3ei9wW1gVonXc8g0PuB+qjW97Tugi7JL4SbtgKFRi3ZvgZ6jXy+71+1GsEzKFS/1QjS8akO2r4TmBd54G2rERg0RDrezHOrCbw/uqQC9l5x9TXwdJ/liUCIuxzmjdWESqVyKS6Q4NJ59pbuuYDvLXx/bN+J64VcJPnwfe1ifVd6OF/BJ/dY2WLhvVO2fwoMnGHzmtU09OhJCoz+HGlyc+L8N2rf+izw9CNDlWdP0tp6mpAQqOtW06ihcYEjDcfDJG8q/nUwPJvkQ/CXViNkjgr9jv2vrTcV5wgc60N/nPWa6Z/soRtW91ORFETAQ/8aeXD0rUaEjR+NY9x4nunP8vwV3U9F2Lg5osAI1csv8lB56NLekrUl6U0naRcCQS7z0A+0xnGZ+6s5/jOC+tm0SeF5XkdSKPSPSMPT44nVLgyGDfCLrqGeUo9uFu7dt34B2raE5fqOeqH0faPX9f7UIOQwagGDX1Hh/JHSqtqbORB4gj8V+rf2CTVuPZkCX5M7+nJaPZOEjcfSe1TRai1aZIC/FS7lP4Y5bkcAAAAASUVORK5CYII=>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACMAAAAWCAYAAABKbiVHAAABkElEQVR4Xu2TPUvEQBCGDwvxoxUDIZ+k0TSCYGVzjVjYnt0J/gCxtLYVsba0t7eOv8BSG49DsLFS/4D6TpiVYdwke7nIgfjAkp13ZydvZje93l8ijuOiq6FrTwQKfGptZiRJcqq1meBiBJ0bUfcwHvVap7geEfI+YPxQ651BXXHpDOFqujWuL4iiaN81txVNxbH+zPfkCOOhKX8qqorneT4v1zC/Z1PW/KmpK8wvfhPSHGk4qisjBEGwiHiHuibyfkIXsu5L6i6tbR9eeqk1AjWuod9o/RssFmymsBUgqnTCZsamGT0Mwy2tW6FkGOtrvaorWZat8otvpc7aiOa+7y9JXcz3zNxKbOlOlRED5eNYjk2M/DXWDihGJzLO2zS1sTbA/C5N0w2zzwp/VSFjua5B4RPes47xinibYpg6w3Ns8jB/x3hBp1bk/lq4cGmgqSsSmBiqeOB53rKJuSb9ZWPUXRCp1VBX2FD51OstKX91mpARjF06TidTsjtdgFoXGOcifjJ3qhE47k9yRP/8Jl+eG5Va9kfIKgAAAABJRU5ErkJggg==>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAXCAYAAADpwXTaAAABNklEQVR4XmNgGAVUA/Ly8pOB+D8Uf0GXJwuADJOTk6tHFycZKCgoJIAMA9Ic6HIkA6BBD0CGoYsTDYCad0MNYIKG13F0NQQBUJM3SLOUlJQIlA8OfKAXPdDV4gVADRpQzVkwMSB7DzYvAiPDGIiV0MXhAOYKQmKgiACK9crIyAghi8OBoqKiG1TjcmRxqFgvshhU/Ce6GBwAJeeANIK8iiYOEhMAsdXV1XmRxZHYjjA2TCAb3TvGxsasMDFg+KQCDbUHsYHe4wSJKysry4LEgewjQKyJrBdk4F0gvgH0sjqQvgpUmAvSBOTrA+nHMHVA8YNA/mtkvVgBUKEgUGEyjA/yIpAfiKwGZAFQXAJEI3udLADzOpB+B7Q8VElJSQ0UJOjqCAJZWVkTpHBcAQ2CZ+jqRgFpAABvE1YAp5bciwAAAABJRU5ErkJggg==>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABoAAAAYCAYAAADkgu3FAAABpUlEQVR4Xu2UO0sDQRSFowiKIhYWgYRkkxBIJ4JY2NrbKDY2/gLRzsLGzsLGUrDwH9jZKFYiivhEQYJFghoFH0WIqGggfhd3kutlDQoWCjlw2Lnnm5m7OwsTCjX0L+R53hO+wefxePyC56NiRZwXhi/xvVraTH2HC/galxQLFpNWcAUfWEbzRWGxWKzfMvJ5YYlEYtyyQLHZqN9o3zKyEWE2F5GXbVZXvFGv36homZ9XIpFIu2W84KTN6iocDne4DXXOCyyQDUieTCZ7DJvS9bcV1Mjzf7DPhg171fW3ZRsxPkun062OcUzTim2kUqkuV/9IulE0Gu3maNYMW5IxeRvjI8e0yAfxmM0/STfi+RDAjv3xi2ZObq0aNylck2uEs5QtAazE1+xmMplOzUQc6wT8zdWMc3hGz6lKNVqtw/YsE5Fv44Kqd/CJnlMVoCyb2VzkGtncyfu4pvKq3sTPakpNgOxXm0nOsc3a3Em+QO5JVW/hKz2nKibO4T6bi1h0aDMt1i176lZhfIrX9Zxfkz6Nr07mV8QVlaHBLc7xhUOWN/S39Q52NI4puiL+3gAAAABJRU5ErkJggg==>

[image12]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACUAAAAYCAYAAAB9ejRwAAAB7klEQVR4Xu1VzysEYRhWSn6lqG1q253Z2aZopJQoSU4Kf4OzkhzkplyUHKQQR/4BNweKlFzcXEVWyY9WSC2SA+t51/eu8fZ90zqINE89fd88z/O97zvTzmxZWYT/Ctd1O1KpVKXUfxWO4+TBV6kHkU6nbWRy4Dl4Al6AT+R5nleH/R14CmbAS/CMz2I/Dl6rM1lw7bOyBghsOh9D5fG0mqQvwVnbtodCvHqNd6W8Lul9AQJtFMbazAVlRgKZQ5Ud0HhHphrQc1LTggokEokqtX9TzVZkLgg8zXXK4UZGg7plWTUhN1bu+36FFLVAgSzv0aw1pGgR8Ocog/yC0J/BSd15aDdS0wJ36sfj8eqgVspQGGZY5TZYSyaT3bieRs1BeR5vdiO0paBmhKN521C8XTVclB4DjXspg+GOWeNBsHbKoeS1EfgdeQjfSp2ghjIWgueqTOGmMOQMfedoj9UiD/Ub6BqDj+G6P3jeiLCmBNX0QeoMHhxNe7BOSA96HwZrwTof9IzAoWVwSupBcFPd94bAPtHgjeg8I0oJI7Ovit9Lj8AD0XfO4D3Sb096WiA8S43wWPew7oI74DautyS5cSwWq9XUKXhSJyjvRepGcLFvsvgtE3UyUieQR/+FUv9xoPGB1BjwVqUWIUKEv4Z38uLAvEc1hjcAAAAASUVORK5CYII=>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAXCAYAAAAC9s/ZAAAA00lEQVR4XmNgGAUYQF5e/jkQ/4fi30D8DohfAfEnmLiUlJQIuj4UAFOILq6urs6LSw4FQBX9RRcHAZgBCgoKBuhycABV1I0uDgIEXQBzJro4CMA0g9Sgy8EB0GmdUIWBICwnJxcKxGVQsT509RgAqOgP0JBTsrKyJkC2EVCzMZBvDmTvBuJ76OoxAMgmoCZXdHEQgLriH7o4HAAlg0CK0MVhAGoATnmQgmv4FBBjAE4FQPEkkBwwPDRQJKCB9RqmGYqfA8PhEZB+giT2DUXjKBhOAADF51KZmdj5WgAAAABJRU5ErkJggg==>

[image14]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAXCAYAAADUUxW8AAAAs0lEQVR4XmNgGN5AXl7eE12MKADUuBSI/6OLEwOYQRrJ0gzU9AGIl5CsGajBUkFB4RQQB5CkWVpaWhio4S2IraSkxA/SLCUlxYWuDisAKv4LpJiQ+P+BLjBHUoIdABUmA/EsNDFQoKUgi2EF2PwH1TwJXRwFAJ2WoKio6IaOoZqvo6tHAdhsBQGoZqxyYACUPAPEjujiIIBVM9CZ9jAJbAqA8p1AsT9I8r+A+DiymlEw6AEAhTw4i4iPpjwAAAAASUVORK5CYII=>

[image15]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmwAAAAyCAYAAADhjoeLAAAD/UlEQVR4Xu3dXYinUxgA8FkiokTGMGb+HzMjNS7Q+Cr5LB8lHyEXK+VqpTa52Nxiw+ZmSy6UUuRi5c4FCnfuSKRNUZtbImqVzzU8z877cvbMO5/ttH/m96vTOec5z/ueM3P19M47///YGAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP9709PTdw0Gg2v6/f5N0S7IWPQ3R/yKaNf1er2zMxY5t0W7Iea3HHuHEyvOekec88o8W547Y8Ph8NYmdmObF+Pbc97mbKXY495o92Ufe96TZ5yZmTmrzusSZ5+oYwAAWWD8tdnYKMhzTU1NnV7HyvlKsa1S7xXzg3WsluvRdtZxAIBlxUVXLOZ7yvkoybNOT09P1rFyPj4+fmZ0O8rYetX3Wkuv17uz65quWCufCMZ1+yLnlXoNAGBZIdE86aljL5bzUZJnHQwGdxfzTzvO/3M534j6XmuJ/CPRPqlie1a7T6wdHg6HV0X/db0GAHC0IFlYWDglx7Ozs+e1sV6vN9+MvyjzR02eNdqz7TzfvSuLoxg/2I43Y7VCq0vml++sxe/x1YgtljmlKDbfa8cb3QsA2CaiSPgmioprm/GPTZ9PrR7OcaxdUqQfd3H/XbHX42WL2GNxht0xfjTaI/U1paZge78dl30zPtCON2OjRVTm5zt18/Pzp8bZT4v2UcQO1nmt+FmfbMcb3QsA2CaioHgtCoVnov3Qxpoi6EB/xJ+upeasWWA+XcaGw+HFXQVQFEgPRO5l7VPFWqw/Uba8RzVf9T9Nu/Zs7vFSV7xudQ4AQBYNu6N9MDc3N17EsgD6KoqeS8vcrRB7LdZFS93qa0qx/n3mREH0XBuLs38e7aki7ajI+6kY/1murWSt/Uv9pY/0WPa+X95jamrqnCr2ejlvYuveCwDYRqKwubouFFYqlCL3+Yi/0+ZkH4XSG9HtiPne7KPIuz7Gb0eBclH0i9FfGP1n5X2Op7j3u/VZY/5ytG/LWBMv/1S67Ofrst681JWbsWgfl7GJiYkzIvZhGUtd1wMApJPrQBQOh+pYyg92jQLt/hy3xUVTkOzPFgXdW/Xav1dvjdhjb5xppozF/KFy3irPs96zbSDv9/6xTwuPNMVsnfdHtO+i/RK/r/OL+OH+0tPC32J6UnEJAMD6TU5OnpvfjpDjtpDpKmgidiiKlX35jQkx/rJeP1HyXMX413JtJU0BBQDw35DviQ0GgzejiBlmodZfemfr8nzhv/lw2n9E7IXmmlH7Oqv9caZdY5v8IF0AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGC7+huGuwbbUnNfKQAAAABJRU5ErkJggg==>

[image16]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAIoAAAAXCAYAAADUf9f5AAAFA0lEQVR4Xu1aXYhWVRQdRynxoSiibH6+881PgUNgFIg/EWIQCkIRQ1REoUQvPVQP9dBT9JCKWiQ9WE+CVA8FUQ8FJUEFElRQUmgFlVi9GJKCWo5Sa3/sM7Pv+va+9858Y9J0Fxy+c9ZZ5+x19j33d6avr0GDBg0aNGjQ4D+CwcHBIeYsRkZGVqaUNjE/W2Ce2zHPLcwz2u323fjpZ95CPA0NDQ0yf6lQkcP+Vqu1eT78Yp5J5Gcp8xbDw8NjmsMy1PekB+4vDNjGfQIEHED/3yi7oblN6vi9lXV1gLFTKFsx/l6dc5ejeV364GsVfr9BOcEa6wnlqV489Yrx8fErEH+/enBziL6vUKZw4DZmv6ypA8z/eM4N5noX9fN93SfTEs3N25pD8fUgaTxPh1nTwdjY2LW5Hi0SkyyXPqsF+tXIBsNVQscUNoZwiPGJaX8pnNXA117LzaenXiEbVn4Rd4XEL8nhaaI7fokrBfQv8BjMfR9z6mMzcyjPmfZp9oT2tzxXF6JFgv/CG6yBf2I+ghiXMbjE3WB5cH/Y+XXeX60GybhZeJwdSTW9elrEBGEJE1VIJRtF/b7m8F1rKIOur2uMcMjRw1rfFWksr+2CJ8zxhDe2ABEEiwzNeXwEaD8S/cTExGWWh7lDZp7FOu9Bq8Hmulr5N6UdxY54hmrczaC3NLmczwqpfKMI/7zHozzGfIRofcof1/q5Ek1ho7AncPdUetKB7iLrBK4CtBc8vdx2Mo/6ep33PZLl28oRaUSxI95BZ742PQziVjac5rBJBKl6ozwd8HuZD5BzcIY77LqjHDDveQK3odKTCJC47R5fJ3AVIj24A5mHwUn18Y6jk/FTpu7N5fIBCs8IiNnO888FSTdKlEPwT3o8ygHmPQwMDFyj+pPcZ9dt66Q5r3zntut5aumLSqknHbjD44PALh8h0oupzNfYKFXJcPkSLELMj+U1MvWwSQRpZqO4OeSDknmUz5j3MF8bJV9FPU9mo8SeVOC9qkaBXT4CtIc9PbiDmcfD6nVSxwI+sBp5rtF4n+sYN3bElwH6EzIGSbqK+2aDpBslxTl8JuB3Mh9B9ecCvmqjFHhtFzwh/3cqH3tSwe6ArwxchaTfRuTBlPgf7TxSx0Y5ZDWjo6Mtjfdq1nixIz6CfovYKHUZ18tmSTMbJcrhnoB/gPkI0fqEwzq+1/pvkcbynies/6FKTyKA8EXm5cm4JPCHzEfQj1IS4w7LBwsoPFAiCXcJny+b8+EJujWJHgzV35w2S5p5mI1y+Cnz3hrKAP2fPAYn0ZUad52081UB+b7c6jQ3p6hd8CTeef4uiAAH4iXmBWpkMrdh7kbh6IPdDxq86xKbof1fO9z05RoeXmGzcrYwV8dTBOjWJpM0C5mDr3p1kGaeUcIc2rb65Q9enRxazgJz36/9izOHHOzjMdJGeTm38627ZT7CoX3EGSev1vxhcPpMlUuVfFaXyaXIF7tjpHtW+vJBUN1bVoP2d8q7B0CQbyG5jfrPnh7c2Xwpxe8WGcMHz3qSs8fz5EHPwN+Zt0D/BeYiQHsM5bjGz+UXlP2kE/59qWe/tl81nRz2dX+Snwb6jybNGXKwWuddS5pNynf+nobfMzmfpKv0dNGAYGeZa1AfyN8beMNZxvyCAl4zb8JCH2G+QX38q2f1pcL/YpEXGXiOeJS5BQX9V4XrmW9QH5JD5ho0aNBg4eAfHehrGuB/46kAAAAASUVORK5CYII=>

[image17]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC0AAAAYCAYAAABurXSEAAABcklEQVR4Xu2VK08EMRSFFwEJD0FChknm/Qpi7GgSFAZBCAaPRsFvwAAGsYpfgAABciUBASHwD1AYCEEgdtVwLulCe3cm7JjuiH7JSdvTu52z3U630zEYDNrwfT8timKa+60kiqLNMAxL6BjqUd+27Xle1xoQcBu6rvBLNFPcbwUIdwO9VvilZVkLiuk4zpw89jxvVh7rAuHOxNG4HXpxHIdip/+AMYCOhmcH7UsQBIcjhZoQoX+EzXOpVXYZb+gqzH25mPfrwPyF/IBxxNeogn5l+TN4MXeUApif1KZpuiyKDoT/AT0oxZrAc3dFlq4UvsfrqPCZJrmvG2T4ws6eM+9JBL+S/bGOAyfP85ksy6wm4mtw6jLAfxuZE6G7ivkPqN+ATpqIr8GhHNwjxMXw/mtgsEXFdK6luolAOaDLKh/HZk027uq+oW6km+MRIRdxR6+j3w/5vyR22E+SZEUxJwyOwx6C3kOnrusu8XmDwWBozjcK9IeWY0UdKwAAAABJRU5ErkJggg==>

[image18]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFoAAAAXCAYAAACLbliwAAADjklEQVR4Xu2YS2gUQRCGNyQiioqi62Nfs49AFMTLIgoGvShi0IMoGMWDkJPmIOg9B0EQBOMDHwcRhXj0IObiISfxoCIKShDxJsYoIr5fmMS/1uqxtqYrm2U3gjIfFNv999/VNT0zu50kEjExMTEx/xTFYjGnNUmhUFgVBEGX1usFedbncrmy1iUYL2Kt7Wi26jEJfFvT6XRG63+bVCo1G7VsTExWLwyduKhviBN6jIC+DDEBXz9iHbXz+fxa7ZsKNBfRgzw7uX3a4xlA/uvZbHY12g8R7z2eSk2ILYhDjdTUKFj7JeIW1t+Nz9eIN1WG9vb2pGtz0ZGNxtO3hC9iqZBbSKOnSWg1oTm1NPTvaY1usNSaWVOjcF0tUkNd3dDHpRZibTQm3dUXTrB/VOsW8HYZeUYRB0Q/khc1LJdzG62JXnOtSfAAztOaBa1Jb57U+E2M1FeBi4xsNOuRSZZuAe+Qz48n8A70T9xtJQ9p0pPJZGbJudbalq6BZww501oncBOvYPyS1i3cmog9QnuAOC59IWyezo0e8/mxqTedjovcQG3StE/Otda2dB9Uj9ZAG9a+rMXJQJ5ety7mHsTnAOKR9oWwsd+n+4q3dAvLD23Q6fjcwb5Bj28Cr/RM1zZyeXULeEdcu1wuz6hnrgQPyD63NgW9gdoTQgZMOOXTfQVYuoXlD+rYaBzjFrq2kcurW+B6d8H/im4gz6v6UZsqmPsWcc2tT4Hcee2rwIYzhh4p3tIt4B32+QPx3V0qlRZz3iGPL5xrrW3pk8E/rD/xuUaPTQVaz71pBPLMF3VEz9Q8eNbQI8VbugW8V33+4PcNeCH6lHdYetwNUB5fLq9uAe99bMwxbn+od7Mx56KxXuWoGYjTVAgPnPPoR33J2H9b6xYdHR1zjTzj+G3YJPqUt+oMig3YLOc2oyYc8RbBe0Rq6L+T/Vqg7gu+OgjSccxbqfXKAE3UOkFjuNhu16fvH9LwpGWF9pRz9DlNw0W1ebQQzD+vNfQfe7SaNVnwJlsb9FFrFu5rAjUvkHoymZxTlR+GbRBGED9ogOML4rmYR74+GqO/yKjPvhvSg/4T1j9LXUKbEPw5M9OcZz4/tK/ujwCsvVeu7XA1UdudGHRNFoH/aFeBnsLA8ye/BTb7JK/dwz+qh6lf6/9GDYNFvmstpslgk1fgDu/XekyTca9yzDSC78xOfL+ltB4TE/M/8QuZ33fV9bFf2AAAAABJRU5ErkJggg==>

[image19]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFoAAAAXCAYAAACLbliwAAAD6klEQVR4Xu2XXUhUQRiG1xSyKIpqQ9R1/GWNILqoIKLEi7C6iTCKoG76I6KIIi9Cug5BCArMq7qIoOiPoC4Korsi+oVECLoISgqD/sustN539xud8+2c3TXXoDgvfHjmOe/MfPOddeacWCxSpEiRIv0zMsY019bWztD8b6umpiapmauqqqpa5LoOl8X63lhUXl4+tbq6uklzLeSzAr5SzV3V19fHkdd6zQOCoRuJv0fsQdxC/AKepH0TLeQxH3O/kfm94j0s+koikViM68eID9qTj9CvH/EWY7Xg70/8vaE9yGcv50NsQ1xCDMU8Dxf8O6IHD2SJ+I9pDwc7gUn2a84Omk2wUg8W874Mmxv8HnJd6zLkfxS802W5JMV4qNgA4rPT7tR5cG7N2GYOmiHuu4xwCHEtANPcu1ilIg3GK5O90Bkci2/08TBxa6Qf/Xa5HMU6644jxcoYV/huuV4jYzUqz+uMvgBPpfNFy9BxE9pfXJ9P8MxDfNLcCsnvxFgzNc8mE17oYh+vrKyc4uNhgvewrHe1y5FnBzn//cVHT8a4wlO/fPy9yTZzcD1Y913pO/pDTCaT0+2gDJhWmvRelJeQ4EL4v2mOcfaBP9I8l0xIoTFPk49TYdwnI2cQYqnLMf4h4VvRLJLrQddDCU/NZ9K7QcbcWPt1ch7agRuYpMwOIIZlAUMO4VewAP1+2Dau2xAPXE++MiGFBmv1cYocp/5kzX2C9wn9OEwXKX5Q1n6koqJittRiZM92fOS20CPXynOVHHVt0TfOIN5Zg5g6AqYc4huDbDntRh8EY5D5w0KzOJr7ZHIUGnG8UIVGtLqQT3GH4yNLbeaINpfnkvQZ1nwsMiGFrqurm+vjVBj3Cd4u+lHo5Yqn9m7UYoO0uZaMLVS4LXSvb24jezdzTgFcJHxGSgbs1zxM8J6UD4AyM45im5BCUz6e7QH4BO92+m1BHZ56ADjYGqTN9WeMK7xPrrkTjBZ01BN8AHZvdjwjAr+NuKC5T/Cd5gFg2/xKChs3l0yOQusTnvug8RzG2cRxuBcrdsedl2PqPOwbDl8Y2LYvEnovBhvWfe0TOhCAwvELnaO5lkk/1cuaI6lZGZPlIZOl0FjgCUS3y+Dt8fw6uaaPLnOFe4OIAcXYp9e2UbyNkkeJZZxb5yb9ejysy2UsSIPcOI/BS5m0tNsDRo/4SgT/Kc2t+HEAzyrNfcJ8fYivMreNFzH1UUTOz29eY+7NbLv3rcfHHZWIJ/VFyf9Gnx/suZEDkYen9Gl2PXb7xRhbpM8zk8c3yH8jX+EiTYBQ6FeaRSqw+B4cj8enaR6pwMK5cE6zSJEiFUq/AVwBfPS22IDdAAAAAElFTkSuQmCC>