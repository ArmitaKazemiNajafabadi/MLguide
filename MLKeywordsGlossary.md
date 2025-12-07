# Machine Learning (Gen AI) Keywords & Glossary

> Comprehensive reference with definitions, examples, and Q&A

**Source:** Coursera - Generative AI Introduction and Applications +  Additions with Claude.ai

---

| Term | Definition | Examples | Q&A |
|------|------------|----------|-----|
| **Data Augmentation** | Technique to increase diversity/amount of training data | 1. Image: rotation, flip, crop, color jitter<br>2. Text: synonym replacement, back-translation | **Q:** Why not just collect more data?<br>**A:** Expensive/time-consuming; augmentation is cheaper and improves model robustness to variations |
| **Deep Learning** | Subset of ML using multi-layer artificial neural networks to learn from data | 1. CNNs for image classification<br>2. RNNs/Transformers for language | **Q:** Deep learning vs machine learning?<br>**A:** Deep learning = subset of ML with neural networks (3+ layers); ML includes simpler models like decision trees, SVM |
| **Diffusion Model** | Generative model: trains by adding noise to data, learns to remove noise (reverse diffusion) | 1. Stable Diffusion (images)<br>2. DALL-E 2 (text-to-image) | **Q:** - |
| **Discriminative AI** | AI that distinguishes/classifies between data classes; models P(Y\|X) | 1. Spam filter (email → spam/not spam)<br>2. Image classifier (image → cat/dog) | **Q:** Discriminative vs Generative?<br>**A:** Discriminative learns decision boundaries (classification); Generative learns data distribution (can create new samples) |
| **Discriminative AI Models** | Models that identify/classify based on patterns; used for prediction/classification | 1. Logistic regression<br>2. SVM (Support Vector Machines) | **Q:** Can they generate data?<br>**A:** No, they only classify existing data; cannot synthesize new samples |
| **Foundation Models** | Large pre-trained models with broad capabilities; adaptable to specific tasks via fine-tuning | 1. GPT-4 (text)<br>2. CLIP (vision-language) | **Q:** more examples |
| **Generative Adversarial Network (GAN)** | Two networks: Generator (creates fake data) vs Discriminator (detects fake); trained adversarially | 1. StyleGAN (photorealistic faces)<br>2. CycleGAN (image-to-image translation) | **Q:** Why hard to train?<br>**A:** Balancing generator/discriminator is tricky; mode collapse (generator produces limited variety) common |
| **Generative AI** | AI that creates new content (text, images, audio, video) by learning data distribution | 1. ChatGPT (text generation)<br>2. Midjourney (image generation) | **Q:** How does it "create" vs "copy"?<br>**A:** Learns statistical patterns from training data; generates novel combinations, not direct copying |
| **Generative AI Models** | Models understanding input context to generate new content; used for content creation/interaction | 1. GPT (text)<br>2. Stable Diffusion (images) | **Q:** ... |
| **Generative Pre-trained Transformer (GPT)** | LLM series by OpenAI; uses transformers + massive pre-training on text | 1. GPT-4 (multimodal reasoning)<br>2. ChatGPT (conversational AI) | **Q:** What's "pre-trained"?<br>**A:** Trained on huge unlabeled text corpus first; then fine-tuned for specific tasks (cheaper than training from scratch) |
| **Large Language Models (LLMs)** | Deep learning models trained on massive text; learn language patterns/structure | 1. GPT-4 (OpenAI)<br>2. LLaMA (Meta) | **Q:** Size = quality always?<br>**A:** Generally yes, but diminishing returns; efficiency techniques (LoRA, quantization) help smaller models compete |
| **Machine Learning** | AI subfield: algorithms/models that learn from data to make predictions/decisions | 1. Linear regression (predict house prices)<br>2. K-means clustering (customer segmentation) | **Q:** ML vs AI?<br>**A:** AI = broad field (any intelligent system); ML = subset (learning from data); Deep Learning = subset of ML |
| **Natural Language Processing (NLP)** | AI branch enabling computers to understand/manipulate/generate human language | 1. Sentiment analysis (reviews → positive/negative)<br>2. Machine translation (English → Spanish) | **Q:** NLP = LLMs?<br>**A:** No, NLP = entire field (includes rule-based, statistical, neural); LLMs = one powerful approach within NLP |
| **Neural Networks** | Computational models inspired by brain structure; nodes (neurons) in layers connected by weights | 1. Feedforward NN (basic classification)<br>2. CNN (image recognition) | **Q:** Why called "neural"?<br>**A:** Structure mimics biological neurons: inputs → weighted sum → activation → output (loosely inspired, not identical) |
| **Prompt** | Instructions/questions given to generative AI model to guide output generation | 1. Text: "Write a poem about stars"<br>2. Image: "A cat wearing a spacesuit" | **Q:** Does wording matter?<br>**A:** Yes! Prompt engineering crucial; small changes (add "step-by-step", "professional") significantly affect output quality |
| **Training Data** | Large datasets (with examples/labels) used to teach ML model patterns | 1. ImageNet (14M labeled images for vision)<br>2. Common Crawl (web text for LLMs) | **Q:** More data always better?<br>**A:** Generally yes, but quality > quantity; biased/noisy data → biased/poor models |
| **Transformers** | Deep learning architecture using **self-attention** mechanism; NOT encoder-decoder always (GPT = decoder-only) | 1. BERT (encoder-only, text understanding)<br>2. GPT (decoder-only, text generation) | **Q:** Transformers = time-series?<br>**A:** No! Common confusion. Originally for sequences (NLP), but **attention mechanism** (not recurrence) is key; works for any data (images, graphs, etc.) attention-based sequence model: processes sequences via attention|
| **Variational Autoencoder (VAE)** | Generative model: encoder → latent space (probabilistic) → decoder; learns smooth compressed representation | 1. Face generation (interpolate features)<br>2. Anomaly detection (reconstruct normal data) | **Q:** VAE vs regular autoencoder?<br>**A:** Regular autoencoder: deterministic compression (encoder → fixed code); VAE: probabilistic (encoder → mean+variance, sample from distribution) - enables smooth generation<br><br>**Q:** VAE vs Transformer?<br>**A:** Completely different! VAE = compression+generation architecture; Transformer = attention-based sequence model. VAE compresses to latent space; Transformer processes sequences via attention (no compression required) |

---

## Architecture Comparisons

### Autoencoder vs VAE vs Transformer

| Aspect | **Autoencoder** | **VAE** | **Transformer** |
|--------|----------------|---------|----------------|
| **Purpose** | Compression, dimensionality reduction | Generative modeling, smooth latent space | Sequence processing, attention-based |
| **Core Mechanism** | Encoder-decoder (deterministic) | Encoder-decoder (probabilistic) | Self-attention (no encoder-decoder required for GPT) |
| **Output** | Reconstructed input | Generated samples from distribution | Sequence predictions |
| **Latent Space** | Fixed point per input | Distribution (mean + variance) | No explicit latent space |
| **Can Generate?** | No (just reconstructs) | Yes (sample from latent space) | Yes (autoregressive generation) |
| **Examples** | Image denoising | Face generation, anomaly detection | GPT (text), ViT (images) |

### Transformer Variants

| Type | Structure | Use Case | Example |
|------|-----------|----------|---------|
| **Encoder-only** | Only encoder layers | Understanding, classification | BERT (fill-in-the-blank, sentiment) |
| **Decoder-only** | Only decoder layers | Generation (autoregressive) | GPT (text generation) |
| **Encoder-Decoder** | Both encoder + decoder | Seq-to-seq (translation) | T5 (text-to-text), BART |

---

## Key Insights

### Why Transformers ≠ Time-Series (Common Misconception)

**Confusion source:** RNNs (older models) were designed for sequences/time-series  
**Reality:** Transformers use **attention**, NOT recurrence
- Attention = model looks at all positions simultaneously (parallel)
- RNN = processes sequentially (one step at a time)
- Transformers work on ANY data (text, images, graphs) - sequence is just one application

### Generative Model Comparison

| Model | Training | Strengths | Weaknesses |
|-------|----------|-----------|------------|
| **GAN** | Adversarial (2 networks compete) | High-quality, sharp images | Hard to train, mode collapse |
| **VAE** | Reconstruction + regularization | Stable training, smooth interpolation | Blurry outputs |
| **Diffusion** | Iterative denoising | Best quality, stable | Slow generation (many steps) |

---

**Last Updated:** December 2025  
**Companion Files:** ML-Guide.md, RAG-Agentic-AI-Reference.md
