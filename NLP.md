### 10. NLP Fundamentals

#### 10.1 What is NLP?

**Natural Language Processing (NLP)** is a collection of tools and techniques that enable machines to understand, interpret, and generate human language.

**Input:** Unstructured text or speech (converted to text)  
**Output:** Structured data (machine-understandable) or generated natural language

**Core Components:**

```
┌─────────────────────────────────────────────────────────┐
│                    NLP System                           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  INPUT                                                  │
│  ┌──────────────────────┐                               │
│  │ Unstructured Text    │                               │
│  │ or Speech           │                                │
│  └──────────┬───────────┘                               │
│             │                                           │
│             ▼                                           │
│  ┌──────────────────────┐                               │
│  │        NLU           │  Natural Language             │
│  │  (Understanding)     │  Understanding                │
│  │                      │  Unstructured → Structured    │
│  └──────────┬───────────┘                               │
│             │                                           │
│             ▼                                           │
│  ┌──────────────────────┐                               │
│  │  Structured Data     │  Machine processes            │
│  │  (Machine Format)    │  (analysis, storage, etc.)    │
│  └──────────┬───────────┘                               │
│             │                                           │
│             ▼                                           │
│  ┌──────────────────────┐                               │
│  │        NLG           │  Natural Language             │
│  │   (Generation)       │  Generation                   │
│  │                      │  Structured → Unstructured    │
│  └──────────┬───────────┘                               │
│             │                                           │
│             ▼                                           │
│  OUTPUT                                                 │
│  ┌──────────────────────┐                               │
│  │  Natural Language    │                               │
│  │  (Human-Readable)    │                               │
│  └──────────────────────┘                               │
└─────────────────────────────────────────────────────────┘
```

**Keywords:** NLP, unstructured text, structured data, NLU (Natural Language Understanding), NLG (Natural Language Generation), text processing

---

#### 10.2 NLP Applications

**Major Use Cases:**

1. **Machine Translation**
   - Google Translate, DeepL
   - Cross-language communication
   
2. **Virtual Assistants & Chatbots**
   - Siri, Alexa, Google Assistant
   - Customer service bots
   
3. **Sentiment Analysis**
   - Detect tone: serious/sarcastic/positive/negative
   - Social media monitoring, product reviews
   
4. **Spam Detection**
   - Email filtering
   - Content moderation

**NLP vs LLMs - Do Traditional NLP Tools Still Matter?**

**Pre-LLM Era (Traditional NLP):**
- Rule-based systems + statistical models
- Required extensive feature engineering
- Language-specific pipelines
- Limited context understanding

**Post-LLM Era:**
- LLMs have largely **replaced** traditional NLP for many tasks
- Machine translation now dominated by transformer-based models (GPT, T5, mT5)
- BUT traditional NLP still relevant for:
  - **Low-resource scenarios** (limited compute, offline, edge devices)
  - **Specialized domains** (medical, legal) where fine-grained control needed
  - **Interpretability** (rule-based systems are more transparent)
  - **Speed-critical applications** (traditional NLP is faster for simple tasks)
  - **Cost-effective solutions** (LLM APIs can be expensive at scale)

**Keywords:** machine translation, virtual assistants, sentiment analysis, spam detection, chatbots, LLM impact

---

#### 10.3 NLP Pipeline - Core Steps

**Text Processing Workflow:**

```
Raw Text → Tokenization → Normalization → Analysis → Output
           (Step 1)      (Steps 2.1-2.2)  (Steps 3-4)
```

**Step 1: Tokenization**
- **Purpose:** Break text into chunks (tokens)
- **Types:** 
  - Word-level: "I love NLP" → ["I", "love", "NLP"]
  - Subword: "running" → ["run", "##ning"] (used in BERT)
  - Character-level: "cat" → ["c", "a", "t"]

**Step 2: Normalization**

**2.1 Stemming**
- Reduce words to root form (crude, rule-based)
- Example: run ↔ ran ↔ running → "run"
- Fast but less accurate
- "better" → "bett" (loses meaning)

**2.2 Lemmatization**
- Find dictionary form (lemma)
- Uses vocabulary and morphological analysis
- Example: "better" → "good" (semantically correct)
- "running" → "run", "ran" → "run"
- More accurate but slower than stemming

**Step 3: Part-of-Speech (POS) Tagging**
- **Purpose:** Identify grammatical role of each word
- Example: "I **love** NLP"
  - "I" = Pronoun
  - "love" = Verb
  - "NLP" = Noun
- Helps understand sentence structure and meaning

**Step 4: Named Entity Recognition (NER)**
- **Purpose:** Identify and categorize entities
- **Categories:** Person, Organization, Location, Date, Money, etc.
- Example: "**Apple** released iPhone in **Cupertino** on **September 12**"
  - Apple = Organization
  - Cupertino = Location  
  - September 12 = Date

**Keywords:** tokenization, stemming, lemmatization, POS tagging, part-of-speech, NER, named entity recognition, text normalization

---

#### 10.4 Gen AI Impact on NLP

**Before LLMs (Pre-2018):**
- **Pipeline-based:** Each NLP task required separate models
  - One model for translation
  - Another for sentiment analysis
  - Different model for NER
- **Feature engineering:** Manual design of linguistic features
- **Limited context:** Models saw only small text windows
- **Language-specific:** Needed different pipelines per language
- **Examples:** 
  - spaCy, NLTK (traditional NLP libraries)
  - Word2Vec, GloVe (static word embeddings)
  - CRF, HMM for sequence labeling

**After LLMs (2018+):**
- **Unified models:** Single model handles multiple NLP tasks
  - GPT, BERT, T5 are multi-task
  - Same model for translation, summarization, Q&A
- **Contextual understanding:** Transformers capture long-range dependencies
- **Transfer learning:** Pre-train once, fine-tune for specific tasks
- **Multilingual:** Models like mBERT, XLM-R work across 100+ languages
- **Few-shot/zero-shot:** Can perform tasks without explicit training
- **Examples:**
  - GPT-4: Zero-shot translation, sentiment analysis
  - BERT: Contextual embeddings for all downstream tasks
  - T5: "Text-to-Text" paradigm unifies all NLP

**Key Shift:**
```
Traditional NLP: Task-specific models with feature engineering
                 ↓
Gen AI Era:      Foundation models with prompt engineering
```

**What Changed:**
- **Accuracy:** LLMs dramatically improved on benchmarks
- **Flexibility:** One model, many tasks
- **Accessibility:** APIs make NLP accessible without ML expertise
- **Understanding:** Deeper semantic comprehension

**What Remains:**
- Traditional NLP still used in production for speed/cost
- Linguistics research still relies on interpretable methods
- Hybrid approaches combine both (LLM + rule-based filters)


---



**Last Updated:** December 2025  
**Status:** Work in Progress - Building from multiple sections
