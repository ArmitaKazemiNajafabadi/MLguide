# RAG & Agentic AI

Source Document: https://www.coursera.org/learn/develop-generative-ai-applications-get-started/ungradedWidget/Nhn3P/reading-comprehensive-guide-to-generative-ai
> Quick reference for Retrieval-Augmented Generation, Multi-Agent Systems, and Advanced AI Techniques

---

## Table of Contents

1. [Core Concepts & Definitions](#core-concepts--definitions)
2. [Tools & Frameworks](#tools--frameworks)
3. [Advanced Prompting Techniques](#advanced-prompting-techniques)
4. [Key Architectures & Workflows](#key-architectures--workflows)
5. [Generative AI vs Agentic AI](#generative-ai-vs-agentic-ai)
6. [References](#references)

---

## Core Concepts & Definitions

| Concept | Definition | Examples/Use Cases |
|---------|------------|-------------------|
| **RAG (Retrieval-Augmented Generation)** | Combines retrieval from external knowledge sources with LLM generation to enhance factual accuracy and reduce hallucinations | Answering questions with real-time data, grounding responses in documents ([RAG Paper](https://arxiv.org/abs/2005.11401)) |
| **Retriever** | A system component designed to fetch relevant information from a dataset or database using semantic search | Vector similarity search using FAISS, Elasticsearch, dense passage retrieval |
| **Agent** | An autonomous AI system that can plan, reason, and execute tasks using tools with minimal human intervention | AutoGPT, LangChain Agents, ReAct agents |
| **Multi-Agent System** | A framework in which multiple AI agents collaborate to solve complex tasks, each with specialized roles | Microsoft AutoGen, CrewAI, coordinated problem-solving |
| **Chain-of-Thought (CoT)** | A prompting technique that encourages models to decompose problems into intermediate reasoning steps | "Let's think step by step…", showing work in math problems |
| **Hallucination Mitigation** | Strategies to reduce incorrect or fabricated outputs from LLMs | RAG for grounding, fine-tuning on accurate data, prompt constraints, fact-checking |
| **Vector Database** | A database optimized for storing and querying high-dimensional vector embeddings | Pinecone, Chroma, Weaviate, Milvus |
| **Orchestration** | Tools to manage and coordinate workflows involving multiple AI components and agents | LangChain, LlamaIndex, LangGraph for stateful workflows |
| **Fine-tuning** | Adapting pre-trained models for specific tasks or domains using targeted data | LoRA (Low-Rank Adaptation), QLoRA (quantized fine-tuning), full fine-tuning |

---

## Tools & Frameworks

### Model Development & Deployment

| Tool/Framework | Definition | Examples/Use Cases | Reference Link |
|----------------|------------|-------------------|----------------|
| **Hugging Face** | A platform hosting pre-trained models, datasets, and tools for NLP/CV tasks | Accessing GPT-2, BERT, Stable Diffusion, fine-tuning transformers | [huggingface.co](https://huggingface.co/) |
| **LangChain** | A framework for building applications with LLMs, agents, chains, and memory | Creating chatbots with memory, web search integration, multi-step reasoning | [langchain.com](https://www.langchain.com/) |
| **AutoGen** | A library by Microsoft for creating multi-agent conversational systems | Simulating debates between AI agents, collaborative coding | [AutoGen](https://microsoft.github.io/autogen/) |
| **CrewAI** | A framework for assembling collaborative AI agents with role-based tasks | Task automation with specialized agents (researcher, writer, analyst) | [crewai.com](https://www.crewai.com/) |
| **BeeAI** | A lightweight framework to build production-ready multi-agent systems | Distributed problem-solving, enterprise agent deployments | [BeeAI](https://github.com/i-am-bee/bee-agent-framework) |
| **LlamaIndex** | A tool to connect LLMs to structured or unstructured data sources | Building Q&A systems over private documents, data indexing | [llamaindex.ai](https://www.llamaindex.ai/) |
| **LangGraph** | A library for building stateful, multi-actor applications with LLMs | Cyclic workflows, agent simulations, complex state management | [LangGraph](https://langchain-ai.github.io/langgraph/) |

### Retrieval & Infrastructure

| Tool/Framework | Definition | Examples/Use Cases | Reference Link |
|----------------|------------|-------------------|----------------|
| **FAISS** | A library by Meta for efficient similarity search of dense vectors | Retrieving top-k documents for RAG, large-scale nearest neighbor search | [FAISS](https://github.com/facebookresearch/faiss) |
| **Pinecone** | A managed cloud service for vector database operations | Storing embeddings for real-time retrieval, scalable semantic search | [pinecone.io](https://www.pinecone.io/) |
| **Chroma** | Open-source embedding database for AI applications | Local vector storage, document retrieval for RAG | [trychroma.com](https://www.trychroma.com/) |
| **Weaviate** | Open-source vector database with built-in ML capabilities | Semantic search, hybrid search (vector + keyword) | [weaviate.io](https://weaviate.io/) |
| **Haystack** | An end-to-end framework for building RAG and search pipelines | Deploying enterprise search systems, question answering | [Haystack](https://haystack.deepset.ai/) |

---

## Advanced Prompting Techniques

| Concept | Definition | Example |
|---------|------------|---------|
| **Few-Shot Prompting** | Providing examples in the prompt to guide the model's output format and style | "Translate to French: 'Hello' → 'Bonjour'; 'Goodbye' → 'Au revoir'; 'Thank you' → __" |
| **Zero-Shot Prompting** | Directly asking the model to perform a task without providing examples | "Classify this tweet as positive, neutral, or negative: {tweet}" |
| **Chain-of-Thought (CoT)** | Encouraging step-by-step reasoning to improve accuracy on complex tasks | "First, calculate the area of the rectangle. Then, compare it to the circle's area. Final answer: ___" |
| **Prompt Chaining** | Breaking complex tasks into smaller prompts executed sequentially, using outputs as inputs | Prompt 1: "Extract keywords from this article" → Prompt 2: "Generate a summary using these keywords: {keywords}" |
| **Self-Consistency** | Generate multiple reasoning paths and select the most consistent answer | Run CoT prompt 5 times, choose the most common final answer |
| **ReAct (Reasoning + Acting)** | Interleave reasoning steps with action execution (tool use) | "Thought: I need current weather. Action: call_weather_api('Boston'). Observation: 45°F. Answer: It's 45°F in Boston." |

---

## Key Architectures & Workflows

### RAG Pipeline

**Standard RAG Workflow:**

```
1. User Query
   ↓
2. Retrieval
   - Encode query into vector embedding
   - Query vector database (e.g., Pinecone, FAISS)
   - Retrieve top-k relevant documents/chunks
   ↓
3. Augmentation
   - Combine retrieved context with user prompt
   - Format: "Context: {retrieved_docs}\n\nQuestion: {query}\n\nAnswer:"
   ↓
4. Generation
   - LLM (e.g., GPT-4, Claude) produces final output
   - Grounded in retrieved documents
   ↓
5. Response to User
```

**Key Benefits:**
- Reduces hallucinations by grounding in real data
- Enables access to proprietary/recent information
- More cost-effective than fine-tuning for knowledge updates

**Components:**
- **Embedder**: Converts text to vectors (e.g., `text-embedding-ada-002`, `sentence-transformers`)
- **Vector DB**: Stores and retrieves document embeddings
- **LLM**: Generates final answer using context

---

### Multi-Agent System Architecture

**Workflow:**

```
┌─────────────────────────────────────────────┐
│         Orchestrator (LangGraph)             │
│  - Manages agent communication               │
│  - Routes tasks to appropriate agents        │
│  - Handles state and memory                  │
└─────────────────┬───────────────────────────┘
                  │
      ┌───────────┼───────────┐
      ▼           ▼           ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│ Agent 1  │ │ Agent 2  │ │ Agent 3  │
│Researcher│ │  Writer  │ │  Critic  │
└────┬─────┘ └────┬─────┘ └────┬─────┘
     │            │            │
     └────────────┼────────────┘
                  │
            ┌─────▼─────┐
            │   Tools   │
            ├───────────┤
            │• Web      │
            │  Search   │
            │• Code     │
            │  Executor │
            │• APIs     │
            └───────────┘
```

**Components:**

1. **Agents**: Specialized roles
   - **Researcher**: Gathers information from web/databases
   - **Writer**: Synthesizes findings into coherent output
   - **Critic**: Reviews and suggests improvements
   
2. **Orchestration**: Coordination layer
   - **LangGraph**: For cyclic workflows, state machines
   - **AutoGen**: For conversational multi-agent systems
   - **CrewAI**: For role-based task delegation

3. **Tools**: External capabilities
   - Web search (Google, Bing APIs)
   - Code execution (Python REPL, sandboxed environments)
   - API integrations (databases, external services)

**Example Use Case:**
```
User: "Research recent AI developments and write a report"
  → Researcher agent: Searches web, gathers articles
  → Writer agent: Drafts report from research
  → Critic agent: Reviews for accuracy and clarity
  → Final output: Polished report delivered to user
```

---

## Generative AI vs Agentic AI

### Key Differences

| Aspect | **Generative AI** | **Agentic AI** |
|--------|-------------------|----------------|
| **Autonomy** | Waits for human prompts | Decides actions autonomously, minimizes human intervention |
| **Human Role** | Human checks and directs AI continuously | Human only intervenes when AI asks for clarification |
| **Decision Making** | Executes single tasks as instructed | Can plan, reason, and decide next steps independently |
| **Task Handling** | One prompt → One response | Breaks complex tasks into smaller subtasks automatically |
| **Reasoning** | Limited to prompt instructions | Chain-of-Thought reasoning, multi-step planning |
| **Tool Use** | Requires explicit tool instructions | Autonomously selects and uses appropriate tools |
| **Memory** | Often stateless between prompts | Maintains state and memory across interactions |
| **Examples** | ChatGPT (standard mode), DALL-E | AutoGPT, BabyAGI, CrewAI agents |

### Common Core: LLMs

**Both systems rely on Large Language Models (LLMs) as their foundation:**
- Generative AI: Uses LLM for single-turn generation
- Agentic AI: Uses LLM for reasoning, planning, and tool selection

### Does Agentic AI Involve Reinforcement Learning (RL)?

**Short Answer:** Sometimes, but not always.

**Details:**
- **Core Agentic AI**: Primarily uses **LLM-based reasoning** without RL
  - Chain-of-Thought prompting
  - ReAct (Reasoning + Acting) framework
  - Tool selection via prompting
  
- **RL-Enhanced Agents**: Some advanced systems incorporate RL
  - **RLHF** (Reinforcement Learning from Human Feedback): Used to align LLMs (e.g., ChatGPT training)
  - **Policy learning**: Agents learn optimal action sequences through trial-and-error
  - **Examples**: Game-playing agents, robotics control

**Most production agentic systems (2025) use:**
- LLM prompting + tool APIs (no RL required)
- Planning algorithms (search, heuristics)
- Few-shot learning and in-context learning

---

## Workflow Comparison

### Generative AI Workflow
```
User Prompt → LLM → Response → Human Review → (Repeat if needed)
```

### Agentic AI Workflow
```
User Goal
  ↓
Agent Planning (break into subtasks)
  ↓
┌─────────────────────────────┐
│ Loop: Until goal achieved   │
│  1. Decide next action      │
│  2. Execute action (tool)   │
│  3. Observe result          │
│  4. Reason about next step  │
└─────────────────────────────┘
  ↓
Final Output
  ↓
