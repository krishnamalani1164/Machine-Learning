# Natural Language Processing (NLP) Fundamentals

## Overview
NLP combines computational linguistics with machine learning to enable computers to understand, interpret, and generate human language.

## Core NLP Pipeline

### 1. Text Preprocessing
```
Raw Text → Tokenization → Normalization → Clean Text
```

**Tokenization**: Split text into words/sentences
```
"Hello world!" → ["Hello", "world", "!"]
```

**Normalization**: Lowercase, remove punctuation, handle contractions
```
"Don't worry!" → "do not worry"
```

### 2. Linguistic Analysis

**Part-of-Speech (POS) Tagging**
```
"The cat sits" → [("The", "DT"), ("cat", "NN"), ("sits", "VBZ")]
```

**Named Entity Recognition (NER)**
```
"John lives in New York" → [(John, PERSON), (New York, LOCATION)]
```

**Dependency Parsing**
```
Tree structure showing grammatical relationships between words
```

## Key NLP Techniques

### Bag of Words (BoW)
```
Text: "cat sat on mat"
BoW: {"cat": 1, "sat": 1, "on": 1, "mat": 1}
Vector: [1, 1, 1, 1]
```

### TF-IDF (Term Frequency-Inverse Document Frequency)
```
TF(t,d) = (Number of times term t appears in document d) / (Total terms in d)

IDF(t,D) = log(Total documents / Documents containing term t)

TF-IDF(t,d,D) = TF(t,d) × IDF(t,D)
```

### N-grams
```
Unigrams: ["natural", "language", "processing"]
Bigrams: ["natural language", "language processing"]  
Trigrams: ["natural language processing"]
```

## Word Embeddings

### Word2Vec
**Skip-gram**: Predict context words from target word
**CBOW**: Predict target word from context

### GloVe (Global Vectors)
```
Xᵢⱼ = Co-occurrence count of words i and j
f(Xᵢⱼ) = (wᵢᵀw̃ⱼ + bᵢ + b̃ⱼ - log Xᵢⱼ)²
```

### Similarity Metrics
```
Cosine Similarity = (A · B) / (||A|| × ||B||)
Euclidean Distance = √(Σ(aᵢ - bᵢ)²)
```

## Language Models

### N-gram Language Model
```
P(w₁w₂...wₙ) = ∏P(wᵢ|wᵢ₋ₙ₊₁...wᵢ₋₁)
```

### Neural Language Models
- **RNN/LSTM**: Sequential processing
- **Transformer**: Attention mechanism
- **GPT**: Generative pre-trained transformer
- **BERT**: Bidirectional encoder representations

## Attention Mechanism
```
Attention(Q,K,V) = softmax(QKᵀ/√d)V

Where:
Q = Query matrix
K = Key matrix  
V = Value matrix
d = Dimension of key vectors
```

## Common NLP Tasks

### Text Classification
```
Input: "This movie is amazing!"
Output: Positive (0.95), Negative (0.05)
```

### Sentiment Analysis
```
Polarity Score ∈ [-1, 1]
-1: Very Negative, 0: Neutral, +1: Very Positive
```

### Machine Translation
```
Encoder-Decoder Architecture:
Source → Encoder → Context Vector → Decoder → Target
```

### Question Answering
```
Context + Question → Answer Span/Generation
```

### Text Summarization
- **Extractive**: Select important sentences
- **Abstractive**: Generate new sentences

## Evaluation Metrics

### Classification Metrics
```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)  
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
```

### Generation Metrics
```
BLEU Score: N-gram overlap between generated and reference text
ROUGE Score: Recall-oriented evaluation for summarization
Perplexity: 2^(-1/N × Σlog₂P(wᵢ))
```

## Modern NLP Architecture

### Transformer Architecture
```
Input → Embedding → Positional Encoding → 
Multi-Head Attention → Feed Forward → Output
```

### Pre-training Strategies
- **Masked Language Modeling** (BERT)
- **Autoregressive Generation** (GPT)
- **Sequence-to-Sequence** (T5)

## Popular Libraries & Frameworks
- **spaCy**: Industrial-strength NLP
- **NLTK**: Natural Language Toolkit
- **Transformers**: Hugging Face library
- **Gensim**: Topic modeling and similarity
- **scikit-learn**: ML algorithms for NLP

## Applications
- **Search engines** - Query understanding
- **Chatbots** - Conversational AI
- **Translation** - Cross-language communication
- **Content moderation** - Spam/hate speech detection
- **Information extraction** - Knowledge graphs
- **Text mining** - Insights from documents
- **Voice assistants** - Speech-to-text processing
- **Legal tech** - Document analysis
- **Healthcare** - Clinical note processing

## Recent Advances
- **Large Language Models** (GPT-4, ChatGPT)
- **Multimodal models** (CLIP, DALL-E)
- **Few-shot learning** with prompting
- **Instruction tuning** and RLHF
- **Retrieval-augmented generation** (RAG)
