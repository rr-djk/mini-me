# Plan 30 jours — Construire ton Agent IA Clone de Personnalité

> **Objectif :** En 30 jours, passer de zéro à un *Personal Digital Twin Agent* fonctionnel,
> en apprenant et en construisant **en parallèle**, chaque jour.
>
> **Rythme :** 1h30 formation + 1h30 code = 3h/jour  
> **Stack :** Python · LangChain · LangGraph · Chroma/Qdrant · OpenAI API

---

## Vue d'ensemble

| Semaine | Thème | Livrable |
|---------|-------|---------|
| **S1** — Jours 1–7 | Fondations & premier agent | Agent qui répond dans ton style avec tes docs |
| **S2** — Jours 8–17 | Mémoire & prise de décision | Clone qui se souvient et raisonne selon tes valeurs |
| **S3** — Jours 18–30 | Autonomie & finalisation | Personal Digital Twin Agent complet sur GitHub |

---

## Semaine 1 — Fondations & premier agent (Jours 1–7)

### Objectif de la semaine
Comprendre les LLMs, les embeddings et le RAG. Construire un premier agent qui parle dans ton style et connaît tes documents.

---

### Formation — Jours 1 à 7

#### LangChain Academy — Modules 1 à 3
- **Lien :** [academy.langchain.com](https://academy.langchain.com)
- **Durée :** ~5–6h (étaler sur 4 jours)

| Module | Contenu clé |
|--------|-------------|
| Module 1 | Introduction aux LLMs · Appels API · Tokens & coûts |
| Module 2 | LangChain basics · Chains · Prompts templates |
| Module 3 | Outils (Tools) · Agents simples · Mémoire courte |

#### DeepLearning.AI — Building AI Agents + RAG & Embeddings
- **Lien :** [deeplearning.ai/short-courses](https://www.deeplearning.ai/short-courses/)
- **Durée :** ~3h (étaler sur 3 jours)
- **Concepts clés :**
  - Comment un LLM "comprend" un texte (embeddings)
  - RAG : Retrieval-Augmented Generation — l'agent consulte tes docs avant de répondre
  - Bases vectorielles : stocker du texte de façon recherchable sémantiquement

---

### Construction en parallèle — Jours 1 à 7

#### Agent v0 — Définir le Persona (Jours 1–3)

L'agent doit parler **comme toi**. Avant d'écrire une ligne de code, rédige ton persona.

**Étape 1 — Rédige ton fichier `persona.txt`**

```
Nom : [ton prénom]
Langue : français (avec expressions québécoises si applicable)
Ton : direct, curieux, analytique, confiant
Style de réponse : concis mais complet, jamais condescendant
Centres d'intérêt : informatique, génie logiciel, IA, agents autonomes
Valeurs : efficacité, apprentissage continu, impact concret
Façon de refuser : poli mais ferme, toujours avec une alternative
Expressions typiques : [tes tics de langage, tournures favorites]
```

**Étape 2 — Code de base (Python)**

```python
# agent_v0.py
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

with open("persona.txt", "r") as f:
    persona = f.read()

SYSTEM_PROMPT = f"""
Tu es le clone numérique de la personne suivante.
Réponds toujours comme cette personne le ferait, avec son style, son ton et ses valeurs.

PERSONA :
{persona}

RÈGLES :
- Ne dis jamais que tu es une IA sauf si on te le demande explicitement
- Utilise le même niveau de langage que la personne
- Réponds avec ses tournures de phrase habituelles
"""

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}")
])
chain = prompt | llm

response = chain.invoke({"question": "Qu'est-ce que tu penses de l'IA générative ?"})
print(response.content)
```

#### Agent v1 — Connecter tes connaissances (Jours 4–7)

**Étape 1 — Prépare tes documents**

Rassemble dans un dossier `/mes_docs/` :
- Tes notes de cours (PDF ou `.txt`)
- Des anciens projets GitHub (README)
- Des messages ou emails exportés
- Tout texte qui reflète ta façon de penser

**Étape 2 — Indexer dans une base vectorielle**

```python
# indexer.py
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Charger tous les documents
loader = DirectoryLoader("./mes_docs/", glob="**/*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

# Découper en chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# Créer la base vectorielle
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
print(f"{len(chunks)} chunks indexés.")
```

**Étape 3 — Agent avec RAG**

```python
# agent_v1_rag.py
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    Tu es le clone numérique d'une personne.
    Utilise les informations suivantes de ses documents pour répondre.
    Si l'information n'est pas dans les documents, réponds avec son style habituel.

    DOCUMENTS :
    {context}

    QUESTION : {question}
    RÉPONSE (dans le style de la personne) :
    """
)

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.6),
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT}
)

response = qa_chain.invoke("Quels sont tes projets en cours ?")
print(response["result"])
```

---

### ✅ Livrable Jour 7

> Un agent Python qui :
> - Répond dans ton style et ton ton
> - Connaît tes documents (PDF, notes, projets)
> - Utilise le RAG pour trouver l'info pertinente avant de répondre

---

---

## Semaine 2 — Mémoire & prise de décision (Jours 8–17)

### Objectif de la semaine
L'agent doit se **souvenir des conversations passées** et commencer à **raisonner selon tes valeurs** pour prendre des décisions.

---

### Formation — Jours 8 à 17

#### LangGraph Academy — Agents avec états
- **Lien :** [academy.langchain.com/courses/intro-to-langgraph](https://academy.langchain.com/courses/intro-to-langgraph)
- **Durée :** ~6–7h (étaler sur 5 jours)

| Module | Contenu clé |
|--------|-------------|
| Graphs & State | Modéliser l'agent comme un graphe d'états |
| Persistence | Sauvegarder l'état entre les sessions |
| Human-in-the-loop | Faire valider une décision avant d'agir |
| Multi-agent flows | Plusieurs agents qui collaborent |

#### DeepLearning.AI — Multi AI Agents Systems (CrewAI)
- **Lien :** [deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai](https://www.deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai/)
- **Durée :** ~3h (étaler sur 2 jours)
- **Concepts clés :**
  - Délégation de tâches entre agents
  - Agent "manager" qui orchestre des agents "workers"
  - Workflows complexes avec rôles définis

---

### Construction en parallèle — Jours 8 à 17

#### Agent v2 — Mémoire persistante (Jours 8–12)

**Concept :** L'agent doit se souvenir de ce qu'on lui a dit hier, la semaine dernière, depuis le début.

**Architecture mémoire :**

```
Mémoire courte (dans la conversation)
    └── LangChain ConversationBufferMemory

Mémoire longue (entre les sessions)
    └── Qdrant / Chroma — stockage vectoriel des conversations passées

Mémoire épisodique (faits importants)
    └── Fichier JSON — événements clés mémorisés explicitement
```

**Code — Mémoire longue avec Qdrant**

```python
# memory_manager.py
from datetime import datetime
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(":memory:")  # ou path="./qdrant_db" pour persister
client.create_collection(
    collection_name="memories",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

def save_memory(text: str, metadata: dict = {}):
    """Sauvegarder un souvenir."""
    embeddings = OpenAIEmbeddings()
    vectorstore = Qdrant(client=client, collection_name="memories", embeddings=embeddings)
    metadata["timestamp"] = datetime.now().isoformat()
    vectorstore.add_texts([text], metadatas=[metadata])

def recall_memories(query: str, k: int = 5) -> list[str]:
    """Retrouver les souvenirs pertinents."""
    embeddings = OpenAIEmbeddings()
    vectorstore = Qdrant(client=client, collection_name="memories", embeddings=embeddings)
    results = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in results]

# Exemple
save_memory("L'utilisateur travaille sur un projet de détection d'anomalies en ML.")
save_memory("L'utilisateur préfère les réponses courtes et directes.")
memories = recall_memories("projet machine learning")
print(memories)
```

**Code — Agent v2 complet avec mémoire**

```python
# agent_v2_memory.py
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from memory_manager import recall_memories, save_memory

def chat_with_clone(user_input: str, session_memory: ConversationBufferWindowMemory):
    # 1. Retrouver les souvenirs pertinents
    past_memories = recall_memories(user_input)
    memory_context = "\n".join([f"- {m}" for m in past_memories])

    # 2. Construire le prompt avec mémoire
    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=f"""
        Tu es le clone numérique de [prénom].

        SOUVENIRS PERTINENTS DE TES CONVERSATIONS PASSÉES :
        {memory_context}

        CONVERSATION EN COURS :
        {{history}}

        Humain: {{input}}
        Clone:"""
    )

    chain = ConversationChain(
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.7),
        memory=session_memory,
        prompt=prompt,
        verbose=False
    )

    response = chain.predict(input=user_input)

    # 3. Sauvegarder les infos importantes
    save_memory(f"Conversation: '{user_input}' → Réponse: '{response[:100]}...'")

    return response
```

#### Agent v3 — Raisonnement selon tes valeurs (Jours 13–17)

**Concept :** L'agent doit prendre des décisions **comme tu le ferais** — pas juste répondre, mais raisonner.

**Étape 1 — Créer ton fichier de valeurs**

```python
# valeurs.py
MES_VALEURS = {
    "priorités": [
        "Impact concret avant théorie abstraite",
        "Code propre et maintenable",
        "Apprendre en faisant, pas en regardant",
        "Ne pas réinventer la roue inutilement",
        "La simplicité bat la complexité"
    ],
    "comment_je_decide": [
        "Je cherche toujours la solution la plus simple d'abord",
        "Je valide avec un prototype avant de construire en grand",
        "Je préfère les outils éprouvés aux outils trendy",
        "Je documente ce que je construis"
    ],
    "ce_que_je_refuse": [
        "Travailler sur des projets sans valeur claire",
        "Copier du code sans le comprendre",
        "Livrer quelque chose de non testé"
    ]
}
```

**Étape 2 — Agent décisionnel avec LangGraph**

```python
# agent_v3_decision.py
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from typing import TypedDict
from valeurs import MES_VALEURS
import json

class AgentState(TypedDict):
    question: str
    contexte: str
    raisonnement: str
    decision: str

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
valeurs_str = json.dumps(MES_VALEURS, ensure_ascii=False, indent=2)

def analyser(state: AgentState) -> AgentState:
    """Analyser la situation avant de décider."""
    response = llm.invoke(f"""
    Tu es le clone de [prénom]. Analyse cette situation selon ses valeurs.

    VALEURS :
    {valeurs_str}

    QUESTION : {state["question"]}

    Analyse la situation en 3 points. Sois concis.
    """)
    return {**state, "raisonnement": response.content}

def decider(state: AgentState) -> AgentState:
    """Prendre une décision finale."""
    response = llm.invoke(f"""
    Sur la base de cette analyse :
    {state["raisonnement"]}

    Donne une décision claire et une action concrète.
    Style : direct, sans fioritures.
    """)
    return {**state, "decision": response.content}

# Construire le graphe de décision
graph = StateGraph(AgentState)
graph.add_node("analyser", analyser)
graph.add_node("decider", decider)
graph.add_edge("analyser", "decider")
graph.add_edge("decider", END)
graph.set_entry_point("analyser")
app = graph.compile()

# Exemple
result = app.invoke({
    "question": "Dois-je utiliser FastAPI ou Django pour mon projet d'API ?",
    "contexte": "",
    "raisonnement": "",
    "decision": ""
})
print("Raisonnement :", result["raisonnement"])
print("\nDécision :", result["decision"])
```

---

### ✅ Livrable Jour 17

> Un agent qui :
> - Se souvient des conversations passées (mémoire long terme dans Qdrant)
> - Raisonne selon tes valeurs avant de répondre
> - Prend des décisions via un graphe LangGraph structuré

---

---

## Semaine 3 — Autonomie & finalisation (Jours 18–30)

### Objectif de la semaine
L'agent peut **agir dans le monde réel** (recherche web, APIs, fichiers). Finalisation, tests et publication sur GitHub.

---

### Formation — Jours 18 à 24

#### Hugging Face — Agents Course
- **Lien :** [huggingface.co/learn/agents-course](https://huggingface.co/learn/agents-course/unit0/introduction)
- **Durée :** ~5–6h (étaler sur 5 jours)
- **Certificat gratuit** à l'issue du cours

| Unité | Contenu clé |
|-------|-------------|
| Unit 1 | Qu'est-ce qu'un agent ? Tool calling, ReAct |
| Unit 2 | smolagents — agents légers et efficaces |
| Unit 3 | LlamaIndex agents |
| Unit 4 | LangGraph agents avancés |
| Finale | Construire et soumettre un agent pour certification |

---

### Construction en parallèle — Jours 18 à 30

#### Agent v4 — Outils & autonomie réelle (Jours 18–24)

**Outils à connecter :**

```python
# tools.py
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
import requests

search = DuckDuckGoSearchRun()

@tool
def recherche_web(query: str) -> str:
    """Rechercher des informations sur le web."""
    return search.run(query)

@tool
def lire_fichier(chemin: str) -> str:
    """Lire le contenu d'un fichier local."""
    with open(chemin, "r") as f:
        return f.read()

@tool
def sauvegarder_note(contenu: str, nom_fichier: str) -> str:
    """Sauvegarder une note dans mes fichiers."""
    with open(f"./mes_notes/{nom_fichier}.txt", "w") as f:
        f.write(contenu)
    return f"Note sauvegardée : {nom_fichier}"

@tool
def appeler_api(url: str) -> str:
    """Faire un appel à une API externe."""
    response = requests.get(url, timeout=10)
    return response.text[:2000]
```

**Agent v4 complet avec outils**

```python
# agent_v4_tools.py
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from tools import recherche_web, lire_fichier, sauvegarder_note
from memory_manager import recall_memories

tools = [recherche_web, lire_fichier, sauvegarder_note]
llm = ChatOpenAI(model="gpt-4o", temperature=0.5)

def build_agent(user_input: str):
    past_memories = recall_memories(user_input)
    memory_context = "\n".join([f"- {m}" for m in past_memories])

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""
        Tu es le clone numérique autonome de [prénom].
        Tu peux utiliser des outils pour agir dans le monde réel.

        SOUVENIRS PERTINENTS :
        {memory_context}

        Agis comme cette personne : direct, efficace, analytique.
        Si tu utilises un outil, explique pourquoi brièvement.
        """),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

# Exemple
executor = build_agent("Recherche les dernières avancées sur les agents IA en 2025")
result = executor.invoke({"input": "Recherche les dernières avancées sur les agents IA en 2025"})
print(result["output"])
```

#### Finalisation & publication (Jours 25–30)

**Structure du projet GitHub :**

```
personal-digital-twin/
├── README.md                   # Présentation claire du projet
├── requirements.txt            # Dépendances
├── .env.example                # Variables d'environnement (sans les clés)
├── persona/
│   ├── persona.txt             # Ton persona
│   └── valeurs.py              # Tes valeurs et critères de décision
├── agents/
│   ├── agent_v0_persona.py     # Agent de base
│   ├── agent_v1_rag.py         # Agent avec RAG
│   ├── agent_v2_memory.py      # Agent avec mémoire
│   ├── agent_v3_decision.py    # Agent décisionnel
│   └── agent_v4_tools.py       # Agent autonome final
├── memory/
│   ├── memory_manager.py       # Gestion de la mémoire
│   └── chroma_db/              # Base vectorielle locale
├── docs/
│   ├── architecture.md         # Schéma d'architecture
│   └── how_it_works.md         # Explication technique
├── tools/
│   └── tools.py                # Outils de l'agent
└── main.py                     # Point d'entrée principal
```

**README.md — Template**

```markdown
# Personal Digital Twin Agent 🤖

> Un agent IA qui reproduit ma personnalité, mes connaissances et ma façon de raisonner.

## Fonctionnalités
- 🎭 **Persona** — Répond dans mon style et avec mes valeurs
- 🧠 **RAG** — Connaît mes documents, notes et projets
- 💾 **Mémoire longue** — Se souvient des conversations passées
- ⚖️ **Décision** — Raisonne selon mes critères avant d'agir
- 🔧 **Outils** — Recherche web, fichiers, APIs

## Stack technique
Python · LangChain · LangGraph · Chroma · Qdrant · OpenAI GPT-4o

## Architecture
[Schéma ici]

## Lancer le projet
\`\`\`bash
pip install -r requirements.txt
cp .env.example .env  # Ajouter ta clé OpenAI
python main.py
\`\`\`
```

---

### ✅ Livrable Jour 30 — Personal Digital Twin Agent

> Un agent IA complet qui :
>
> | Capacité | Technologie |
> |---------|-------------|
> | Parle dans ton style | Prompt engineering · Persona |
> | Connaît tes documents | RAG · Chroma |
> | Se souvient des conversations | Mémoire longue · Qdrant |
> | Raisonne selon tes valeurs | LangGraph · StateGraph |
> | Agit dans le monde réel | Tools · DuckDuckGo · API |
>
> **3 certificats obtenus :**  
> - LangChain Academy  
> - DeepLearning.AI (Building AI Agents + CrewAI)  
> - Hugging Face Agents Course

---

---

## Ressources essentielles

### Documentation officielle

| Ressource | URL |
|-----------|-----|
| LangChain docs | [python.langchain.com](https://python.langchain.com) |
| LangGraph docs | [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/) |
| OpenAI API | [platform.openai.com/docs](https://platform.openai.com/docs) |
| Qdrant docs | [qdrant.tech/documentation](https://qdrant.tech/documentation/) |
| Hugging Face Agents | [huggingface.co/learn/agents-course](https://huggingface.co/learn/agents-course) |

### Concepts à maîtriser absolument

```
LLMs & Tokens          → Comprendre comment un modèle traite le texte
Embeddings             → Représenter du texte sous forme de vecteurs
RAG                    → Retrieval-Augmented Generation
Vector Stores          → Chroma, Qdrant — stocker et chercher des embeddings
Prompt Engineering     → Concevoir des instructions précises et efficaces
LangChain Chains       → Chaîner des appels LLM
LangGraph StateGraph   → Modéliser des workflows d'agents complexes
Tool Calling           → Donner des outils à un agent pour agir
Memory Systems         → Mémoire courte vs longue vs épisodique
```

### Stack recommandé

```python
# requirements.txt
langchain>=0.2.0
langchain-openai>=0.1.0
langchain-community>=0.2.0
langgraph>=0.1.0
openai>=1.0.0
chromadb>=0.5.0
qdrant-client>=1.9.0
python-dotenv>=1.0.0
pypdf>=4.0.0
duckduckgo-search>=6.0.0
```

---

## Conseils pour réussir en 30 jours

### La règle d'or
> **Ce que tu apprends le jour J, tu le codes le jour J+1.**
> Pas dans 3 semaines. Le lendemain.

### Quand tu bloques
1. Lis l'erreur en entier (vraiment)
2. Cherche dans la doc officielle
3. Demande à Claude ou ChatGPT en collant le code complet
4. Maximum 20 minutes bloqué — ensuite tu demandes de l'aide

### Pour le portfolio
- Commit tous les jours, même petits
- Écris des messages de commit descriptifs (`feat: add long-term memory with Qdrant`)
- Document chaque version d'agent dans le README
- Filme une démo de 2 minutes pour LinkedIn

---

*Plan créé le 18 juin 2026 — Bac en informatique & génie logiciel*
