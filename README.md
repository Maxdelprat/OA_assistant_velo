# OA_assistant_velo


## Team
- Max DELPRAT
- Maxime MAEDER
- Filipe 
- Maxime KLEIN

## Objectif du projet
Implémenter et mettre en place un assistant à la mécanique vélo. 

### Choix techniques


**Architecture RAG (Retrieval-Augmented Generation)** : Permet de combiner la précision des documents techniques vélo avec la capacité de raisonnement d'un LLM, évitant ainsi les hallucinations tout en maintenant des réponses à jour basées sur des sources vérifiables.

**Vectorisation (OpenAI Embeddings + ChromaDB)** : La transformation des chunks en vecteurs haute dimension permet une recherche sémantique intelligente qui comprend l'intention de la question au-delà des simples mots-clés.

**Chunking optimisé (500 caractères, overlap 50)** : Taille suffisante pour conserver le contexte technique tout en restant assez petite pour une récupération précise, avec un overlap minimal pour réduire la redondance.

**Reranking LLM (GPT-3.5 batch, temperature 0.1)** : Le retriever vectoriel peut retourner des chunks sémantiquement proches mais non pertinents ; le reranking via LLM évalue la pertinence réelle dans le contexte de la question et attribue des scores de 0 à 1. Les chunks avec une note en dessous de 0.6 sont supprimés et des recherches web prennent leur place. La faible température permet d'avoir une note fiable sans artifice.

**Fallback web Tavily (seuil 0.5)** : Complète automatiquement les lacunes de la base documentaire locale en recherchant sur le web quand les chunks locaux sont insuffisants (>0.6), garantissant des réponses complètes même pour des questions hors périmètre initial (ex: historique de marques, innovation récente dans e monde du vélo).

**Génération finale (GPT-3.5, temperature 0.7)** : Synthétise les informations récupérées en réponses techniques claires et précises, avec une température modérée pour équilibrer créativité et fidélité aux sources.


### Description textuelle
Le système charge les PDFs de mécanique vélo, les découpe en chunks de 500 caractères, et crée une base vectorielle avec embeddings OpenAI pour la recherche sémantique.
Quand l'utilisateur pose une question, le retriever trouve les 6 chunks les plus similaires sémantiquement dans la base vectorielle.
Un LLM (GPT-3.5) évalue en une seule requête batch la pertinence réelle de chaque chunk et attribue un score de 0 à 1, puis sélectionne les 5 meilleurs.
Si des chunks ont un score inférieur à 0.5, le système déclenche automatiquement une recherche Tavily sur le web et remplace les chunks faibles par des résultats web pertinents.
Le contexte final (chunks locaux + résultats web si nécessaire) est envoyé à GPT-3.5 qui génère une réponse technique détaillée de mécanicien vélo.

### Description visuelle
<img width="4250" height="1626" alt="flow_project_OA_IAGEN" src="https://github.com/user-attachments/assets/70ffb6ba-c897-4dfb-8ef7-18b5aa8b624d" />

## Installation

Créer dans votre dossier un fichier **.env** qui comprend 2 clés API. Une clé OpenAI en ce qui concerne les services de vectorisation et de génération, une clé Tavily pour la recherche internet.


```
pip install -r requirements.txt
```




