# OA_assistant_velo


## Team
- Max DELPRAT
- Maxime MAEDER
- Filipe
- Maxime KLEIN

## Objectif of the projet
Implémenter et mettre en place un assistant à la mécanique vélo. 

Le système charge les PDFs de mécanique vélo, les découpe en chunks de 500 caractères, et crée une base vectorielle avec embeddings OpenAI pour la recherche sémantique.
Quand l'utilisateur pose une question, le retriever trouve les 6 chunks les plus similaires sémantiquement dans la base vectorielle.
Un LLM (GPT-3.5) évalue en une seule requête batch la pertinence réelle de chaque chunk et attribue un score de 0 à 1, puis sélectionne les 5 meilleurs.
Si des chunks ont un score inférieur à 0.5, le système déclenche automatiquement une recherche Tavily sur le web et remplace les chunks faibles par des résultats web pertinents.
Le contexte final (chunks locaux + résultats web si nécessaire) est envoyé à GPT-3.5 qui génère une réponse technique détaillée de mécanicien vélo.

<img width="4250" height="1626" alt="flow_project_OA_IAGEN" src="https://github.com/user-attachments/assets/70ffb6ba-c897-4dfb-8ef7-18b5aa8b624d" />

## Installation

Créer dans votre dossier un fichier **.env** qui comprend 2 clés API. Une clé OpenAI en ce qui concerne les services de vectorisation et de génération, une clé Tavily pour la recherche internet.


```
pip install -r requirements.txt
```




