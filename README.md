ENGLISH

In this project, user profiles were generated using a Large Language Model (LLM) with the aim of employing them in a Recommender System (RS) to produce prediction lists.

These lists were evaluated and compared with recommendations generated using centroid-based and graph-based methods.

Two datasets were used: Movielens for movies and Dbbook for books.


1) LLM-BASED USER PROFILE GENERATION
   
User profiles were generated using Llama3 ChatQA-1.5 by Nvidia, starting from the list of items liked by each user.

The corresponding code for the movie dataset is available in src/01_generate_profiles/LLMFilm.ipynb.

The generated profiles were organized into two JSON dictionaries (one per dataset), sorted by user ID.

The code for this step (movies) is available in src/01_generate_profiles/CreazioneDizionarioFilm.ipynb.


2) USED EMBEDDINGS
   
For each user profile, an embedding was created using the all-MiniLM-L12-v1 model from Sentence Transformers.

The code related to the movie dataset is available in src/02_generate_embeddings/CreaEmbProfiloFilm.ipynb.

In this project, three types of embeddings were used to generate recommendations:

    a) Graph-based embeddings generated with CompGCN
    
    📄 File: compgcn_384_n1.pkl
    
    b) Textual embeddings based on LLM user profiles (item embedding + LLM-generated user profile embedding)
    
    📄 File: user-profile_item_emb.pkl
    
    c) Textual embeddings based on centroid of liked items (item embedding + centroid embedding).

    The centroid corresponds to the average of the embeddings of the items liked by the user.
    
    📄 File: user-centroid_item_emb.pkl
    
Item embeddings were obtained through sentence encoding of the plot.

All files are located in the data/03_prediction_lists/used_embeddings directory.


3) GENERATION OF PREDICTION LISTS
   
Prediction lists were generated using the GETAll recommendation architecture, which is based on neural networks.

This architecture uses multiple information sources (embeddings) and different strategies for information fusion.

Embeddings were used under three different configurations:

    a) Single-source input
    
    b) Double source combination using concatenation and attention mechanisms
    
    c) Simultaneous combination of all sources using concatenation and attention
    
For each configuration, five versions were tested with different dropout values: 0, 0.2, 0.4, 0.6, and 0.8.

Top-5 and top-10 prediction lists were evaluated.

📁 The code for all configurations is available in src/03_getall_model/config.


4) EVALUATION OF PREDICTION LISTS
   
Prediction lists were evaluated using ClayRS and the following seven metrics: precision, recall, F1 Score, nDCG, EPC, APLT, Gini Index




ITALIANO

In questo progetto di tesi, sono stati generati dei profili utente utilizzando un Large Language Model (LLM) al fine di utilizzarli in un Recommender System (RS) per la produzione di liste di predizione. 

Tali liste sono state valutate e confrontate con le raccomandazioni ottenute tramite metodi basati sul centroide e sul grafo.


Sono stati utilizzati 2 dataset: Movielens per i film e Dbbook per i libri.

1) GENERAZIONE DEI PROFILI UTENTE LLM
   
I profili utente sono stati generati dal modello Llama3 ChatQA-1.5 di Nvidia, a partire dalla lista di item apprezzati da ciascun utente.

Il relativo codice è disponibile in src\01_generate_profiles\LLMFilm.ipynb nel caso dei film.

Successivamente, i profili utente sono stati organizzati in due dizionari JSON (uno per dataset) ordinati per id utente.

Il relativo codice è disponibile in src\01_generate_profiles\CreazioneDizionarioFilm.ipynb nel caso dei film.


2) EMBEDDING UTILIZZATI
   
Per ogni profilo utente è stato generato un embedding tramite il modello all-MiniLM-L12-v1 di Sentence Transformer.

Il relativo codice è disponibile in src\02_generate_embeddings\CreaEmbProfiloFilm.ipynb nel caso dei film.

In questo progetto di tesi sono stati utilizzati 3 tipologie di embedding per generare le raccomandazioni:

  a) Embedding del grafo ottenuto tramite CompGCN.
  
     📄 File compgcn_384_n1.pkl
     
  b) Embedding testuale basato sul profilo utente generato dal LLM: embedding dell'item + embedding del profilo utente LLM
  
     📄 File user-profile_item_emb.pkl
     
  c) Embedding testuale basato sul centroide dei like: embedding dell'item + embedding del centroide. Il centroide corrisponde alla media degli embedding degli item apprezzati dall'utente.
  
     📄 File user-centroid_item_emb.pkl
     
Gli embedding degli item sono stati ottenuti tramite la sentence encoding della trama.

I file sono disponibili nella cartella data\03_prediction_lists\used_embeddings


3) GENERAZIONE DELLE LISTE DI PREDIZIONE
   
Le liste di predizione sono state ottenute tramite l'architettura di raccomandazione GETAll basata su reti neurali. Essa utilizza diverse sorgenti informative (embedding) e sfrutta differenti tipologie di fusione delle informazioni.

Gli embedding sono stati utilizzati secondo 3 diverse configurazioni:

  a) Sorgenti prese singolarmente.
  
  b) Sorgenti combinate due alla volta tramite concatenazione e il meccanismo di attenzione.
  
  c) Sorgenti prese simultaneamente combinate tramite concatenazione e attenzione.
  
Per ogni configurazione sono state generate 5 versioni con 5 valori di dropout differenti (0, 0.2, 0.4, 0.6 e 0.8) e sono state valutate le liste top 5  e top 10.

📁 Il codice per eseguire le diverse configurazioni è contenuto nella cartella src\03_getall_model\config


4) VALUTAZIONE DELLE LISTE DI PREDIZIONE
   
Le liste di predizione sono state valutate con ClayRS utilizzando 7 metriche: precisione, recall, F1 score, nDCG, EPC, APLT e Gini index.


