# NLP-RAG-LLM-GENAI


Proyectos de Procesamiento de Lenguaje Natural, Modelos de Lenguaje (LLM) e IA Generativa.

1. **RAG_Gemini_Consulta_Documentos_PDF.ipynb**
   Sistema de Retrieval-Augmented Generation (RAG) completo: extracción de texto de un PDF, chunking del contenido con LangChain, generación de embeddings en español, búsqueda por similitud coseno sobre los fragmentos más relevantes, y generación de respuestas con el modelo Gemini. Incluye una interfaz de chat construida en Gradio.

2. **BETO_Clasificacion_Sesgo_Politico_Noticias.ipynb**
   Fine-tuning de BETO (BERT en español) para clasificar el sesgo político (Izquierda / Centro / Derecha) de noticias colombianas. El dataset se construyó scrapeando 277+ artículos de medios como El Espectador y El Tiempo. El modelo alcanza 81% de accuracy, con matriz de confusión incluida.

3. **BETO_Deteccion_Ideacion_Suicida_Espanol.ipynb**
   Clasificador binario (Suicida / No Suicida) de tendencia suicida en texto en español, mediante fine-tuning de BETO (`dccuchile/bert-base-spanish-wwm-cased`). Pipeline de entrenamiento propio en PyTorch, con optimizador AdamW, warmup y precisión mixta (fp16).

4. **LSTM_CNN_Word2Vec_Deteccion_Depresion_Tweets.ipynb**
   Arquitectura híbrida CNN + LSTM con embeddings preentrenados Word2Vec (GoogleNews, 300 dimensiones) para detectar señales de depresión en ~13.000 tweets (2.921 depresivos vs. 10.000 aleatorios). Incluye una segunda versión del modelo usando BERT + LSTM.

5. **MobileNetV2_Busqueda_Semantica_Imagenes.ipynb**
   Motor de búsqueda semántica de imágenes: extracción de embeddings visuales con MobileNetV2 (TensorFlow Hub), construcción de una base vectorial y recuperación de las 5 imágenes más similares por distancia coseno. Interfaz de usuario en Gradio.

6. **Embeddings_LLM.ipynb**
   Ejercicio práctico sobre embeddings vectoriales aplicados a modelos de lenguaje: generación, comparación y uso de representaciones semánticas de texto.
