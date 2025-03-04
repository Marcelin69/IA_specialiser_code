query = input("Saisir votre recherche ")

texts = [ 
  query, 
  "Le traitement du langage naturel est fascinant.", 
  "Le traitement des langues est une branche de l'intelligence artificielle.", 
  "L'analyse de texte est utilisée pour la traduction automatique.",
  "Allumer la lampe de la salle 505",
"Eteindre la lumière de la cuisine"

]

print("")
print(texts)

from sklearn.feature_extraction.text import TfidfVectorizer 

# Vectorisation TF-IDF
vect = TfidfVectorizer()
tfidf_mat = vect.fit_transform(texts).toarray()

query_tf_idf = tfidf_mat[0]
corpus = tfidf_mat[1:]

print("APRES")
print(str(tfidf_mat))

from scipy.stats import pearsonr

# Corellation de pearson
for id, document_tf_idf in enumerate(corpus):
   pearson_corr, _ = pearsonr(query_tf_idf, document_tf_idf)
   if pearson_corr > 0.20:
       result = {"ID": id, "document": texts[id+1], "similarity": pearson_corr}
       print(str(result))
