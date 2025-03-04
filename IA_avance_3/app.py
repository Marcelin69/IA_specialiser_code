from flask import Flask, render_template,request,url_for
from sklearn.feature_extraction.text import TfidfVectorizer 
from scipy.stats import pearsonr
from lifxlan import LifxLAN



app = Flask(__name__)

num_lights = None
lifx = LifxLAN(num_lights)

lifx.get_power_all_lights()


@app.route("/index",methods=["POST","GET"])
def index():
    print(request)

    if request.method == "POST":
        name = request.form["query"]
        

        texts = [ 
           name, 
           "Le traitement du langage naturel est fascinant.", 
           "Le traitement des langues est une branche de l'intelligence artificielle.", 
           "L'analyse de texte est utilisée pour la traduction automatique.",
           "Allumer",
           "Eteindre"
         ]

         # Vectorisation TF-IDF
        vect = TfidfVectorizer()
        tfidf_mat = vect.fit_transform(texts).toarray()

        query_tf_idf = tfidf_mat[0]
        corpus = tfidf_mat[1:]

  


        # Corellation de pearson
        for id, document_tf_idf in enumerate(corpus):
           pearson_corr, _ = pearsonr(query_tf_idf, document_tf_idf)
           num_lights = None
           if pearson_corr > 0.20:
            
            
            if texts[id+1] == "Allumer":
                lifx.set_power_all_lights("on")
            
                print("Couleur changée avec succès !") 
                print(texts[id+1])
            if texts[id+1] == "Eteindre":
                lifx.set_power_all_lights("off")
                print(texts[id+1])
            result = {"ID": id, "document": texts[id+1], "similarity": pearson_corr}   
            return render_template("index.html", resultat = result)
    return render_template("index.html")



if __name__ == '__main__':
    app.run(debug=True)














