from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import pandas as pd
import pickle


app = Flask(__name__,template_folder='templates')

collaborative= pickle.load(open('collaborative.pkl', 'rb'))


@app.route("/",methods=['GET'])
def home():
    return render_template('index.html')

@app.route("/kitchen",methods=['GET'])
def kitchen():
    return render_template('kitchen.html')


@app.route("/care",methods=['GET'])
def care():
    return render_template('care.html')


@app.route("/hold",methods=['GET'])
def hold():
    return render_template('hold.html')


@app.route("/contact",methods=['GET'])
def contact():
    return render_template('contact.html')

@app.route('/predict',methods=['post'])
def predict():
    form_value = request.form.values()
    result =[]
    listpred=[]
    for item in form_value:
        result.append(item)
    product_descriptions = pd.read_csv(r'C:\Users\user\Desktop\finalProject\product.csv')
    vectorizer = TfidfVectorizer(stop_words='english')
    X1 = vectorizer.fit_transform(product_descriptions["product_description"])
    Y = vectorizer.transform([result[0]])
    prediction = collaborative.predict(Y)
    order_centroids = collaborative.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    def print_cluster(i):
        print("Cluster %d:" % i),
        for ind in order_centroids[i, :6]:
                print(' %s' % terms[ind])
    for i in range(6):
        print_cluster(i)
    while(prediction):
        prediction = collaborative.predict(Y)
        if(prediction not in listpred):
            listpred.append(prediction)
            print_cluster(prediction[0])
            s1=str(prediction)
        else:
            break
    if(prediction[0]==0):
        return render_template('cluster0.html')
    elif(prediction[0]==1):
        return render_template('cluster1.html')
    elif(prediction[0]==2):
        return render_template('cluster2.html')
    elif(prediction[0]==3):
        return render_template('cluster3.html')
    elif(prediction[0]==4):
        return render_template('cluster4.html')
    elif(prediction[0]==5):
        return render_template('cluster5.html')

    else:
        return render_template('index.html')
        
if __name__ =="__main__":
    app.run(debug=True)
