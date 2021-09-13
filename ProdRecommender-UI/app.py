# import Flask class from the flask module
#from flask import Flask, request
from flask import Flask, jsonify,  request, render_template
import joblib
import numpy as np
#from sklearn.externals 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Create Flask object to run
app = Flask(__name__)

item_final_rating_pk = joblib.load('pickle/item_final_rating.pkl')
Item_Reviews = joblib.load('pickle/Item_Reviews.pkl')
#xgb_model_pk = joblib.load('pickle/xgb_model.pkl')
#word_vectorizer = joblib.load('pickle/word_vectorizer.pkl')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict')
def predict():
    user_input = request.args['username']
    try:
        Top20_Recomm_products_df = pd.DataFrame(item_final_rating_pk.loc[user_input].sort_values(ascending=False)[0:20])
        ##### merging with main dataset and getting reviews for the above 20 products
        Top20_Prod_reviews =    Item_Reviews[Item_Reviews['name'].isin(Top20_Recomm_products_df.index)]
        ####### Following code measures sentiment using model, but it takes around 25 seconds, to avoid timeout, I used generated sentiment
        #Top20_Prod_reviews.insert(15, "Sentiscore", 0 )
        #for index, row in Top20_Prod_reviews.iterrows():
        #    review_str = Top20_Prod_reviews.at[index,'reviews_text']
        #    Result = self.xgb_model_pk.predict(self.word_vectorizer.transform(review_str.split('.')))
        #    Top20_Prod_reviews.at[index,'Sentiscore'] = Result[0]
        ##### End of on the fly sentiment code
        Top5_recommended_products=Top20_Prod_reviews.groupby(['name']).sum().sort_values(by='user_sentiment',ascending=False)[0:5]
        Top5_recommended_products_df = pd.DataFrame(Top5_recommended_products.index)
        output2 = Top5_recommended_products_df.to_string(header=False,index=False)
        output= output2.split("\n")
    except Exception as e:
        output = "Sorry, we are unable to predict for the given user"
    return render_template('index.html', header='TOP 5 Recommended Products', prediction_text=output)
if __name__ == "__main__":
    app.run()