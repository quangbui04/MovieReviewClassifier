from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import numpy as np
from vectorizer import vect

app = Flask(__name__)

cur_dir = os.path.dirname(__file__)
clf = pickle.load(open(os.path.join(cur_dir, 'pkl_objects', 'classifier.pkl'), 'rb'))

db = os.path.join(cur_dir, 'reviews.sqlite')

def classifying(document):
    label = {0: 'negative', 1: 'positive'}
    X = vect.transform([document])
    y = clf.predict(X)[0]
    proba = np.max(clf.predict_proba(X))
    return label[y], proba

def train(document, y):
    X = vect.transform([document])
    clf.partial_fit(X, [y])

def sqlite_entry(document, y):
    conn = sqlite3.connect(os.path.join(cur_dir, 'reviews.sqlite'))
    c = conn.cursor()

    c.execute("INSERT INTO review_db (review, sentiment, date) VALUES (?, ?, DATETIME('now'))", (document, y))
    conn.commit()
    conn.close()

class ReviewForm(Form):
    moviereview = TextAreaField('', [validators.length(min=15), validators.DataRequired()])

@app.route("/")
def index():
    form = ReviewForm(request.form)
    return render_template('home.html', form=form)

@app.route("/result", methods=["POST"])
def results():
    form = ReviewForm(request.form)
    if request.method == "POST" and form.validate():
        review = request.form['moviereview']
        y, proba = classifying(review)
        return render_template('result.html', content=review, prediction=y, probability=round(proba*100,2))
    return render_template('home.html', form=form)

@app.route('/thanks', methods=['POST'])
def feedback():
    feedback = request.form["feedback_button"]
    review = request.form["review"]
    prediction = request.form["prediction"]

    inv_label = {'negative': 0, 'positive': 1}
    y = inv_label[prediction]
    if feedback == "Incorrect":
        y = int(not(y))
    train(review, y)
    sqlite_entry(review, y)
    return render_template('thanks.html')

if __name__ == '__main__':
    app.run(debug=True)