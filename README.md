# Malicious_URL_Detection
Malicious URL Detection using Logistic Regression Model

Please use python 3.0+

First, install needed mechine learning packets:numpy, scipy, scikit-learn, pandas, etc.
lgs.pickle and vct are binary files that store logistic regresion model and vectorizer respectively.

Run 
python3 predictor.py (or pred.py inside ngram directory).

If you want to start training again, just comment out last part of the code of above files. Uncomment following code:

'''
vectorizer, lgs = TL()
with open('lgs.pickle', 'wb') as model:
        pickle.dump(lgs, model)
with open('vct', 'wb') as vct:
        pickle.dump(vectorizer, vct)
print('Finished writing models')
'''
(comment out code afer it)

After that, you can recover the code, so as to load the model and vectorizer directly.
