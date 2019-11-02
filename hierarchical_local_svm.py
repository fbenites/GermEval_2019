#!/usr/bin/env python
"""
GermEval 2019 Hierarchical classification shared task
Twistbytes Approach (Fernando Benites)
"""

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# needs the one from pip install git+https://github.com/fbenites/sklearn-hierarchical-classification.git
#or the developer branch
from sklearn_hierarchical_classification.classifier import HierarchicalClassifier
from sklearn_hierarchical_classification.constants import ROOT

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, classification_report, make_scorer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer  
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.multiclass import OneVsRestClassifier  

import nltk
import sys

#read data utilities
from parse_data import *


GOOD_PARAMS = {    'context__vect__word__use_idf': True,
'context__vect__word__sublinear_tf': False,
'context__vect__word__ngram_range': (2, 5),
'context__vect__word__binary': False,    
'context__vect__char__use_idf': True,    
'context__vect__char__sublinear_tf': True,
'context__vect__char__ngram_range': (2, 3),
'context__vect__char__lowercase': True,
'context__vect__char__binary': False}

# Used for seeding random state
RANDOM_STATE = 42


def build_feature_extractor():
    context_features = FeatureUnion(
        transformer_list=[
            ('word', TfidfVectorizer(
                strip_accents=None,
                lowercase=True,
                analyzer='word',
                ngram_range=(1, 7),
                max_df=1.0,
                min_df=0.0,
                binary=False,
                use_idf=True,
                smooth_idf=True,
                sublinear_tf=True,
                max_features=70000
            )),
            ('word3', TfidfVectorizer(
                strip_accents=None,
                lowercase=True,
                analyzer='word',
                ngram_range=(1, 3),
                max_df=1.0,
                min_df=0.0,
                binary=False,
                use_idf=True,
                smooth_idf=True,
                sublinear_tf=True,
                stop_words=nltk.corpus.stopwords.words('german'),
                max_features=70000
            )),
            ('char', TfidfVectorizer(
                strip_accents=None,
                lowercase=False,
                analyzer='char',
                ngram_range=(2, 3),
                max_df=1.0,
                min_df=0.0,
                binary=False,
                use_idf=True,
                smooth_idf=True,
                sublinear_tf=True,
            )),
        ]
    )

    features = FeatureUnion(
        transformer_list=[
            ('context', Pipeline(
                steps=[('vect', context_features)]
            )),
        ]
    )

    return features
    
def print_results(fname,hierarchy,y_pred,mlb,ids,graph):
    
        it_hi=[tj for tk in hierarchy.values() for tj in tk]
        roots=[tk for tk in hierarchy if tk not in it_hi]
        prec=lambda x: [tk for tk in graph.predecessors(x )]+ [tk for tj  in graph.predecessors(x )for tk in prec(tj)]
        with open(fname, "w") as f1:
            for task in range(2):
                if task==0:
                    f1.write("subtask_a\n")
                    for i in range(y_pred.shape[0]):
                        f1.write(ids[i]+"\t")
                        st1=""
                        labs=set()
                        for j in y_pred[i,:].nonzero()[0]:
                            if mlb.classes_[j] in roots:
                                st1+=mlb.classes_[j][2:]+"\t"
                            else:
                                for tk in prec(mlb.classes_[j]):
                                    if tk==-1:
                                        continue
                                    if tk[0]=="0":
                                        labs.add(tk[2:])
                        f1.write(st1[:-1]+"\t".join(labs)+"\n")
                if task==1:
                    f1.write("subtask_b\n")
                    for i in range(y_pred.shape[0]):
                        f1.write(ids[i]+"\t")
                        st1=""
                        for j in y_pred[i,:].nonzero()[0]:
                            st1+=mlb.classes_[j][2:]+"\t"
                        f1.write(st1[:-1]+"\n")

if __name__ == "__main__":
    train=1
    
    if "data" not in globals():
        if train==1:
            data,labels=read_data("blurbs_train.txt")
            data_dev,labels_dev=read_data("blurbs_dev.txt")
        else:
            data,labels=read_data("blurbs_train_and_dev.txt")
            data_dev,labels_dev=read_data("blurbs_test_nolabel.txt")

        hierarchy, levels=read_hierarchy("hierarchy.txt")

    keywords = ["title","authors","body","copyright","isbn"]

    mlb = MultiLabelBinarizer()
    
    #depending on the mode load different data
    if train==1:
        keywords = ["title","authors","body","copyright","isbn"]
        data_train=["\n".join([tk[ky] for ky in keywords if tk[ky]!=None]) for tk in data ]

        labels_train=mlb.fit_transform(labels)

        X_train_raw, X_dev_raw, y_train, y_dev = train_test_split(data_train,labels_train,test_size=0.2,random_state=42)

    else:
        X_train_raw=["\n".join([tk[ky] for ky in keywords if tk[ky]!=None]) for tk in data ]
        y_train=mlb.fit_transform(labels)
        y_train = mlb.transform(labels)

        del data

        ids= [tk["isbn"] for tk in data_dev]
        X_dev_raw=["\n".join([tk[ky] for ky in keywords if tk[ky]!=None]) for tk in data_dev ]


                                            
    vectorizer = build_feature_extractor() #.set_params(**GOOD_PARAMS)
    

    r"""Test that a nontrivial hierarchy leaf classification behaves as expected.
    We build the following class hierarchy along with data from the handwritten digits dataset:
            <ROOT>
           /      \
          A        B
         / \       |  \
        1   7      C   9
                 /   \
                3     8
    """
    if "ROOT" in hierarchy:
        hierarchy[ROOT] = hierarchy["ROOT"]
        del hierarchy["ROOT"]
    class_hierarchy = hierarchy
    bclf = OneVsRestClassifier(LinearSVC())

    base_estimator = make_pipeline(
        vectorizer, bclf)
    

    clf = HierarchicalClassifier(
        base_estimator=base_estimator,
        class_hierarchy=class_hierarchy,
        algorithm="lcn", training_strategy="siblings",
        preprocessing=True,
        mlb=mlb,
        use_decision_function=True
    )

    print("training classifier")
    clf.fit(X_train_raw, y_train[:,:])
    print("predicting")

    y_pred_scores = clf.predict_proba(X_dev_raw)
        
    y_pred_scores[np.where(y_pred_scores==0)]=-10
    y_pred=y_pred_scores>-0.25
    
    if train==1:
        print('f1 micro:',
          f1_score(y_true=y_dev, y_pred=y_pred, average='micro'))
        print('f1 macro:',
          f1_score(y_true=y_dev, y_pred=y_pred, average='macro'))
        print(classification_report(y_true=y_dev, y_pred=y_pred))
    else:
        import networkx as nx
        graph = nx.DiGraph(hierarchy)
        print_results("submission_baseline.txt",hierarchy,y_pred>-0.25,mlb,ids,graph)
                    
                
