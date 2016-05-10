#!/usr/bin/env python
import pandas as pd
import numpy as np
import sys
import cPickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from treeinterpreter import treeinterpreter as ti
from optparse import OptionParser

from featureizer import featureize
from flowenhancer import enhance_flow
from clearcut_utils import load_brofile
from train_flows_rf import fields_to_use
import logging
logging.basicConfig()

if __name__ == "__main__":
    __version__ = '1.0'
    usage = """analyze_flows [options] inputfile"""
    parser = OptionParser(usage=usage, version=__version__)
    parser.add_option("-f", "--randomforestfile", action="store", type="string", \
                      default='/tmp/rf.pkl', help="")
    parser.add_option("-x", "--vectorizerfile", action="store", type="string", \
                      default='/tmp/vectorizers.pkl', help="")
    parser.add_option("-v", "--verbose", action="store_true", default=False, \
                      help="enable verbose output")

    (opts, args) = parser.parse_args()

    print('Loading HTTP data')
    df = load_brofile(args[0], fields_to_use)

    total_rows = len(df.index) 
    print('Total number of rows: %d' % total_rows)

    print('Loading trained model')
    #read the vectorizers and trained RF file
    clf = joblib.load(opts.randomforestfile)
    vectorizers = joblib.load(opts.vectorizerfile)

    print('Calculating features')
    featureMatrix = featureize(enhance_flow(df), vectorizers)
    featureMatrix['prediction'] = clf.predict(featureMatrix)
    featuresWithoutPredictions = featureMatrix.drop('prediction',axis=1)

    print
    print('Analyzing')
    outliers = featureMatrix[featureMatrix.prediction == 1].drop('prediction',axis=1)

    num_outliers = len(outliers.index) 
    print 'Number of outliers detected: %d (%.2f%% reduction)' % (num_outliers, (1.0 - (num_outliers * 1.0 / total_rows))*100)
    
    if (opts.verbose):
        print 'investigating all the outliers'
        prediction, bias, contributions = ti.predict(clf, outliers)
        print 'done'
        print(contributions.shape)

    i=0
    for index, row in outliers.iterrows():
        print('-----------------------------------------')
        print 'line ',index
        print pd.DataFrame(df.iloc[index]).T.to_csv(header=False, index=False)
        if (opts.verbose):
            instancecontributions = zip(contributions[i], outliers.columns.values)
            print "Top feature contributions to class 1:"
            for (c, feature) in sorted(instancecontributions, key=lambda (c,f): c[1], reverse=True)[:10]:
              print '  ',feature, c[1]
        i=i+1
