#!/usr/bin/env python
import pandas as pd
from sklearn.externals import joblib
from treeinterpreter import treeinterpreter as ti
from optparse import OptionParser
from sklearn.ensemble import RandomForestClassifier

from featureizer import featureize
from flowenhancer import enhance_flow
from clearcut_utils import load_brofile
from train_flows_rf import fields_to_use
import logging
import sys
logging.basicConfig()

if __name__ == "__main__":
    __version__ = '1.0'
    usage = """analyze_flows [options] inputfile"""
    parser = OptionParser(usage=usage, version=__version__)
    parser.add_option("-f", "--randomforestfile", action="store", type="string", \
                      default='/tmp/rf.pkl', help="")
    parser.add_option("-x", "--vectorizerfile", action="store", type="string", \
                      default='/tmp/vectorizers.pkl', help="")
    parser.add_option("-a", "--anomalyclass", action="store", type="int", \
                      default='1', help="")
    parser.add_option("-v", "--verbose", action="store_true", default=False, \
                      help="enable verbose output")

    (opts, args) = parser.parse_args()

    if len(args)!=1:
        parser.error('Incorrect number of arguments')

    #load the http data in to a data frame
    print('Loading HTTP data')
    df = load_brofile(args[0], fields_to_use)

    total_rows = len(df.index)
    if opts.verbose: print('Total number of rows: %d' % total_rows)

    print('Loading trained model')
    #read the vectorizers and trained RF file
    clf = joblib.load(opts.randomforestfile)
    vectorizers = joblib.load(opts.vectorizerfile)

    print('Calculating features')
    #get a numberic feature dataframe using our flow enhancer and featurizer
    featureMatrix = featureize(enhance_flow(df), vectorizers, verbose=opts.verbose)

    #predict the class of each row using the random forest
    featureMatrix['prediction'] = clf.predict(featureMatrix)

    print
    print('Analyzing')
    #get the class-1 (outlier/anomaly) rows from the feature matrix, and drop the prediction so we can investigate them
    outliers = featureMatrix[featureMatrix.prediction == opts.anomalyclass].drop('prediction',axis=1)

    num_outliers = len(outliers.index)
    print 'detected %d anomalies out of %d total rows (%.2f%%)' % (num_outliers, total_rows, (num_outliers * 1.0 / total_rows)*100)

    if num_outliers == 0:
        sys.exit(0)

    if (opts.verbose) and type(clf) is RandomForestClassifier:
        print 'investigating all the outliers'
        #investigate each outlier (determine the most influential columns in the prediction)
        prediction, bias, contributions = ti.predict(clf, outliers)
        print 'done'
        print(contributions.shape)

    i=0
    #for each anomaly
    for index, row in outliers.iterrows():
        print('-----------------------------------------')
        print 'line ',index
        #find the row in the original data of the anomaly. print it out as CSV.
        print pd.DataFrame(df.iloc[index]).T.to_csv(header=False, index=False)
        if (opts.verbose) and type(clf) is RandomForestClassifier:
            #if we are verbose print out the investigation by zipping the heavily weighted columns with the appropriate features
            instancecontributions = zip(contributions[i], outliers.columns.values)
            print "Top feature contributions to anomaly class:"
            for (c, feature) in sorted(instancecontributions, key=lambda (c,f): c[1], reverse=True)[:10]:
              print '  ',feature, c[1]
        i=i+1
