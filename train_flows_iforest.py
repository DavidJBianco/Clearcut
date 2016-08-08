#!/usr/bin/env python
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.externals import joblib
from optparse import OptionParser

from featureizer import build_vectorizers,featureize
from flowenhancer import enhance_flow
from clearcut_utils import load_brofile, create_noise_contrast
import logging
import os, sys

logging.basicConfig()

fields_to_use=['uid','resp_p',
               'method',
               'host',
               'uri',
               'referrer',
               'user_agent',
               'request_body_len',
               'response_body_len',
               'status_code']

if __name__ == "__main__":

    __version__ = '1.0'
    usage = """train_flows [options] normaldatafile"""
    parser = OptionParser(usage=usage, version=__version__)
    parser.add_option("-o", "--maliciousdatafile", action="store", type="string", \
                      default=None, help="An optional file of malicious http logs")
    parser.add_option("-f", "--forestfile", action="store", type="string", \
                      default='/tmp/rf.pkl', help="the location to store the forest classifier")
    parser.add_option("-x", "--vectorizerfile", action="store", type="string", \
                      default='/tmp/vectorizers.pkl', help="the location to store the vectorizer")

    parser.add_option("-m", "--maxfeaturesperbag", action="store", type="int", \
                          default=100, help="maximum number of features per bag")

    parser.add_option("-t", "--maxtrainingfeatures", action="store", type="int", \
                      default=1000000, help="maximum number of rows to train with per class")
    parser.add_option("-n", "--numtrees", action="store", type="int", \
                      default=50, help="number of trees in the forest")

    parser.add_option("-p", "--maxoutlierpct", action="store", type="float", default=0.1, \
                      help="max % allowed outliers")

    parser.add_option("-v", "--verbose", action="store_true", default=False, \
                      help="enable verbose output")

    (opts, args) = parser.parse_args()

    if len(args)!=1:
        parser.error('Incorrect number of arguments')

    print('Reading normal training data')

    #check if file is a directory
    if os.path.isdir(args[0]):
        dfs = []
        for f in os.listdir(args[0]):
            dfs.append(load_brofile(os.path.join(args[0], f), fields_to_use))

        df= pd.concat(dfs, ignore_index=True)
    else:
        df = load_brofile(args[0], fields_to_use)



    if opts.verbose: print('Read normal data with %s rows ' % len(df.index))

    numSamples = len(df.index)

    maxoutliers = int(len(df.index) * opts.maxoutlierpct)
    if opts.verbose: print('Allowing a max of %d outliers ' % maxoutliers)


    if (numSamples > opts.maxtrainingfeatures):
        if opts.verbose: print('Too many normal samples for training, downsampling to %d' % opts.maxtrainingfeatures)
        df = df.sample(n=opts.maxtrainingfeatures)
        numSamples = len(df.index)

    if opts.maliciousdatafile != None:
        print('Reading malicious training data')
        df1 = load_brofile(opts.maliciousdatafile, fields_to_use)
        if opts.verbose: print('Read malicious data with %s rows ' % len(df1.index))
        if (len(df1.index) > maxoutliers):
            if opts.verbose: print('Too many malicious samples for training, downsampling to %d' % maxoutliers)
            df1 = df1.sample(n=maxoutliers)

        #set the classes of the dataframes and then stitch them together in to one big dataframe
        df['class'] = 1
        df1['class'] = -1
        classedDf = pd.concat([df,df1], ignore_index=True)
    else:
        #we weren't passed a file containing class-1 data, so we should generate some of our own.
        noiseDf = create_noise_contrast(df, maxoutliers)
        if opts.verbose: print('Added %s rows of generated malicious data'%maxoutliers)
        df['class'] = 1
        noiseDf['class'] = -1
        classedDf = pd.concat([df,noiseDf], ignore_index=True)

    #add some useful columns to the data frame
    enhancedDf = enhance_flow(classedDf)

    if opts.verbose: print('Concatenated normal and malicious data, total of %s rows' % len(enhancedDf.index))

    #construct some vectorizers based on the data in the DF. We need to vectorize future log files the exact same way so we
    # will be saving these vectorizers to a file.
    vectorizers = build_vectorizers(enhancedDf, max_features=opts.maxfeaturesperbag, verbose=opts.verbose, ngram_features=[], bow_features=['method','status_code','browser_string'])

    #use the vectorizers to featureize our DF into a numeric feature dataframe
    featureMatrix = featureize(enhancedDf, vectorizers, verbose=opts.verbose)

    #add the class column back in (it wasn't featurized by itself)
    featureMatrix['class'] = enhancedDf['class']

    #randomly assign 3/4 of the feature df to training and 1/4 to test
    featureMatrix['is_train'] = np.random.uniform(0, 1, len(featureMatrix)) <= .75

    #split out the train and test df's into separate objects
    train, test = featureMatrix[featureMatrix['is_train']==True], featureMatrix[featureMatrix['is_train']==False]

    #drop the is_train column, we don't need it anymore
    train = train.drop('is_train', axis=1)
    test = test.drop('is_train', axis=1)

    #create the isolation forest class and factorize the class column
    clf = IsolationForest(n_estimators=opts.numtrees)


    #train the isolation forest on the training set, dropping the class column (since the trainer takes that as a separate argument)
    print('\nTraining')
    clf.fit(train.drop('class', axis=1))

    #remove the 'answers' from the test set
    testnoclass = test.drop('class', axis=1)

    print('\nPredicting (class 1 is normal, class -1 is malicious)')

    #evaluate our results on the test set.
    test.is_copy = False
    test['prediction'] = clf.predict(testnoclass)
    print

    #group by class (the real answers) and prediction (what the forest said). we want these values to match for 'good' answers
    results=test.groupby(['class', 'prediction'])
    resultsagg = results.size()
    print(resultsagg)

    tp = float(resultsagg[-1,-1]) if (-1,-1) in resultsagg.index else 0
    fp = float(resultsagg[1,-1]) if (1,-1) in resultsagg.index else 0
    fn = float(resultsagg[-1,1]) if (-1,1) in resultsagg.index else 0
    f1 = 2*tp/(2*tp + fp + fn)
    print("F1 = %s" % f1)

    #save the vectorizers and trained RF file
    joblib.dump(vectorizers, opts.vectorizerfile)
    joblib.dump(clf, opts.forestfile)
