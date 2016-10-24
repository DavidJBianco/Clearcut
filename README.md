# Clearcut
Clearcut is a tool that uses machine learning to help you focus on the log entries that really need manual review. This is a beta branch of Clearcut with iforest support. It requires the beta scikit-learn 0.18.0 to be installed.

## Prereqs

You need a few libraries installed for this to work.

    % sudo pip install scikit-learn
    % pip install sklearn-extensions pandas httpagentparser tldextract treeinterpreter


## Quick Start: random forest mode
    % ./train_flows_rf.py <normal_training_data> -o <malicious_training_data>
    % ./analyze_flows.py <bro_http_log>

## Quick Start:iforest mode.
    % ./train_flows_iforest.py <normal_training_data> -o <malicious_training_data> 
    % ./analyze_flows.py <bro_http_log>


## More Info
See our BSidesDC 2016 presentation, ["Practical Cyborgism: Getting Started with Machine Learning for Incident Detection"](https://speakerdeck.com/davidjbianco/practical-cyborgism-getting-started-with-machine-learning-for-incident-detection).  

