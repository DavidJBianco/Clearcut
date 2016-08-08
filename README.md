# Clearcut
Clearcut is a tool that uses machine learning to help you focus on the log entries that really need manual review.  

## Quick Start
    % sudo pip install git+https://github.com/scikit-learn/scikit-learn.git#egg=scikit-learn
    % pip install sklearn-extensions pandas httpagentparser tldextract treeinterpreter
    % ./train_flows_rf.py <normal_training_data> -o <malicious_training_data>
    % ./analyze_flows.py <bro_http_log>


## More Info
See our BSidesBoston 2016 presentation, ["Getting Started with Machine Learning for Incident Detection"](https://speakerdeck.com/davidjbianco/getting-started-with-machine-learning-for-incident-detection).  

