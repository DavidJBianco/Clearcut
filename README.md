# Clearcut
Clearcut is a tool that uses machine learning to help you focus on the log entries that really need manual review.  

## Quick Start
    % pip install sklearn sklearn-extensions pandas httpagentparser tldextract treeinterpreter
    % ./train_flows_rf.py <normal_training_data> -o <malicious_training_data>
    % ./analyze_flows.py <bro_http_log>


## More Info
See our BSidesBoston 2016 presentation, ["Getting Started with Machine Learning for Incident Detection"](https://goo.gl/UYfPau).  

