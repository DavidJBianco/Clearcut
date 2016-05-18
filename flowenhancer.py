#!/usr/bin/env python
import httpagentparser
import tldextract
import logging
from urlparse import urlparse, parse_qs


logging.getLogger("tldextract").setLevel(logging.CRITICAL)


def enhance_flow(flowDF):
    """
      Add some useful columns to a http dataframe.

      Parameters
      ----------
      flowDF : dataframe
          The enhanced HTTP log dataframe

      Returns
      -------
      flowDF: the dataframe with some columns added

    """

    #create some useful pre-features

    #stringify the port. probably no longer needed since we defensivley stringify things elsewhere.
    flowDF['resp_p_str'] = flowDF['resp_p'].apply(str)

    #extract the browser string from the user agent.
    flowDF['browser_string'] = flowDF['user_agent'].apply(lambda agent: httpagentparser.simple_detect(agent)[1])
    
    def paramsSSV(uri):
        fullUri = 'http://bogus.com/'+uri
        parseResult = parse_qs(urlparse(fullUri).query)
        return ' '.join(parseResult.keys())

    #create a SSV of the URI parameter keys
    flowDF['URIparams'] = flowDF['uri'].apply(paramsSSV)
    
    def tokensSSV(uri):
        fullUri = 'http://bogus.com/'+uri
        parseResult = parse_qs(urlparse(fullUri).query)
        return ' '.join([" ".join(vals) for vals in parseResult.values()])

    #create a SSV of the URI parameter values
    flowDF['URItokens'] = flowDF['uri'].apply(tokensSSV)

    #extract the subdomain from the host
    flowDF['subdomain'] = flowDF['host'].apply(lambda host: tldextract.extract(host)[0])

    #extract the TLD from the host
    flowDF['tld'] = flowDF['host'].apply(lambda host: tldextract.extract(host)[1])

    return flowDF
