
import httpagentparser
import tldextract
from urlparse import urlparse, parse_qs


def enhance_flow(flowDF):
    #create some useful pre-features
    flowDF['resp_p_str'] = flowDF['resp_p'].apply(str)
    flowDF['browser_string'] = flowDF['user_agent'].apply(lambda agent: httpagentparser.simple_detect(agent)[1])
    
    def paramsSSV(uri):
        fullUri = 'http://bogus.com/'+uri
        parseResult = parse_qs(urlparse(fullUri).query)
        return ' '.join(parseResult.keys())
    
    flowDF['URIparams'] = flowDF['uri'].apply(paramsSSV)
    
    def tokensSSV(uri):
        fullUri = 'http://bogus.com/'+uri
        parseResult = parse_qs(urlparse(fullUri).query)
        return ' '.join([" ".join(vals) for vals in parseResult.values()])
    
    flowDF['URItokens'] = flowDF['uri'].apply(tokensSSV)
    flowDF['subdomain'] = flowDF['host'].apply(lambda host: tldextract.extract(host)[0])
    flowDF['tld'] = flowDF['host'].apply(lambda host: tldextract.extract(host)[1])

    return flowDF