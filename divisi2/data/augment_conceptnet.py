from csc import divisi2
import codecs

GRAPH_FILE = 'graphs/conceptnet_en.graph'
GRAPH_OUT = 'graphs/conceptnet_verbs_en.graph'

def emit(c1, c2, rel, freq, score, file):
    props = {
        'rel': rel,
        'freq': freq,
        'score': score,
    }
    print (u'%s\t%s\t%s' % (c1, c2, props)).encode('utf-8')
    print >> file, u'%s\t%s\t%s' % (c1, c2, props)

def read_graph_file(filename, outfile):
    file = codecs.open(filename, encoding='utf-8')
    out = codecs.open(outfile, 'w', encoding='utf-8')
    for line in file:
        if line.strip():
            c1, c2, propstr = line.strip().split('\t')
            props = eval(propstr)
            if props['rel'] == u'CapableOf':
                freq = props['freq']
                score = props['score']
                if len(c2.split()) == 2:
                    verb, noun = c2.split()
                    emit(c1, verb, u'CapableOf', freq, score, out)
                    emit(noun, verb, u'ReceivesAction', freq, score, out)

read_graph_file(GRAPH_FILE, GRAPH_OUT)
