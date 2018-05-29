import bif_parser
import prettytable
import pydot
from IPython.core.display import Image
from bayesian.bbn import *
name = 'asia'
module_name = bif_parser.parse(name)
module = __import__(module_name)
bg = module.create_bbn()


def show_graphgiz_image(graphviz_data):
    graph = pydot.graph_from_dot_data(graphviz_data)
    graph[0].write_png('temp.png')
    return 'temp.png'

sf = bg.get_graphviz_source()
Image(filename=show_graphgiz_image(sf))

gu=make_undirected_copy(bg)
m1=make_moralized_copy(gu,bg)
s2=m1.get_graphviz_source()
Image(filename=show_graphgiz_image(s2))

cliques, elimination_ordering = triangulate(m1, priority_func)
s2 = m1.get_graphviz_source()
Image(filename=show_graphgiz_image(s2))

jt = bg.build_join_tree()
sf = jt.get_graphviz_source()
Image(filename=show_graphgiz_image(sf))

assignments = jt.assign_clusters(bg)
jt.initialize_potentials(assignments, bg)
jt.propagate()

bronc_clust=[i for i in jt.clique_nodes for v in i.variable_names if v=='bronc']
pot=bronc_clust[0].potential_tt
sum_assignments=lambda imap,tup:sum([v for k,v in imap.iteritems() for i in k if i == tup])
yes,no=[sum_assignments(pot,('bronc',i)) for i in ['yes','no']]
print 'bronc: yes ', yes/float(yes+no)," no ", no/float(yes+no)

pot2=bronc_clust[1].potential_tt
yes,no=[sum_assignments(pot2,('bronc',i)) for i in ['yes','no']]
print 'bronc: yes ', yes/float(yes+no)," no ", no/float(yes+no)
