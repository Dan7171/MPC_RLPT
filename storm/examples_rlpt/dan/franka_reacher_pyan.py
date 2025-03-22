from os import system
import os
import pyan
from IPython.display import HTML
call_graph =pyan.create_callgraph(filenames="storm/examples_rlpt/dan/tmp.py", format="html") 
html = HTML(call_graph)
# print(call_graph)
# print(html)

if input("render?(y/n)") == "y":
    os.system("dot -Tsvg myuses.dot >myuses.svg")