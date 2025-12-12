from typing import Generator
from tree_sitter import Language, Parser, Tree, Node
import pandas as pd

LANGUAGE = Language('my-languages.so','python')
parser = Parser()
parser.set_language(LANGUAGE)

def traverse_tree(tree: Tree) -> Generator[Node, None, None]:
    cursor = tree.walk()

    visited_children = False
    while True:
        if not visited_children:
            yield cursor.node
            if not cursor.goto_first_child():
                visited_children = True
        elif cursor.goto_next_sibling():
            visited_children = False
        elif not cursor.goto_parent():
            break

df = pd.read_excel("mbpp_test.xlsx")
print(df['code'].iloc[0])
code = df['code'].iloc[0]
tree = parser.parse(bytes(code, 'utf8'))
tree.root_node
for node in traverse_tree(tree):
    print(node)
    print(node.text.decode("utf-8"))
    print(node.end_byte-node.start_byte)