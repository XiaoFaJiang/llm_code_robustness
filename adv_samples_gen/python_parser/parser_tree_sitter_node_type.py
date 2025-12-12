import json
import os
import sys

def parser_literal_type(file_path):
    node_types = []
    with open(file_path,"r") as f:
        node_types = json.loads(f.read())
    ret = []
    def dfg(node:dict):
        if node.get("type","").count("literal") > 0:
            return ret.append(node["type"])
        for child in node.get("subtypes",[]):
            dfg(child)
    for node in node_types:
        dfg(node)
    return list(set(ret))

def parser_identifier_type(file_path):
    pass


if __name__ == "__main__":
    file_path = "/home/ljc/desktop/blackbox-attack-for-code-pretrained-models/python_parser/parser_folder/tree-sitter-rust/src/node-types.json"
    print(parser_literal_type(file_path))