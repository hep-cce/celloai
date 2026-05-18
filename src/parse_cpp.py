# This file was originally part of fynnfluegge/doc-comments-ai and has been modified.
#
# The original code is licensed under the MIT License, a copy of which
# is available in the LICENSES/ directory.
#
# All modifications are licensed under the BSD-3-Clause License.

import tree_sitter
from tree_sitter_languages import get_language, get_parser

from enum import Enum

class Language(Enum):
    CPP = "cpp"
    C = "c"

class TreesitterMethodNode:
    def __init__(
        self,
        name: "str | bytes | None",
        doc_comment: "str | None",
        method_source_code: "str | None",
        node: tree_sitter.Node,
        param_count: int = 0,
        return_count: int = 0,
        list_params: list[str] = []
    ):
        self.doc_comment = doc_comment
        self.method_source_code = method_source_code or node.text.decode()
        self.node = node

        self.name = self.method_source_code.split('(')[0].split()[-1]
        
        def _count_params_and_return(self, src: str) -> tuple[int, int, list[str]]:
            list_of_params = []
            import re
            decl = src.split('{', 1)[0].strip().rstrip(';')
            l = decl.find('(')
            if l == -1: return 0, 0, list_of_params
            depth, r = 0, -1
            for i, ch in enumerate(decl[l:], start=l):
                if ch == '(': depth += 1
                elif ch == ')':
                    depth -= 1
                    if depth == 0: r = i; break
            params_str = decl[l+1:r].strip() if r != -1 else ""
            #list_of_params.append(params_str)
            if not params_str or params_str == 'void':
                param_count = 0
            else:
                parts = [p.strip() for p in params_str.split(',') if p.strip()]
                list_of_params.extend(parts)
                param_count = len(parts)

            arrow = decl.find('->', r if r != -1 else 0)
            if arrow != -1:
                ret_is_void = re.search(r'\bvoid\b', decl[arrow+2:]) is not None
                return param_count, 0, list_of_params if ret_is_void else 1

            # constructors/destructors: no return
            # name is last identifier before '('
            name = re.findall(r'([~\w:<>]+)\s*\(', decl)
            lead = decl[:l].strip()

            if "void" in lead:
                return param_count, 0, list_of_params
            elif name:
                if lead == name[0]:
                    return param_count, 0, list_of_params
                else:
                    return param_count, 1, list_of_params
            else: # struct ot class
                return 0, 0, list_of_params

        p, r, list_params = _count_params_and_return(self, self.method_source_code)
        self.param_count = p
        self.return_count = r
        self.list_params = list_params

class TreesitterRegistry:
    _registry = {}

    @classmethod
    def register_treesitter(cls, name, treesitter_class):
        cls._registry[name] = treesitter_class

    @classmethod
    def create_treesitter(cls, name: Language):
        treesitter_class = cls._registry.get(name)
        if treesitter_class:
            return treesitter_class()
        else:
            raise ValueError("Invalid tree type")

class Treesitter():
    def __init__(
        self,
        language: Language,
        method_declaration_identifier: str,
        name_identifier: str,
        doc_comment_identifier: str,
    ):
        self.parser = get_parser(language.value)
        self.language = get_language(language.value)
        self.method_declaration_identifier = method_declaration_identifier
        self.method_name_identifier = name_identifier
        self.doc_comment_identifier = doc_comment_identifier

    @staticmethod
    def create_treesitter(language: Language) -> "Treesitter":
        return TreesitterRegistry.create_treesitter(language)

    def parse(self, file_bytes: bytes) -> list[TreesitterMethodNode]:
        self.tree = self.parser.parse(file_bytes)
        result = []
        methods = self._query_all_methods(self.tree.root_node)
        list_params = []
        for method in methods:
            method_name = self._query_method_name(method["method"])
            doc_comment = method["doc_comment"]
            result.append(
                TreesitterMethodNode(method_name, doc_comment, None, method["method"], 0, 0, list_params)
            )
        classes = self._query_all_classes(self.tree.root_node)
        for classe in classes:
            class_name = self._query_class_name(classe["classe"])
            doc_comment = classe["doc_comment"]
            result.append(
                TreesitterMethodNode(class_name, doc_comment, None, classe["classe"], 0, 0, list_params)
            )
        return result

    def _query_all_classes(
        self,
        node: tree_sitter.Node,
    ):
        classes = []
        if node.type == 'class_specifier' or node.type == 'struct_specifier':
            doc_comment_node = None
            if (
                node.prev_named_sibling
                and node.prev_named_sibling.type == self.doc_comment_identifier
            ):
                doc_comment_node = node.prev_named_sibling.text.decode()
            classes.append({"classe": node, "doc_comment": doc_comment_node})
        else:
            for child in node.children:
                classes.extend(self._query_all_classes(child))
        return classes

    def _query_class_name(self, node: tree_sitter.Node):
        if node.type == 'class_specifier':
            for child in node.children:
                if child.type == 'class_specifier':
                    return child.text.decode()
        return None

    def _query_all_methods(
        self,
        node: tree_sitter.Node,
    ):
        methods = []
        if node.type == self.method_declaration_identifier:
            doc_comment_node = None
            if (
                node.prev_named_sibling
                and node.prev_named_sibling.type == self.doc_comment_identifier
            ):
                doc_comment_node = node.prev_named_sibling.text.decode()
            methods.append({"method": node, "doc_comment": doc_comment_node})
        else:
            for child in node.children:
                methods.extend(self._query_all_methods(child))
        return methods

    def _query_method_name(self, node: tree_sitter.Node):
        if node.type == self.method_declaration_identifier:
            for child in node.children:
                if child.type == self.method_name_identifier:
                    return child.text.decode()
        return None

class TreesitterCpp(Treesitter):
    def __init__(self):
        super().__init__(Language.CPP, "function_definition", "identifier", "comment")

    def _query_method_name(self, node: tree_sitter.Node):
        if node.type == self.method_declaration_identifier:
            for child in node.children:
                # if method returns pointer, skip pointer declarator
                if child.type == "pointer_declarator":
                    child = child.children[1]
                if child.type == "function_declarator":
                    for child in child.children:
                        if child.type == self.method_name_identifier:
                            return child.text.decode()
        return None

# Register the TreesitterCpp class in the registry
TreesitterRegistry.register_treesitter(Language.CPP, TreesitterCpp)
