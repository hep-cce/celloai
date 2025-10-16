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
    ):
        self.name = name
        self.doc_comment = doc_comment
        self.method_source_code = method_source_code or node.text.decode()
        self.node = node

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
        for method in methods:
            method_name = self._query_method_name(method["method"])
            doc_comment = method["doc_comment"]
            result.append(
                TreesitterMethodNode(method_name, doc_comment, None, method["method"])
            )
        classes = self._query_all_classes(self.tree.root_node)
        for classe in classes:
            class_name = self._query_class_name(classe["classe"])
            doc_comment = classe["doc_comment"]
            result.append(
                TreesitterMethodNode(class_name, doc_comment, None, classe["classe"])
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
