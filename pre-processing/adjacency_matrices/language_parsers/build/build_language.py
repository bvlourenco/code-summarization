from tree_sitter import Language


def build_language_library():
    Language.build_library(
        # Store the library in the `languages.so` file
        'languages.so',

        # Include one or more languages
        [
            '../parsers/tree-sitter-go',
            '../parsers/tree-sitter-java',
            '../parsers/tree-sitter-javascript',
            '../parsers/tree-sitter-php',
            '../parsers/tree-sitter-python',
            '../parsers/tree-sitter-ruby'
        ]
    )
