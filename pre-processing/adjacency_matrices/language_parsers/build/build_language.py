from tree_sitter import Language


def build_language_library():
    Language.build_library(
        # Store the library in the `languages.so` file
        'language_parsers/build/languages.so',

        # Include one or more languages
        [
            'language_parsers/parsers/tree-sitter-go',
            'language_parsers/parsers/tree-sitter-java',
            'language_parsers/parsers/tree-sitter-javascript',
            'language_parsers/parsers/tree-sitter-php',
            'language_parsers/parsers/tree-sitter-python',
            'language_parsers/parsers/tree-sitter-ruby'
        ]
    )
