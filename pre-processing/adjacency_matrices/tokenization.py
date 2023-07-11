import re


def tokenize_with_camel_case(token):
    matches = re.finditer(
        '.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', token)
    return [m.group(0) for m in matches]


def tokenize_with_snake_case(token, keep_delimiter=False):
    # The re.split keeps the underscore and places it as a list element.
    if keep_delimiter:
        return re.split(r'(_)', token)
    else:
        return token.split('_')
