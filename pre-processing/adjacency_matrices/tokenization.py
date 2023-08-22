import re


def tokenize_snake_camel_case(token):
    camelCase_tokens = tokenize_with_camel_case(token)
    subtokens = []
    for subtoken in camelCase_tokens:
        subtokens_snake = tokenize_with_snake_case(subtoken)
        if len(subtokens_snake) > 1:
            for subtoken_snake in subtokens_snake:
                if subtoken_snake != '':
                    subtokens.append(subtoken_snake)
        else:
            subtokens.append(subtoken)
    return subtokens


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
