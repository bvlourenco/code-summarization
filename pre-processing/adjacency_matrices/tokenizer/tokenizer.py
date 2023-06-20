def tokenize_code_snippet(code_snippet, code_tokenizer):
    instructions_tokens_start_end = [(0, 1)]
    code_lines = code_snippet.split('\n')
    start = 1
    for line in code_lines:
        if line.startswith('def'):
            continue
        num_tokens_instructions = len(code_tokenizer.tokenize(line).words())
        instructions_tokens_start_end.append(
            (start, start + num_tokens_instructions))
        start += num_tokens_instructions
    instructions_tokens_start_end.append((start, start + 1))
    return instructions_tokens_start_end