from bug2code.localization.tokenize import tokenize


def test_tokenize_splits_camel_case():
    tokens = tokenize("NullPointerException")
    assert tokens[0] == "nullpointerexception"
    assert {"null", "pointer", "exception"} <= set(tokens)


def test_tokenize_splits_snake_case():
    tokens = tokenize("max_chunks_per_file")
    assert {"max", "chunks", "per", "file"} <= set(tokens)


def test_tokenize_keeps_plain_words_once():
    assert tokenize("parse query") == ["parse", "query"]


def test_tokenize_ignores_punctuation_and_numbers_only_tokens():
    tokens = tokenize("foo.bar(1, 2);")
    assert "foo" in tokens
    assert "bar" in tokens
    assert all(not t.isdigit() for t in tokens)
