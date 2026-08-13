from bug2code.localization.frozen_codebert import chunk_token_ids


def test_chunk_token_ids_short_input_is_one_chunk():
    chunks = chunk_token_ids(list(range(10)), max_length=16, stride=8, max_chunks=12)
    assert chunks == [list(range(10))]


def test_chunk_token_ids_empty_input_yields_one_empty_chunk():
    assert chunk_token_ids([], max_length=16, stride=8, max_chunks=12) == [[]]


def test_chunk_token_ids_slides_with_stride_and_covers_the_tail():
    tokens = list(range(20))
    chunks = chunk_token_ids(tokens, max_length=10, stride=5, max_chunks=12)
    # inner window = max_length - 2 = 8
    assert chunks[0] == tokens[0:8]
    assert chunks[1] == tokens[5:13]
    assert chunks[-1][-1] == tokens[-1]


def test_chunk_token_ids_respects_max_chunks():
    tokens = list(range(1000))
    chunks = chunk_token_ids(tokens, max_length=10, stride=1, max_chunks=3)
    assert len(chunks) == 3
