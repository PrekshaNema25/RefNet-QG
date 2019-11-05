from tokenizer.ptbtokenizer import PTBTokenizer


def test_tokenize():
    t = PTBTokenizer()
    tokens = t.tokenize(dict(id1=[dict(caption="Is this a good question?"),
                                  dict(caption="Is this a better question?")],
                             id2=[dict(caption="How's this question?")]))
    assert tokens == dict(id1=['is this a good question',
                               'is this a better question'],
                          id2=['how \'s this question'])
