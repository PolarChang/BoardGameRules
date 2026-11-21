import pytest
from src.editorial_answer_contract import render_answer
from src.editorial_answer_contract import source_evidence


def test_keeps_conclusion_reason_and_conditional_scope():
    value = {'answer': '不會因為該標記而跳過生產。',
             'rule_explanation': '生產開始時，第3級工資公司的罷工標記會被棄掉。',
             'scope_limit': '這段規則並未保證其他生產條件已符合。', 'cannot_determine': False}
    assert render_answer(value).splitlines() == [value['answer'], value['rule_explanation'], value['scope_limit']]


@pytest.mark.parametrize('reason', ['', '  ', None])
def test_answer_without_rule_reason_cannot_be_rendered_as_supported(reason):
    with pytest.raises(ValueError):
        render_answer({'answer': '可以。', 'rule_explanation': reason, 'scope_limit': '', 'cannot_determine': False})


def test_refusal_does_not_require_inventing_an_explanation():
    assert render_answer({'answer': '目前證據不足。', 'rule_explanation': '', 'scope_limit': '', 'cannot_determine': True}) == '目前證據不足。'


def test_original_quote_is_attached_by_source_id_without_model_transcription():
    selected = {'R': {'fields': {'原文': 'Pay 5 coins.\n', '內容': '翻譯'}}}
    assert source_evidence(['R', 'R'], selected) == [{'id': 'R', 'quote': 'Pay 5 coins.\n'}]
    assert selected['R']['fields']['原文'] == 'Pay 5 coins.\n'


def test_unknown_source_and_translation_only_source_are_rejected():
    with pytest.raises(ValueError):
        source_evidence(['OTHER'], {'R': {'fields': {'原文': 'Pay 5 coins.'}}})
    with pytest.raises(ValueError):
        source_evidence(['R'], {'R': {'fields': {'內容': '支付五枚硬幣。'}}})
