import bot


def test_card_grouped_by_spaces():
    assert bot.mask_sensitive("4276 1234 5678 9012") == "[КАРТА]"


def test_card_digits_only():
    assert bot.mask_sensitive("4276123456789012") == "[КАРТА]"


def test_phone_with_plus_and_separators():
    assert bot.mask_sensitive("+7 (912) 345-67-89") == "[ТЕЛЕФОН]"


def test_phone_starting_with_eight():
    assert bot.mask_sensitive("89123456789") == "[ТЕЛЕФОН]"


def test_sms_code_is_generic_number():
    assert bot.mask_sensitive("код 123456") == "код [ЧИСЛО]"


def test_short_numbers_stay():
    assert bot.mask_sensitive("2 умножить на 2") == "2 умножить на 2"
    assert bot.mask_sensitive("в 2024 году") == "в 2024 году"
    assert bot.mask_sensitive("100 рублей") == "100 рублей"


def test_mask_keeps_surrounding_words():
    text = "карта 4276-1234-5678-9012, звоните +7 (912) 345-67-89, код 123456"
    assert bot.mask_sensitive(text) == "карта [КАРТА], звоните [ТЕЛЕФОН], код [ЧИСЛО]"


def test_log_writes_masked_question(tmp_path, monkeypatch):
    log_path = tmp_path / "questions.csv"
    monkeypatch.setattr(bot, "LOG_PATH", str(log_path))
    raw = "моя карта 4276 1234 5678 9012 и телефон +7 (912) 345-67-89"
    bot.log_question(raw, {"answer": "ответ", "top_score": 0.2})
    stored = log_path.read_text(encoding="utf-8-sig")
    assert "4276" not in stored
    assert "912" not in stored
    assert "[КАРТА]" in stored
    assert "[ТЕЛЕФОН]" in stored
    assert raw.startswith("моя карта 4276")
