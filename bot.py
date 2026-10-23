import asyncio
import csv
import os
import re
from datetime import datetime

from aiogram import Bot, Dispatcher
from aiogram.filters import CommandStart
from aiogram.types import Message

import config
from rag import NO_ANSWER, answer, source_name

LOG_PATH = "logs/questions.csv"

# Cards are 13–19 digits, plain or split by spaces/hyphens.
# Russian phones are 11 digits starting with 7/8, or 10 national digits,
# with an optional plus, spaces, hyphens and parentheses.
# Any other run of 6+ digits (SMS codes, account numbers) is masked too.
# Shorter numbers — a year, a sum, «2 умножить на 2» — stay as written.
_SENSITIVE = re.compile(r"(?<![\d+])\+?\d(?:[ \-()]*\d){5,}")
_CARD = re.compile(r"\d+(?:[ \-]+\d+)*")
_PHONE_11 = re.compile(r"(?:\+7|7|8)[ \-]*\(?\d{3}\)?[ \-]*\d{3}[ \-]*\d{2}[ \-]*\d{2}")
_PHONE_10 = re.compile(r"\(?\d{3}\)?[ \-]*\d{3}[ \-]*\d{2}[ \-]*\d{2}")


def mask_sensitive(text: str) -> str:
    """Mask cards, phones and long digit runs. The model still sees the original text."""

    def replace(match: re.Match[str]) -> str:
        raw = match.group(0)
        digits = re.sub(r"\D", "", raw)
        count = len(digits)
        if count == 11 and digits[0] in "78" and _PHONE_11.fullmatch(raw):
            return "[ТЕЛЕФОН]"
        if count == 10 and _PHONE_10.fullmatch(raw):
            return "[ТЕЛЕФОН]"
        if 13 <= count <= 19 and _CARD.fullmatch(raw):
            return "[КАРТА]"
        if count >= 6:
            return "[ЧИСЛО]"
        return raw

    return _SENSITIVE.sub(replace, text)


dp = Dispatcher()


def log_question(question: str, result: dict):
    """Append one question row. Only the copy written to disk is masked."""
    os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
    is_new = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        if is_new:
            writer.writerow(["time", "question", "answered", "top_score"])
        writer.writerow(
            [
                datetime.now().isoformat(timespec="seconds"),
                mask_sensitive(question),
                result["answer"] != NO_ANSWER,
                result["top_score"],
            ]
        )


@dp.message(CommandStart())
async def start(message: Message):
    await message.answer(
        "Привет! Я помогаю распознать финансовых мошенников и подсказываю, что делать, "
        "по официальным материалам Банка России. Опишите ситуацию или задайте вопрос. "
        "Вопросы сохраняются без привязки к аккаунту, цифры номеров карт и телефонов скрываются, "
        "чтобы улучшать бота."
    )


@dp.message()
async def handle_question(message: Message):
    if not message.text:
        return
    await message.answer("Ищу ответ в документах…")
    try:
        result = await asyncio.to_thread(answer, message.text)
    except Exception as e:
        print("Error:", e)
        await message.answer("Не получилось ответить, попробуйте чуть позже.")
        return

    log_question(message.text, result)

    sources = "\n".join(source_name(d) for d in result["cited"])
    reply = result["answer"]
    if sources:
        reply += f"\n\nИсточники:\n{sources}"
    await message.answer(reply)


async def main():
    bot = Bot(token=config.TELEGRAM_TOKEN)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
