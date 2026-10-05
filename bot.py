import asyncio 

  

from aiogram import Bot, Dispatcher 

from aiogram.filters import CommandStart 

from aiogram.types import Message 

  

import config 

from rag import answer, source_name

import csv
import os
from datetime import datetime

from rag import NO_ANSWER

LOG_PATH = "logs/questions.csv"


  

dp = Dispatcher() 

def log_question(question: str, result: dict):
    """Append an anonymous question record to the log."""
    os.makedirs("logs", exist_ok=True)
    is_new = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        if is_new:
            writer.writerow(["time", "question", "answered", "top_score"])
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            question,
            result["answer"] != NO_ANSWER,
            result["top_score"],
        ]) 

  

@dp.message(CommandStart()) 

async def start(message: Message): 

    await message.answer( 

        "Привет! Я помогаю распознать финансовых мошенников и подсказываю, что делать, " 

        "по официальным материалам Банка России и банков. Опишите ситуацию или задайте вопрос."

        "Вопросы сохраняются анонимно, чтобы улучшать бота." 

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