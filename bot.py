import asyncio 

  

from aiogram import Bot, Dispatcher 

from aiogram.filters import CommandStart 

from aiogram.types import Message 

  

import config 

from rag import answer, source_name 

  

dp = Dispatcher() 

  

  

@dp.message(CommandStart()) 

async def start(message: Message): 

    await message.answer( 

        "Привет! Я помогаю распознать финансовых мошенников и подсказываю, что делать, " 

        "по официальным материалам Банка России и банков. Опишите ситуацию или задайте вопрос." 

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