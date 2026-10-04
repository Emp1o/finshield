     sources = "\n".join(source_name(d) for d in result["cited"])
    reply = result["answer"]
    if sources:
        reply += f"\n\nИсточники:\n{sources}"
    await message.answer(reply)