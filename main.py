import asyncio, aiofiles
from motor.motor_asyncio import AsyncIOMotorClient

from beanie import Document, Indexed, init_beanie
import os
import time
import logging
from watchdog.observers import Observer
from watchdog.events import LoggingEventHandler, PatternMatchingEventHandler
from pymongo.errors import DuplicateKeyError


def validate_csv(record):
    if record.count("\"") != 2:
        return False
    record = record.split("\"")
    return record[0].count(",") == 13


class Record(Document):
    transaction: Indexed(str, unique=True)
    date: str
    card: str
    client: str
    date_of_birth: str
    passport: str
    passport_valid_to: str
    phone: str
    operation_type: str
    amount: str
    operation_result: str
    terminal_type: str
    city: str
    address: str

class Test(Document):
    id: Indexed(int)
    text: str


async def test(path):
    client = AsyncIOMotorClient("mongodb://localhost:27017")
    await init_beanie(database=client.transactions, document_models=[Record])

    async with aiofiles.open(path, mode='r', encoding='utf-8') as f:
        async for record in f: 
            if validate_csv(record):
                _record = record.split("\"")[0].split(",")
                new_record = Record(
                        transaction=_record[0],
                        date=_record[1],
                        card=_record[2],
                        client=_record[3],
                        date_of_birth=_record[4],
                        passport=_record[5],
                        passport_valid_to=_record[6],
                        phone=_record[7],
                        operation_type=_record[8],
                        amount=_record[9],
                        operation_result=_record[10],
                        terminal_type=_record[11],
                        city=_record[12],
                        address=record.split("\"")[1]
                    )
                try:
                    await new_record.insert()
                except DuplicateKeyError:
                    print("Ошибка первичного ключа")

def on_create(event):
    path = os.path.dirname(__file__) + "\\raw_data\\" + event.src_path.split("\\")[1]
    print(f"\nПолучен файл {path}")
    if path.endswith('.csv'):
        print("Загрузка файла в базу данных...")
        asyncio.run(test(path))
    else:
        print("Ошибка: неверный формат файла")


if __name__ == "__main__":
    path = './raw_data'
    event_handler = PatternMatchingEventHandler(["*"], None, False, True)
    event_handler.on_created = on_create
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    finally:
        observer.stop()
        observer.join()