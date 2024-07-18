import asyncio
from time import time


async def main():
    print(f"time is :{time()}")
    await asyncio.sleep(2)
    print(f"time is :{time()} seconds")


async def count():
    print("one")
    await asyncio.sleep(1)
    print("two")


async def main1():
    await asyncio.gather(count(), count(), count())


if __name__ == "__main__":
    asyncio.run(main1())
