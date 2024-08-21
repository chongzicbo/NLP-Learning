import asyncio


async def task1():
    await asyncio.sleep(1)
    print("Task 1 completed")
    return "Result 1"


async def task2():
    await asyncio.sleep(2)
    print("Task 2 completed")
    return "Result 2"


async def main():
    results = await asyncio.gather(task1(), task2())
    print(results)


asyncio.run(main())
