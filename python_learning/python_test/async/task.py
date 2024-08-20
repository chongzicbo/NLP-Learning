import asyncio
from time import time

print(time())


async def task1():
    await asyncio.sleep(1)
    print("Task 1 completed")


async def task2():
    await asyncio.sleep(2)
    print("Task 2 completed")


async def main():
    task1_task = asyncio.create_task(task1())
    task2_task = asyncio.create_task(task2())

    # 等待所有任务完成
    await task1_task
    await task2_task


asyncio.run(main())
print(time())
