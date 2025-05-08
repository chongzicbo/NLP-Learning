import asyncio


async def hello():
    print("Hello")
    await asyncio.sleep(1)  # 模拟I/O操作
    print("World")


# 运行事件循环
asyncio.run(hello())  # Python 3.7+


async def foo():
    await asyncio.sleep(1)
    return "Foo"


async def bar():
    await asyncio.sleep(2)
    return "Bar"


async def main():
    # 创建Task（自动调度）
    task1 = asyncio.create_task(foo())
    task2 = asyncio.create_task(bar())

    # 等待所有Task完成
    results = await asyncio.gather(task1, task2)
    print(results)  # ['Foo', 'Bar']


asyncio.run(main())


async def simulate_io(delay, name):
    print(f"{name} 开始")
    await asyncio.sleep(delay)
    print(f"{name} 完成")


async def main():
    await asyncio.gather(
        simulate_io(1, "任务1"),
        simulate_io(2, "任务2"),
    )


asyncio.run(main())
