import asyncio


class AsyncIterator:
    def __init__(self):
        self.count = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.count < 5:
            self.count += 1
            return self.count
        else:
            raise StopAsyncIteration


async def async_for_example():
    async for number in AsyncIterator():
        print(number)


# asyncio.run(async_for_example())
class AsyncContextManager:
    async def __aenter__(self):
        print("Enter context")
        return self

    async def __aexit__(self, exc_type, exc, tb):
        print("Exit context")


async def async_with_example():
    async with AsyncContextManager() as manager:
        print("Inside context")


asyncio.run(async_with_example())
