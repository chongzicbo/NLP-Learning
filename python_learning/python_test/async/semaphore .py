import asyncio


async def worker(semaphore, worker_id):
    async with semaphore:
        print(f"Worker {worker_id} is working")
        await asyncio.sleep(1)
        print(f"Worker {worker_id} has finished")


async def main():
    semaphore = asyncio.Semaphore(3)  # Limit concurrency to 3
    tasks = [worker(semaphore, i) for i in range(10)]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
