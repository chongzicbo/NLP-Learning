import asyncio
import aiofiles
import time


# 同步读写文件
def sync_read_write(file_paths, output_file_paths):
    for i, file_path in enumerate(file_paths):
        with open(file_path, "r") as file:
            content = file.read()
        with open(output_file_paths[i], "w") as file:
            file.write(f"Sync write to {file_path}")


# 异步读写文件
async def async_read_write(file_paths, output_file_paths):
    async def read_file(file_path):
        async with aiofiles.open(file_path, mode="r") as file:
            content = await file.read()

    async def write_file(file_path):
        async with aiofiles.open(file_path, mode="w") as file:
            await file.write(f"Async write to {file_path}")

    tasks = [read_file(file_path) for file_path in file_paths] + [
        write_file(output_file_paths[i]) for i, file_path in enumerate(file_paths)
    ]
    await asyncio.gather(*tasks)


# 测试同步和异步读写文件的性能
def main():
    import os

    input_file_dir = "/data/bocheng/code/source_code/AnimateAnyone/src/dwpose"
    input_file_paths = [
        os.path.join(input_file_dir, f)
        for f in os.listdir(input_file_dir)
        if f.endswith(".py")
    ]
    os.makedirs("output", exist_ok=True)
    output_file_paths = [
        os.path.join("./output", f"output_{i}.txt")
        for i in range(len(input_file_paths))
    ]

    # 同步读写
    start_time = time.time()
    sync_read_write(input_file_paths, output_file_paths)
    sync_time = time.time() - start_time
    print(f"Sync read/write time: {sync_time:.2f} seconds")

    # 异步读写
    start_time = time.time()
    asyncio.run(async_read_write(input_file_paths, output_file_paths))
    async_time = time.time() - start_time
    print(f"Async read/write time: {async_time:.2f} seconds")


if __name__ == "__main__":
    main()
