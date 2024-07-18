from pyppeteer import launch
import asyncio
import time

async def main():
    browser = await launch()
    page = await browser.newPage()
    await page.goto("https://www.ruanyifeng.com/blog/2019/11/python-asyncio.html")
    await page.screenshot({"path": "example.png"})
    await browser.close()

async def say_after(delay,what):
    await asyncio.sleep(delay)
    print(what)

async def main2():
    print(f"started at {time.strftime("%X")} seconds")    
    await say_after(1,"hello")
    await say_after(2,"world")
    print(f"finished at {time.strftime("%X")} seconds")
    
    
    
async def main3():
    task1=asyncio.create_task(say_after(1,"hello"))
    task2=asyncio.create_task(say_after(2,"world"))
    print(f"started as {time.strftime("%X")}")
    await task1
    await task2
    print(f"finished at {time.strftime("%X")} seconds")
    
asyncio.run(main3())
