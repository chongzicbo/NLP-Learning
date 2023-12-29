# -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: wrapper.py
@time: 2022/6/28 11:14
"""
import logging


def use_logging(func):
    def wrapper():
        logging.warn("%s is running" % func.__name__)
        return func()

    return wrapper


@use_logging
def foo():
    print("i am foo")


# foo = use_logging(foo)
foo()


def use_logging(level):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if level == "warn":
                logging.warn("%s is running" % func.__name__)
            elif level == "info":
                logging.info("%s is running" % func.__name__)
            return func(*args)

        return wrapper

    return decorator


@use_logging(level="warn")
def foo(name="foo"):
    print("i am %s" % name)


foo()


class Foo(object):
    def __init__(self, func):
        self._func = func

    def __call__(self):
        print("class decorator runing")
        self._func()
        print("class decorator ending")


@Foo
def bar():
    print("bar")


bar()


def use_logging(func):
    def wrapper(name):
        logging.warn("%s is running" % func.__name__)
        return func(name)

    return wrapper


@use_logging
def foo(name):
    print("hhhhh i am %s " % name)


foo("hahahahaha")
