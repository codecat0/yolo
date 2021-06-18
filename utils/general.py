"""
@File : general.py
@Author : CodeCat
@Time : 2021/6/18 下午5:37
"""
import math


def make_divisible(x, divisor):
    return math.ceil(x / divisor) * divisor
