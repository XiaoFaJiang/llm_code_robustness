from datetime import datetime
from functools import wraps
from math import floor
import os
import sys
import threading
import time

from .exception import RateLimitException


# 基于ratelimit改造


def now():
    '''
    Use monotonic time if available, otherwise fall back to the system clock.

    :return: Time function.
    :rtype: function
    '''
    if hasattr(time, 'monotonic'):
        return time.monotonic
    return time.time


class PeakRateLimitDecorator(object):
    def __init__(self, calls=15, peak_calls=None, period=900, clock=now(), raise_on_limit=True):
        '''
        自定义高峰期限流装饰器

        :param int calls: Maximum function invocations allowed within a time period.
        :param int peak_calls: Maximum function invocations allowed within a peak period.
        :param float period: An upper bound time period (in seconds) before the rate limit resets.
        :param function clock: An optional function retuning the current time.
        :param bool raise_on_limit: A boolean allowing the caller to avoiding rasing an exception.
        '''
        self.clamped_calls = max(1, min(sys.maxsize, floor(calls)))
        self.peak_clamped_calls = self.clamped_calls if peak_calls is None else max(
            1, min(sys.maxsize, floor(peak_calls)))
        self.period = period
        self.clock = clock
        self.raise_on_limit = raise_on_limit
        self.disable_peak_rate_limit = False
        if os.getenv("DISABLE_PEAK_RATE_LIMIT", "false").lower() == "true":
            self.disable_peak_rate_limit = True
        print(f"The Status of toggle disable_peak_rate_limit is {self.disable_peak_rate_limit}")

        # Initialise the decorator state.
        self.last_reset = clock()
        self.num_calls = 0

        # Add thread safety.
        self.lock = threading.RLock()

    def __call__(self, func):
        '''
        返回一个装饰器方法
        '''
        @wraps(func)
        def wrapper(*args, **kargs):
            '''
            从环境变量中获取DISABLE_PEAK_RATE_LIMIT变量判断是否需要进行限流。
            若方法调用次数超过高峰期/低峰期限流阈值，则抛出自定义异常。
            结合sleep_and_retry装饰器，若超出限流阈值则等待下一个时间窗口重试。

            :param args: non-keyword variable length argument list to the decorated function.
            :param kargs: keyworded variable length argument list to the decorated function.
            :raises: RateLimitException
            '''

            with self.lock:
                if self.disable_peak_rate_limit:
                    return func(*args, **kargs)

                period_remaining = self.__period_remaining()

                # If the time window has elapsed then reset.
                if period_remaining <= 0:
                    self.num_calls = 0
                    self.last_reset = self.clock()

                # 调用计数+1
                self.num_calls += 1

                # 超出限流阈值抛异常
                threshold: int = self.peak_clamped_calls if self.__determine_peak_period() else self.clamped_calls
                if self.num_calls > threshold:
                    print(f"Trigger peak rate limiting, threshold: {threshold}")
                    if self.raise_on_limit:
                        raise RateLimitException('too many calls', period_remaining)
                    return

            return func(*args, **kargs)

        return wrapper

    def __period_remaining(self):
        '''
        返回当前限流窗口内的剩余时间

        :return: The remaing period.
        :rtype: float
        '''
        elapsed = self.clock() - self.last_reset
        return self.period - elapsed

    def __determine_peak_period(self):
        '''
        判断当前是否处于高峰期内
        TODO 自定义多个高峰期时间段

        :return: true/false
        :rtype: bool
        '''
        now = datetime.now().time()

        # 定义高峰期的开始和结束时间
        start = datetime.strptime("07:00:00", "%H:%M:%S").time()
        end = datetime.strptime("22:00:00", "%H:%M:%S").time()

        return start <= now <= end


def sleep_and_retry(func):
    '''
    Return a wrapped function that rescues rate limit exceptions, sleeping the
    current thread until rate limit resets.

    :param function func: The function to decorate.
    :return: Decorated function.
    :rtype: function
    '''
    @wraps(func)
    def wrapper(*args, **kargs):
        '''
        Call the rate limited function. If the function raises a rate limit
        exception sleep for the remaing time period and retry the function.

        :param args: non-keyword variable length argument list to the decorated function.
        :param kargs: keyworded variable length argument list to the decorated function.
        '''
        while True:
            try:
                return func(*args, **kargs)
            except RateLimitException as exception:
                time.sleep(exception.period_remaining)
    return wrapper