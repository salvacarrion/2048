import time


def get_pretty_time(start_time, end_time=None, s="", divisor=1.0, max_decimals=3):
    if not end_time:
        end_time = time.time()
    hours, rem = divmod((end_time - start_time)/divisor, 3600)
    minutes, seconds = divmod(rem, 60)
    return ("{}{:0>2}:{:0>2}:{:05." + str(max_decimals) + "f}").format(s, int(hours), int(minutes), seconds)
