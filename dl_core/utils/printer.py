import time

def clear_line(SIZE=200, flush=True):
    """
    Method to clear current line with empty spaces

    """
    c = '\r'.join([' '] * 200)
    print(c, flush=flush, end=' ')

def printd(s, ljust=80, flush=True):
    """
    Method to print output with timestamp

    """
    t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) 
    s = '\r[ %s ] %s' % (t, s)
    print(s.ljust(ljust), flush=flush)

def printr(s, ljust=80, flush=True):
    """
    Method to print output with timestamp + continuous refresh on same line

    """
    t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) 
    s = '\r[ %s ] %s' % (t, s)
    print(s.ljust(ljust), flush=flush, end=' ')

def printp(s, progress, pattern='%0.3f', SIZE=20, ljust=80, flush=True):
    """
    Method to print output with timestamp + progress bar 

    |========.....| 64.2l3% : Awaiting download...

    :params

      (float) progress

    """
    t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) 
    a = ''.join(['='] * int(SIZE * progress))
    b = ''.join(['.'] * (SIZE - len(a) - 1))

    pattern = '\r[ %s ] |%s>%s| ' + pattern + '%% : %s'
    s = pattern % (t, a, b, progress * 100, s)
    print(s.ljust(ljust), flush=flush, end=' ')
