import time

def verbose(func):

    def wrapper(*args, **kwargs):

        verbose = kwargs.pop('verbose') if 'verbose' in kwargs else True
        if verbose:
            func(*args, **kwargs)

    return wrapper

@verbose
def clear_line(SIZE=200, flush=False):
    """
    Method to clear current line with empty spaces

    """
    c = '\r'.join([' '] * 200)
    print(c, flush=flush, end=' ')

@verbose
def printd(s, ljust=80, flush=False):
    """
    Method to print output with timestamp

    """
    t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) 
    s = '\r[ %s ] %s' % (t, s)
    print(s.ljust(ljust), flush=flush)

@verbose
def printr(s, ljust=80, flush=False):
    """
    Method to print output with timestamp + continuous refresh on same line

    """
    t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) 
    s = '\r[ %s ] %s' % (t, s)
    print(s.ljust(ljust), flush=flush, end=' ')

@verbose
def printp(s, progress, pattern='%0.3f', SIZE=20, ljust=80, flush=False):
    """
    Method to print output with timestamp + progress bar 

    [========>.....] 64.2l3% : Awaiting download...

    :params

      (float) progress

    """
    t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) 
    a = ''.join(['='] * int(SIZE * progress))
    b = ''.join(['.'] * (SIZE - len(a) - 1))
    c = '>' if progress < 1 else ''

    pattern = '\r[ %s ] [%s%s%s] ' + pattern + '%% : %s'
    s = pattern % (t, a, c, b, progress * 100, s)
    print(s.ljust(ljust), flush=flush, end=' ')
