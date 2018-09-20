#! /usr/bin/env python
import functools
import time

def load(lib_name):
    def decorator_load(func):
        @functools.wraps(func)
        def wrapper_load(*args, **kwargs):
            model = (**kwargs)['model']
            print(f'Loading {lib_name} model: {model}...')
            value = func(*args, **kwargs)
            print("Finished.")
            return value
        return wrapper
    return decorator_load

def debug(func):
    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        debug = (**kwargs)['debug']
        if debug:
            args_repr = [repr(a) for a in args]                      
            kwargs_repr = [f'{k}={v!r}' for k, v in kwargs.items()]  
            signature = ', '.join(args_repr + kwargs_repr)          
            print(f'Calling {func.__name__}({signature})')
            value = func(*args, **kwargs)
            print(f'{func.__name__!r} returned {value!r}')           
        else:
            value = func(*args, **kwargs)
        return value
    return wrapper_debug 

def action(action_name):
    def decorator_action(func):
        @functools.wraps(func)
        def wrapper_action(*args, **kwargs):
            verbose = (**kwargs)['verbose']
            if verbose:
                print(f'Executing {action_name} process...')
                start_time = time.perf_counter()
                value = func(*args, **kwargs)
                end_time = time.perf_counter()
                run_time = end_time - start_time
                print(f'Finished, {action_name} process took {run_time:.4f} seconds.')
            else:
                value = func(*args, **kwargs)
            return value
        return wrapper_action
    return decorator_action