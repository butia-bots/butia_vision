#! /usr/bin/env python
import functools
import time

def load(model_id):
    def decorator_load(func):
        @functools.wraps(func)
        def wrapper_load(*args, **kwargs):
            model = kwargs['model']
            print('Loading {} model: {}...'.format(model_id, model))
            value = func(*args, **kwargs)
            print("Finished.")
            return value
        return wrapper_load
    return decorator_load

def debug(func):
    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        debug = kwargs['debug']
        if debug:
            args_repr = [repr(a) for a in args]                      
            kwargs_repr = ['{}={!r}'.format(k, v) for k, v in kwargs.items()]  
            signature = ', '.join(args_repr + kwargs_repr)          
            print('Calling {}({})'.format(func.__name__, signature))
            value = func(*args, **kwargs)
            print('{!r} returned {!r}'.format(func.__name__, value))           
        else:
            value = func(*args, **kwargs)
        return value
    return wrapper_debug 

def action(action_name):
    def decorator_action(func):
        @functools.wraps(func)
        def wrapper_action(*args, **kwargs):
            verbose = kwargs['verbose']
            if verbose:
                print('Executing {} process...'.format(action_name))
                start_time = time.time()
                value = func(*args, **kwargs)
                end_time = time.time()
                run_time = end_time - start_time
                print('Finished, {} process took {:.4f} seconds.'.format(action_name, run_time))
            else:
                value = func(*args, **kwargs)
            return value
        return wrapper_action
    return decorator_action