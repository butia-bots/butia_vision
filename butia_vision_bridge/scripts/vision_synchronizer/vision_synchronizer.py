#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import message_filters

class VisionSynchronizer:
    def syncSubscribers(topic_type_dict, callback, queue_size=1, exact_time=False, slop=0.1):
        subscribers = []
        for topic, tp in topic_type_dict.values():
            subscribers.append(message_filters.Subscriber(topic, tp))
        
        ts = None
        if exact_time:
            ts = message_filters.TimeSynchronizer(subscribers, queue_size=queue_size)
        else:
            ts = message_filters.ApproximateTimeSynchronizer(subscribers, queue_size=queue_size, slop=slop)
        ts.registerCallback(callback)
        