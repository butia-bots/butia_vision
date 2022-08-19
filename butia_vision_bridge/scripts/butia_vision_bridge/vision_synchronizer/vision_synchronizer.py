#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import message_filters

from butia_vision_bridge import VisionBridge

class VisionSynchronizer:
    def syncSubscribers(source_topic_dict, callback, queue_size=1, exact_time=False, slop=0.1):
        subscribers = {}
        for source, topic in source_topic_dict.values():
            subscribers[source] = message_filters.Subscriber(topic, VisionBridge.SOURCES_TYPES[source])
        
        ts = None
        if exact_time:
            ts = message_filters.TimeSynchronizer(subscribers.values(), queue_size=queue_size)
        else:
            ts = message_filters.ApproximateTimeSynchronizer(subscribers.values(), queue_size=queue_size, slop=slop)
        ts.registerCallback(callback)

        return subscribers
        