#!/usr/bin/env python
# coding: utf-8

from tracker import MultiTracker


object_tracker = MultiTracker('MOSSE')
object_tracker.tracking(num=100)
