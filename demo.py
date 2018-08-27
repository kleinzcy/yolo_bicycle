#!/usr/bin/env python
# coding: utf-8

from tracker import MultiTracker


object_tracker = MultiTracker('KCF')
object_tracker.tracking(num=30,filename=\"videos/riding2.mp4\",detect_way=2)
