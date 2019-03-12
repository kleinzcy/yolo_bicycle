# coding: utf-8

# calculate overlap ratio.

"""计算多个box的重合度，返回重合度大的人车组合。
    参数：人的bbox，车的bbox
    返回值：返回一个列表，包含多过重合度大的人车组合，其中人在车后面"""
# 这里假设系统启动的时候，路口没有单独的车。
def overlap(bbox_person, bbox_bicycle):
    bbox = []
    ratioes = []
    #找到与车重合度最大的人
    for bbox1 in bbox_bicycle:
        max_overlap = -1
        max_person = None
        bbox.append(bbox_transform(bbox1))
        
        for bbox2 in bbox_person:
            overlap = intersect(bbox1, bbox2)
            if overlap > max_overlap:
                max_overlap = overlap
                max_person = bbox2
                
        ratioes.append(max_overlap)

        bbox.append(bbox_transform(max_person))
    #print(ratioes)
    return bbox


"""判断两个矩形是否相交,若相交，则返回重合度，否则返回false"""
def intersect(bbox1,bbox2):
    x01, y01, x02, y02 = bbox1
    x11, y11, x12, y12 = bbox2
    lx = abs((x01 + x02) / 2 - (x11 + x12) / 2)
    sax = abs(x01 - x02)
    sbx = abs(x11 - x12)
    if lx <= (sax + sbx) / 2:
        return calculate_overlap(bbox1,bbox2)
    else:
        return False

# question:为什么这里反复判断是否为None是应对跟踪失败吗？
"""计算两个矩形框的重合度"""
def calculate_overlap(bbox1,bbox2):
    if bbox1 is not None and bbox2 is not None:
        x01, y01, x02, y02 = bbox1
        x11, y11, x12, y12 = bbox2
        col=min(x02,x12)-max(x01,x11)
        row=min(y02,y12)-max(y01,y11)
        intersection=col*row
        area1=(x02-x01)*(y02-y01)
        area2=(x12-x11)*(y12-y11)
        coincide=intersection/(area1+area2-intersection)
        return coincide
    else:
        return None


"""转换bbox的形式，以衔接object_detection和tracking。因为前者输出左上与右下坐标，后者需要左上坐标与长宽"""
def bbox_transform(bbox):
    if bbox is not None:
        new_bbox = (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])
        return new_bbox
    else:
        return None
