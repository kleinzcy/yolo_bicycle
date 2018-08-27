
# coding: utf-8

# calculate overlap ratio.


"""计算多个box的重合度，返回重合度大的人车组合。
    参数：人的bbox，车的bbox,重合度的阈值
    返回值：返回一个列表，包含多过重合度大的人车组合，其中人在车后面"""
def overlap_one(bbox_person, bbox_bicycle,confidence):
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
    
    for ratioe in ratioes:
        if ratioe < confidence and ratioe is not -1:
            print("people leave bicycle!")
    print(ratioes)
    return bbox

""" 计算每个单车与每个人的重合度，若其中有一个单车与每个人的重合度过小，则视为违停
    参数：人的bbox,车的bbox,重合度的阈值
    返回值：返回一个列表，包含每个单车和每个人的bbox"""
def detect_bicycle_two(bbox_person,bbox_bicycle,confidence):
    bbox = []
    flag = True
    
    for bbox1 in bbox_bicycle:
        bbox.append(bbox_transform(bbox1))
        
        for bbox2 in bbox_person:
            bbox.append(bbox_transfrom(bbox2))
            
            result = intersect(bbox1,bbox2)
            if (result is not False and result > confidence) or result is -1:
                flag = True
            else:
                flag = False
    
    if flag is True:
        print("Nobody leaves bicycles")
    else:
        print("Somebody leaves bicycles")  

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


"""计算两个矩形框的重合度"""
def calculate_overlap(bbox1,bbox2):
    x01, y01, x02, y02 = bbox1
    x11, y11, x12, y12 = bbox2
    col=min(x02,x12)-max(x01,x11)
    row=min(y02,y12)-max(y01,y11)
    intersection=col*row
    area1=(x02-x01)*(y02-y01)
    area2=(x12-x11)*(y12-y11)
    coincide=intersection/(area1+area2-intersection)
    return coincide


"""转换bbox的形式，以衔接object_detection和tracking。因为前者输出左上与右下坐标，后者需要左上坐标与长宽"""
def bbox_transform(bbox):
    if bbox is not None:
        new_bbox = (list(bbox)[0], list(bbox)[1], list(bbox)[2] - list(bbox)[0], list(bbox)[3] - list(bbox)[1])
        return new_bbox
    else:
        return (0, 0, 0, 0)
