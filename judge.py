from overlap_ratio import intersect

"""对当前情况进行判断
                    """

def Judgement(bboxes, confidence):
    bbox_leave = []
    ratioes = []
    ratioe_num = 0
    current_num = 0
    if len(bboxes) > 0:
        bicycle_num = len(bboxes)/2
        while current_num < bicycle_num:
            result = intersect(inverse_transform(bboxes[2*(current_num)]),inverse_transform(bboxes[2*(current_num)+1]))
            current_num += 1
            ratioes.append(result)
            print(result)
            
    if len(ratioes) > 0:
        for ratioe in ratioes:
            ratioe_num += 1
            if ratioe < confidence:
                bbox_leave.append(bboxes[2*(ratioe_num-1)])
                bbox_leave.append(bboxes[2*(ratioe_num-1)+1])
        return bbox_leave
    else:
        return None

def inverse_transform(bbox):
    if bbox is not None:
        new_bbox = (bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1])
        return new_bbox
    else:
        return (0,0,0,0)
