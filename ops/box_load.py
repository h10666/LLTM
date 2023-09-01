import torch
import os
import numpy as np
    

def box_deal(coord_nr_frames, num_boxes, coord_frame_list, video_data, object_set, 
             scale_resize_w, scale_resize_h, scale_crop_w, scale_crop_h,
             offset_w, offset_h, crop_w, crop_h):

    box_tensors = torch.zeros((coord_nr_frames, num_boxes, 4), dtype=torch.float32)  # (cx, cy, w, h)
    box_categories = torch.zeros((coord_nr_frames, num_boxes))
    for frame_index, frame_id in enumerate(coord_frame_list):
        try:
            frame_data = video_data[frame_id]
        except:
            frame_data = {'labels': []}
        for box_data in frame_data['labels']:
            standard_category = box_data['standard_category']
            global_box_id = object_set.index(standard_category)

            box_coord = box_data['box2d']
            x0, y0, x1, y1 = box_coord['x1'], box_coord['y1'], box_coord['x2'], box_coord['y2']
            # print('x0, y0, x1, y1:\t', x0, y0, x1, y1)

            # scaling due to initial resize
            x0, x1 = x0 * scale_resize_w, x1 * scale_resize_w
            y0, y1 = y0 * scale_resize_h, y1 * scale_resize_h

            # shift
            x0, x1 = x0 - offset_w, x1 - offset_w
            y0, y1 = y0 - offset_h, y1 - offset_h

            x0, x1 = np.clip([x0, x1], a_min=0, a_max=crop_w-1)
            y0, y1 = np.clip([y0, y1], a_min=0, a_max=crop_h-1)

            # scaling due to crop
            x0, x1 = x0 * scale_crop_w, x1 * scale_crop_w
            y0, y1 = y0 * scale_crop_h, y1 * scale_crop_h

            # precaution
            x0, x1 = np.clip([x0, x1], a_min=0, a_max=223)
            y0, y1 = np.clip([y0, y1], a_min=0, a_max=223)

            # (cx, cy, w, h)
            gt_box = np.array([(x0 + x1) / 2., (y0 + y1) / 2., x1 - x0, y1 - y0], dtype=np.float32)

            # normalize gt_box into [0, 1]
            gt_box /= 224

            # load box into tensor
            try:
                box_tensors[frame_index, global_box_id] = torch.tensor(gt_box).float()
            except:
                pass
            # load box category
            try:
                box_categories[frame_index, global_box_id] = 1 if box_data['standard_category'] in ['hand', 'person'] else 2 # 0 is for none
            except:
                pass

            # load image into tensor
            x0, y0, x1, y1 = list(map(int, [x0, y0, x1, y1]))
    return box_tensors, box_categories