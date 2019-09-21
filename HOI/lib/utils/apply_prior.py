# --------------------------------------------------------
# Tensorflow iCAN
# Licensed under The MIT License [see LICENSE for details]
# Written by Chen Gao
# --------------------------------------------------------

def apply_prior(Object, prediction):

    if Object[4] != 74: # not a book, then the action is impossible to be read
        prediction[0][7] = 0

    if (Object[4] != 41) and (Object[4] != 40) and (Object[4] != 42) and (Object[4] != 46): # not 'wine glass', 'bottle', 'cup', 'bowl', then the action is impossible to be drink
        prediction[0][4] = 0

    if Object[4] != 68: # not a cellphone, then the action is impossible to be talk_on_phone
        prediction[0][0] = 0
                            
    if (Object[4] != 14) and (Object[4] != 61) and (Object[4] != 62) and (Object[4] != 60) and (Object[4] != 58)  and (Object[4] != 57): # not 'bench', 'dining table', 'toilet', 'bed', 'couch', 'chair', then the action is impossible to be lay
        prediction[0][3] = 0

    if (Object[4] != 47) and (Object[4] != 48) and (Object[4] != 49) and (Object[4] != 50) and (Object[4] != 51) and (Object[4] != 52) and (Object[4] != 53) and (Object[4] != 54) and (Object[4] != 55) and (Object[4] != 56): # not ''banana', 'apple', 'sandwich', 'orange', 'carrot', 'broccoli', 'hot dog', 'pizza', 'cake', 'donut', then the action is impossible to be eat_obj
        prediction[0][1] = 0

    if (Object[4] != 43) and (Object[4] != 44) and (Object[4] != 45): # not 'fork', 'knife', 'spoon', then the action is impossible to be eat_instr
        prediction[0][5] = 0
                            
    if (Object[4] != 2) and (Object[4] != 4) and (Object[4] != 18) and (Object[4] != 21) and (Object[4] != 14) and (Object[4] != 57) and (Object[4] != 58) and (Object[4] != 60) and (Object[4] != 62) and (Object[4] != 61) and (Object[4] != 29) and (Object[4] != 27) and (Object[4] != 25): # not 'bicycle', 'motorcycle', 'horse', 'elephant', 'bench', 'chair', 'couch', 'bed', 'toilet', 'dining table', 'suitcase', 'handbag', 'backpack', then the action is impossible to be sit
        prediction[0][2] = 0

    if (Object[4] != 63) and (Object[4] != 64) and (Object[4] != 68):
        prediction[0][6] = 0
    return prediction