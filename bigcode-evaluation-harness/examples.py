def is_subset(l,s):
    sub_set = False
    if s == []:
        sub_set = True
    elif s == l:
        sub_set = True
    elif len(s) > len(l):
        sub_set = False
    ...


def is_subset(set,set_item):
    if set_item in set:
        return True
    return False