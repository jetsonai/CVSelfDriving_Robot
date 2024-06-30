def class_indexs(class_names):
    """
    Create a dict with one random BGR color for each
    class name
    """
    idxs = list(range(1, 80))
    names = list(class_names)
    res = {}
    for name in names:
        for idx in idxs:
            res[name] = idx
            #test_values.remove(value)
            #break
            
    print(res)

def dic_test(names):
    test_keys = ["Rash", "Kil", "Varsha"]
    test_values = [1, 4, 5]
     
    # Printing original keys-value lists
    print("Original key list is : " + str(test_keys))
    print("Original value list is : " + str(test_values))
     
    # using naive method
    # to convert lists to dictionary
    res = {}
    for key in test_keys:
        for value in test_values:
            res[key] = value
            test_values.remove(value)
            break
            
    print(res)

name_file = './coco.names'
f = open(name_file,'r')
class_names = f.read().splitlines

print(class_names)
#class_colors = class_indexs(class_names)
f.close()
#class_indexs(class_names)
dic_test(class_names)

#print(class_colors)

