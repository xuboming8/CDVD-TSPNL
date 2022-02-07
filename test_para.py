import os

def replace(f_old, f_new, filename):
    index = 0
    for line in f_old:
        index += 1
        if index == 26:
            curLine1 = line.strip().split("/")
            curLine1[-1] = filename
            new_str = "  "
            for str in curLine1:
                new_str += str
                if (str != curLine1[-1]):
                    new_str += '/'
            line = line.replace(line, new_str)
        f_new.write(line)
        if index == 26:
            f_new.write('\r')

root = os.listdir("./experiments/Deblur_test/models")
root.sort(key=lambda x: int(x[6:-4]))
for finename in root:
    # f_old = open("./options/test/Deblur/test_Deblur_BSD.yml", "r")
    # f_new = open("./options/test/Deblur/test_Deblur_BSD_" + str(finename) + ".yml", "w")
    # replace(f_old, f_new, filename=finename)

    print(finename)
    command = 'python basicsr/test.py -opt options/test/Deblur/test_Deblur_BSD_' + str(finename) + '.yml'
    print(command)
    os.system(command)
    print("finish")