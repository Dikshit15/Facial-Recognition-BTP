# import os

# def rename_files(folder):
#     i = 0
#     for filename in os.listdir(folder):
#         dst ="image_side_"+ str(i) + ".png"
#         src =os.path.join(folder, filename)
#         dst =os.path.join(folder,dst)
#         os.rename(src, dst)
#         i += 1


# folders = ["abhishek", "dikshit", "garvit", "jatin", "neelansh" , "saket"]
# for folder in folders:
#     rename_files(folder)



#         

import os

def rename_files(folder):
    i = 0
    for filename in os.listdir(folder):
        dst = "image_side_"+ str(i) + ".png"
        src =os.path.join(folder, filename)
        dst = =os.path.join(folder,dst)
        os.rename(src,dst)
        i+=1

folders = ["abhishek", "dikshit", "garvit", "jatin", "neelansh" , "saket"]
for folder in folders:
    rename_files(folder)