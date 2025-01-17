# convert image to pdf
import os
import img2pdf

if os.name=='nt':
    proj_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
else:
    proj_dir = os.path.dirname(os.path.realpath(__file__))

plotWarehouse = os.path.join(proj_dir,'plotWarehouse')
image_files = [i for i in os.listdir(plotWarehouse) if i.endswith(".png")]

# Sort the list by creation time (the first element of each tuple)
image_files.sort()

with open(os.path.join(proj_dir,"all_plots.pdf"), "wb") as file:
    file.write(img2pdf.convert([os.path.join(plotWarehouse,i) for i in image_files]))