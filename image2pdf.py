# convert image to pdf
import os
from PyPDF2 import PdfReader, PdfWriter
from PyPDF2.generic import RectangleObject as RO

def merge_pdf(proj_dir):
    plotWarehouse = os.path.join(proj_dir,'plotWarehouse')
    pdf_filles = [i for i in os.listdir(plotWarehouse) if i.endswith(".pdf")]

    # Sort the list by creation time (the first element of each tuple)
    pdf_filles.sort()

    prefix = os.path.basename(proj_dir).split('_')[0]
    writer = PdfWriter()

    for i in range(len(pdf_filles)):
        pdf_file = os.path.join(plotWarehouse,pdf_filles[i])
        page = PdfReader(pdf_file).pages[0]

        # resize mediabox of the last two in the set
        if (i%4)>1:
            page.mediabox = RO([
                page.mediabox.lower_left[0],
                -int(float(page.mediabox.upper_right[1])/11.*3.5),
                page.mediabox.upper_right[0],
                page.mediabox.upper_right[1],
            ])
        
        writer.add_page(page)

    merge_file = os.path.join(proj_dir,f"{prefix}_all_plots.pdf")
    if os.path.exists(merge_file):
        os.remove(merge_file)
    with open(merge_file, 'wb') as f:
        writer.write(f)

def png2pdf(proj_dir):
    from PIL import Image
    margin_points = int(300*0.75)  # 0.75 inch margin

    writer = PdfWriter()

    plotWarehouse = os.path.join(proj_dir,'plotWarehouse')
    image_paths = [i for i in os.listdir(plotWarehouse) if i.endswith(".png")]
    image_paths.sort()

    for i, image_path in enumerate(image_paths):
        img = Image.open(os.path.join(plotWarehouse,image_path))

        # Convert image to PDF
        tmp_pdf_path = os.path.join(plotWarehouse,"temp.pdf")
        img.save(tmp_pdf_path, "PDF", resolution=100.0)

        # Add image PDF to the final PDF
        page = PdfReader(tmp_pdf_path).pages[0]

        # Set landscape mode for selected pages
        if (i%4)>1:
            page.mediabox = RO(
                [0, 0-1038, page.mediabox.upper_right[0], page.mediabox.upper_right[1]])

        # Add margins
        page.mediabox = RO([
            page.mediabox.lower_left[0] - margin_points,
            page.mediabox.lower_left[1] - margin_points,
            page.mediabox.upper_right[0] + margin_points,
            page.mediabox.upper_right[1] + margin_points,
        ])

        writer.add_page(page)

        # Remove temporary image PDF
        os.remove(tmp_pdf_path)

    # Write the final PDF
    prefix = os.path.basename(proj_dir).split('_')[0]
    merge_file = os.path.join(proj_dir,f"{prefix}_all_plots.pdf")
    if os.path.exists(merge_file):
        os.remove(merge_file)
    with open(merge_file, "wb") as f:
        writer.write(f)

if __name__ == '__main__':
    if os.name=='nt':
        proj_dir = os.path.dirname(os.path.dirname(__file__))
    else:
        proj_dir = os.path.dirname(__file__)

    merge_pdf(proj_dir)
    # png2pdf(proj_dir)